import autodiff as ad
from utils import find_topo_sort, OutputInjectedMode, replace_node
from numpy.core.einsumfunc import _parse_einsum_input
from collections import defaultdict
import copy
import numpy as np


def dedup(*nodes):
    """Remove the duplicate nodes with same name.
    Args:
        nodes: One or many nodes.
    """
    assert len(nodes) > 0

    topo_order = find_topo_sort(nodes)
    with OutputInjectedMode(topo_order):
        unique_nodes_map = {}
        unique_nodes = set()
        # Use the last occurrence.
        for tmp in topo_order:
            unique_nodes_map[tmp.name] = tmp
        unique_nodes = set(unique_nodes_map.values())

        for tmp in topo_order:
            if tmp not in unique_nodes:
                unique_copy = unique_nodes_map[tmp.name]
                replace_node(tmp, unique_copy)


def declone(o_node):
    """
    Args:
        o_node: An output node.
    Returns:
        o_node: A new node with new name.
    """

    if isinstance(o_node, ad.VariableNode):
        return o_node
    if isinstance(o_node, ad.CloneNode):
        assert len(o_node.inputs) == 1
        return declone(o_node.inputs[0])

    new_inputs = []
    for i_node in o_node.inputs:
        i_node = declone(i_node)
        new_inputs.append(i_node)
    o_node.set_inputs(new_inputs)
    return o_node


def transpose_equivalent(A, B):
    """
    Two nodes are transpose equivalent if following meets:
    1. The contract dims are referring to the same dim.
    2. others inputs dim are permutation of each other.
    
    Args:
        A, B: einsum nodes.
    Returns:
        True/False

    Assume already through rewrite_einsum, a.k.a inputs are sorted.
    """
    if A.inputs != B.inputs:
        return False

    def get_contract_node_dims(node):
        # Returns a nested map.
        # Contract_char -> node -> index
        # e.g {'c': {A: 0, B:1}}
        ret = defaultdict(dict)
        in_subs, out_subs, _ = _parse_einsum_input(
            (node.einsum_subscripts, *node.inputs))
        in_subs_list = in_subs.split(',')
        contract_dims = set("".join(in_subs_list)) - set(out_subs)
        for contract_dim in contract_dims:
            for inode, in_sub in zip(node.inputs, in_subs_list):
                innode_index = in_sub.index(contract_dim)
                ret[contract_dim][inode] = innode_index
        return ret

    def get_node_chars(node):
        # Returns a map:
        # node -> indices
        # e.g {A: set('abc'), B: set('cd')}
        in_subs, out_subs, _ = _parse_einsum_input(
            (node.einsum_subscripts, *node.inputs))
        in_subs_list = in_subs.split(',')
        in_subs_list = [set(isl) for isl in in_subs_list]
        return dict(zip(node.inputs, (in_subs_list)))

    # Check if all the contracted char refers to the same dim.
    if (get_contract_node_dims(A)) != get_contract_node_dims(B):
        return False

    # Check if nodes contain the same chars.
    if get_node_chars(A) != get_node_chars(B):
        return False

    return True


def get_transpose_indices(A, B):
    """
    If two nodes are transposable, then get the indices such that transposing
    one of the node with the specific indices will get same nodes.

    Two nodes are transposable if transposing one node's output tensor indices
    will get the same tensor as the second node.

    Parameters
    ----------
    A, B : einsum nodes.

    Returns
    -------
        Return None if these nodes are not transposable, return a list of tranpose indices elsewise.

    Examples
    --------
    >>> node_A = ad.einsum('bec,ca->abe',input_tensor,C)
    >>> node_B = ad.einsum('ebc,ca->abe',input_tensor,C)
    >>> get_transpose_indices(node_A, node_B)
    [1, 0, 2]
    """
    from graph_ops.graph_transformer import generate_einsum_info

    if not isinstance(A, ad.EinsumNode) or not isinstance(B, ad.EinsumNode):
        return None
    if A.inputs != B.inputs:
        return None
    if len(A.shape) != len(B.shape):
        return None

    def get_disjoint_set(node):
        """
        The returned set element has the following [key]:[value] structure:
        ["".join(connected_dims)]:[output index].

        connected_dims: list of input node dims connected by one char.
        Each element in the connected_dims is a string,
            formatted as f'{nodename}-{node order in einsum}-{dim number}'.

        When the list of dims is connected by a contraction char, the output index will be -1.

        Example:
        For the example above, the dset_ret for node_A will be:
        {'C-1-1': 0, 'input_tensor-0-0': 1, 'input_tensor-0-1': 2, 'C-1-0input_tensor-0-2': -1}
        """
        node_copy = copy.deepcopy(node)
        # this is used to normalize the output node name
        temp_out_name = "_temp_einsum"
        node_copy.name = temp_out_name

        uf, _, _ = generate_einsum_info(node_copy)
        # each set contains the dimension names connected by one char
        dset = uf.disjoint_set()
        dset_ret = {}

        for connected_dims in dset:
            if not any(temp_out_name in name for name in connected_dims):
                # contracted char
                connect_dims_str = "|".join(sorted(list(connected_dims)))
                dset_ret[connect_dims_str] = -1
            else:
                # uncontracted char
                for name in connected_dims:
                    if temp_out_name in name:
                        connect_dims_str = "|".join(
                            sorted(
                                list(
                                    filter(lambda dim: dim != name,
                                           connected_dims))))
                        # get the output dim
                        dset_ret[connect_dims_str] = int(
                            name.replace(f'{temp_out_name}-', ''))
                        break
        return dset_ret

    dset_A = get_disjoint_set(A)
    dset_B = get_disjoint_set(B)

    if dset_A.keys() != dset_B.keys():
        return None
    # Check if all the contracted char refers to the same dim.
    for key in dset_A.keys():
        if (dset_A[key] == -1 and dset_B[key] != dset_A[key]):
            return None

    # indices_A(B) is an array stores the output dimension index ordered by the sorted keys.
    # the goal is two find the reorder of indices_A, such that it equals indices_B.
    indices_A = []
    indices_B = []
    for key in sorted(dset_A.keys()):
        if dset_A[key] != -1:
            indices_A.append(dset_A[key])
            indices_B.append(dset_B[key])

    transpose_indices = [indices_A.index(elem) for elem in indices_B]
    # If the transpose indices is sorted ascendingly. There is no transpose.
    if transpose_indices == sorted(transpose_indices):
        return None

    return trans_indices


def dedup_transpose(graph, node, trans_node, trans_indices):
    """
    Replace the node with the trans_node, and change its output nodes in graph accordingly.

    Parameters
    ----------
    graph: list of nodes denoting a connected graph.
    node: node to be replaced.
    trans_node: the transposed node that will replace node.
    trans_indices: the transpose indices.
    """
    assert node in graph
    assert trans_node in graph

    with OutputInjectedMode(graph):
        for onode in node.outputs:
            # NOTE: currently we cannot deal with non-einsum nodes.
            assert isinstance(onode, ad.EinsumNode)
            in_subs, out_subs, _ = _parse_einsum_input(
                (onode.einsum_subscripts, *onode.inputs))
            in_subs_list = in_subs.split(',')
            for (i, n) in enumerate(onode.inputs):
                if n is node:
                    onode.inputs[i] = trans_node
                    str_list = list(in_subs_list[i])
                    in_subs_list[i] = "".join(
                        [str_list[j] for j in trans_indices])

            new_subscripts = ",".join(in_subs_list) + "->" + out_subs
            onode.einsum_subscripts = new_subscripts
            onode.set_inputs(onode.inputs)


def remove_transposes(topo_list):
    for i in range(len(topo_list)):
        for j in range(i):
            trans_indices = get_transpose_indices(topo_list[i], topo_list[j])
            if trans_indices != None:
                dedup_transpose(topo_list, topo_list[i], topo_list[j],
                                trans_indices)
