import autodiff as ad
from utils import find_topo_sort, OutputInjectedModeP, replace_node, PseudoNode
from utils import find_topo_sort_p
from numpy.core.einsumfunc import _parse_einsum_input
from collections import defaultdict
import copy, logging
import numpy as np

FORMAT = '[%(asctime)-15s %(filename)s:%(lineno)s] %(message)s'

logger = logging.getLogger('optimizer')
logging.basicConfig(format=FORMAT)
logger.setLevel(logging.DEBUG)


def dedup(*nodes):
    """Remove the duplicate nodes with same name.
    Args:
        nodes: One or many nodes.
    """
    assert len(nodes) > 0

    topo_order = find_topo_sort_p([PseudoNode(n) for n in nodes])
    with OutputInjectedModeP(topo_order):
        unique_nodes_map = {}
        unique_nodes = set()
        # Use the last occurrence.
        for ptmp in topo_order:
            tmp = ptmp.node
            unique_nodes_map[tmp.name] = tmp
        unique_nodes = set(unique_nodes_map.values())

        for ptmp in topo_order:
            tmp = ptmp.node
            if tmp not in unique_nodes:
                unique_copy = unique_nodes_map[tmp.name]
                replace_node(ptmp, unique_copy)


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


def get_disjoint_set(node):
    """
    Get the disjoint set information of the input einsum node.

    The returned set element has the following [key]:[value] structure:
    ["|".join(connected_dims)]:[output index].

    connected_dims: list of input node dims connected by one char.
    Each element in the connected_dims is a string,
        formatted as f'{nodename}-{node order in einsum}-{dim number}'.

    When the list of dims is connected by a contraction char, the output index will be -1.

    Examples
    --------
    >>> node_A = ad.einsum('bec,ca->abe',input_tensor,C)
    >>> get_disjoint_set(node_A)
    {'C-1-1': 0, 'input_tensor-0-0': 1, 'input_tensor-0-1': 2, 'C-1-0|input_tensor-0-2': -1}
    """
    from graph_ops.graph_transformer import generate_einsum_info

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

    # Note: the value list of dset_ret will have an order of 0, ..., len(node.shape), -1.
    # It is determined because of the routine in generate_einsum_info.
    return dset_ret


def collapse_symmetric_expr(A, B):
    """
    Inplace replace A's inputs with B's if allowed by symmetry constraints.

    Note: we assume that inputs can be collapsed only if they have the same input
        list (input order is the same).

    Parameters
    ----------
    A, B : einsum nodes.
    """
    if not isinstance(A, ad.EinsumNode) or not isinstance(B, ad.EinsumNode):
        logger.info(f"Cannot collapse {A} and {B}")
        return
    if A.inputs != B.inputs:
        logger.info(f"Cannot collapse {A} and {B}")
        return
    if len(A.shape) != len(B.shape):
        logger.info(f"Cannot collapse {A} and {B}")
        return

    dset_A = get_disjoint_set(A)
    dset_B = get_disjoint_set(B)

    for (key_A, key_B) in zip(dset_A.keys(), dset_B.keys()):
        connected_dims_str_A = key_A.split("|")
        connected_dims_str_B = key_B.split("|")

        connected_dims_A = []
        for dim_A_str in connected_dims_str_A:
            dim_A_info = dim_A_str.split("-")
            inode, dim = A.inputs[int(dim_A_info[1])], int(dim_A_info[2])
            connected_dims_A.append((inode, dim))

        connected_dims_B = []
        for dim_B_str in connected_dims_str_B:
            dim_B_info = dim_B_str.split("-")
            inode, dim = B.inputs[int(dim_B_info[1])], int(dim_B_info[2])
            connected_dims_B.append((inode, dim))

        for (inode, dim) in connected_dims_A:
            symmetric_dims = [dim]
            for s in inode.symmetry:
                if dim in s:
                    symmetric_dims = s
            if any((inode, d) in connected_dims_B for d in symmetric_dims):
                continue
            else:
                logger.info(f"Cannot collapse {A} and {B}")
                return

    A.einsum_subscripts = B.einsum_subscripts
    A.set_inputs(B.inputs)


def get_transpose_indices(A, B):
    """
    If two nodes are transposable, then get the indices such that transposing
    one of the node with the specific indices will get same nodes.

    Two nodes are transposable if transposing one node's output tensor indices
    will get the same tensor as the second node.

    Note: we assume that inputs are transposable only if they have the same input
        list (input order is the same).

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
    if not isinstance(A, ad.EinsumNode) or not isinstance(B, ad.EinsumNode):
        return None
    if A.inputs != B.inputs:
        return None
    if len(A.shape) != len(B.shape):
        return None

    dset_A = get_disjoint_set(A)
    dset_B = get_disjoint_set(B)

    if dset_A.keys() != dset_B.keys():
        return None
    # Check if all the contracted char refers to the same dim.
    for key in dset_A.keys():
        if (dset_A[key] == -1 and dset_B[key] != dset_A[key]):
            return None

    inv_dset_B = dict(zip(dset_B.values(), dset_B.keys()))
    transpose_indices = [dset_A[inv_dset_B[i]] for i in range(len(B.shape))]

    # If the transpose indices is sorted ascendingly. There is no transpose.
    if transpose_indices == sorted(transpose_indices):
        return None

    return transpose_indices


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

    with OutputInjectedModeP([PseudoNode(n) for n in graph]):
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
