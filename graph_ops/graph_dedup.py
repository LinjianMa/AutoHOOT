import autodiff as ad
import copy, logging
import numpy as np
import itertools

from utils import find_topo_sort, OutputInjectedModeP, replace_node, PseudoNode
from utils import find_topo_sort_p
from numpy.core.einsumfunc import _parse_einsum_input
from collections import defaultdict

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
    [connected_dims]:[output index].

    connected_dims: tuple of input node dims info connected by one char.
        Each element in the tuple is a DimInfo object.

    When the list of dims is connected by a contraction char, the output index will be -1.
    """
    from graph_ops.graph_transformer import generate_einsum_info

    def sort_hash(dim_info):
        return dim_info.name

    node_copy = copy.deepcopy(node)
    # this is used to normalize the output node name
    temp_out_name = "_temp_einsum"
    node_copy.name = temp_out_name

    uf, _, _ = generate_einsum_info(node_copy)
    # each set contains the dimension names connected by one char
    dset = uf.disjoint_set()
    dset_ret = {}

    for connected_dims in dset:
        if not any(temp_out_name == dim_info.node_name
                   for dim_info in connected_dims):
            # contracted char
            dset_ret[tuple(sorted(connected_dims, key=sort_hash))] = -1
        else:
            output_dim_info, = filter(
                lambda dim_info: dim_info.node_name == temp_out_name,
                connected_dims)
            # uncontracted char
            connected_dims_ret = tuple(
                sorted(list(
                    filter(
                        lambda dim_info: dim_info.node_name != temp_out_name,
                        connected_dims)),
                       key=sort_hash))
            dset_ret[connected_dims_ret] = output_dim_info.dim_index

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
    def equivalent_dims(dims_A, dims_B):
        """
        Check if the two dimension info tuples are equivalent under the symmetry constraint.
        Return True if equivalent and return False otherwise.
        """
        for dim_A_info in dims_A:
            inode, dim = dim_A_info.node, dim_A_info.dim_index
            symmetric_dims = [dim]
            for s in inode.symmetry:
                if dim in s:
                    symmetric_dims = s
                    break

            search_list = [(dim_B_info, d) for dim_B_info in dims_B
                           for d in symmetric_dims]
            if not any((inode == dim_B_info.node and dim_B_info.dim_index == d)
                       for (dim_B_info, d) in search_list):
                return False
        return True

    def equivalent_list_dims(list_dims_A, list_dims_B):
        """
        Check if all the elements in the two lists are equivalent.
        """
        for (dims_A, dims_B) in zip(list_dims_A, list_dims_B):
            if not equivalent_dims(dims_A, dims_B):
                return False
        return True

    if not isinstance(A, ad.EinsumNode) or not isinstance(B, ad.EinsumNode):
        logger.info(f"Cannot collapse {A} and {B}")
        return
    if A.inputs != B.inputs:
        logger.info(f"Cannot collapse {A} and {B}")
        return
    if len(A.shape) != len(B.shape):
        logger.info(f"Cannot collapse {A} and {B}")
        return

    dims_dict_A = get_disjoint_set(A)
    dims_dict_B = get_disjoint_set(B)

    uncontracted_list_A = [
        key for key, value in dims_dict_A.items() if value != -1
    ]
    uncontracted_list_B = [
        key for key, value in dims_dict_B.items() if value != -1
    ]

    contracted_set_A = set(dims_dict_A) - set(uncontracted_list_A)
    contracted_set_B = set(dims_dict_B) - set(uncontracted_list_B)

    if not equivalent_list_dims(uncontracted_list_A, uncontracted_list_B):
        logger.info(f"Cannot collapse {A} and {B}")
        return

    # The connected_dims for the contracted dimensions are unordered.
    # A and B are equivalent if one of the permutations are equivalent.
    for contracted_set_B_permute in set(
            itertools.permutations(contracted_set_B)):

        if equivalent_list_dims(contracted_set_A, contracted_set_B_permute):
            # One of the permutations is equivalent
            A.einsum_subscripts = B.einsum_subscripts
            A.set_inputs(B.inputs)
            return

    logger.info(f"Cannot collapse {A} and {B}")
    return


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

    # change the keys to string format for later equality check
    dimname_func = lambda x: f"{x.node_name}-{x.dim_index}"
    dset_A = [("|".join(map(dimname_func, key)), value)
              for key, value in dset_A.items()]
    dset_B = [("|".join(map(dimname_func, key)), value)
              for key, value in dset_B.items()]
    dset_A_keys, dset_A_vals = zip(*dset_A)
    dset_B_keys, dset_B_vals = zip(*dset_B)

    if sorted(dset_A_keys) != sorted(dset_B_keys):
        return None

    # Check if all the contracted char refers to the same dim.
    for i, key in enumerate(dset_A_keys):
        if (dset_A_vals[i] == -1
                and dset_B_vals[dset_B_keys.index(key)] != -1):
            return None

    argsort_A = np.argsort(dset_A_keys[:len(A.shape)])
    argsort_B = np.argsort(dset_B_keys[:len(A.shape)])

    transpose_indices = [0 for _ in range(len(A.shape))]
    for index_A, index_B in zip(argsort_A, argsort_B):
        transpose_indices[index_B] = index_A

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
