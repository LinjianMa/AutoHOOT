"""
    This file will contains the graph transformations and optimizations for 
    tensor inverse.
"""
import logging
import autodiff as ad
from utils import PseudoNode

from graph_ops.graph_optimizer import UF

from numpy.core.einsumfunc import _parse_einsum_input

FORMAT = '[%(asctime)-15s %(filename)s:%(lineno)s] %(message)s'

logger = logging.getLogger('optimizer')
logging.basicConfig(format=FORMAT)
logger.setLevel(logging.DEBUG)


def find_one_subset_sum(arr, n, sum, indices=[]):
    """
    Find one subset of the input array with given sum.

    Parameters
    ----------
    arr: The input array.
    n: The cursor position. n should be int within [0, len(arr)].
    sum: int, the given sum of the sebset.
    indices: the input indices.

    Returns
    -------
    A list representing the indices of the subset.
    """

    if sum == 0 and indices != []:
        return indices
    if n == 0:
        return []

    new_indices = find_one_subset_sum(arr, n - 1, sum, indices)
    if new_indices != []:
        return new_indices

    new_indices = indices.copy()
    new_indices.append(n - 1)
    return find_one_subset_sum(arr, n - 1, sum - arr[n - 1], new_indices)


def inv_disjoint_sets(p_einsum_node, p_in_nodes, uf):
    """
    Get the disjoint sets for inverse optimization.

    Parameters
    ----------
    p_einsum_node: The PseudoNode for the einsum node (inv node input).
    p_in_nodes: The PseudoNode of inputs for the einsum node.
    uf: The union find class for subscripts of the einsum node.

    Returns
    -------
    The disjoint sets of the subscripts. Each set represents part of the subscripts
    that can be decomposable from other parts for the inverse.

    """

    for node in p_in_nodes:
        for char in node.subscript:
            uf.connect(node.subscript[0], char)

    matrix_dim = int(len(p_einsum_node.node.shape) / 2)

    # For each dim in the matrix col, connect it to the corresponding dim
    # in the matrix row.
    for i in range(matrix_dim):
        uf.connect(p_einsum_node.subscript[i],
                   p_einsum_node.subscript[i + matrix_dim])

    # For the case when the size of one dim of the matrix col is different with the
    # correspoding dim size of the matrix row, we connect it with some other dims, such
    # that for each decomposable set, the corresponding matrix is squared.
    shape_diff = [
        p_einsum_node.node.shape[i] - p_einsum_node.node.shape[i + matrix_dim]
        for i in range(matrix_dim)
    ]
    shape_unequal_tuple = [(i, v) for (i, v) in enumerate(shape_diff)
                           if v != 0]

    while shape_unequal_tuple != []:

        _, shape_diff = list(zip(*shape_unequal_tuple))
        indices = find_one_subset_sum(shape_diff, len(shape_diff), 0)

        subset = [
            v for (i, v) in enumerate(shape_unequal_tuple) if i in indices
        ]
        shape_unequal_tuple = [
            v for (i, v) in enumerate(shape_unequal_tuple) if i not in indices
        ]

        # connect subscripts in the subset
        indices, _ = list(zip(*subset))
        for i in indices:
            uf.connect(p_einsum_node.subscript[i],
                       p_einsum_node.subscript[indices[0]])

    return uf.disjoint_set()


def optimize_inverse(inv_node):
    """
    Optimize the inverse of an einsum expression, such that
    inverse is operated on several smaller tensors.

    Parameters
    ----------
    node: The inverse of a fused einsum node

    Returns
    -------
    If the input node cannot be optimized, then return the input node.
    If it can be optimized, return the optimized node.

    """
    def generate_new_einsum(p_inputs, out_subs):
        new_input_subs = [node.subscript for node in p_inputs]
        new_input_subs = ','.join(new_input_subs)
        new_subscripts = new_input_subs + '->' + out_subs
        inputs = [p_node.node for p_node in p_inputs]
        new_einsum = ad.einsum(new_subscripts, *inputs)
        return new_einsum

    # Note: currently, the optimization algorithm only works for the
    # case when the matrix row and column has same number of dimension.
    if inv_node.input_indices_length * 2 != len(inv_node.shape):
        return inv_node

    assert isinstance(inv_node, ad.TensorInverseNode)
    einsum_node = inv_node.inputs[0]
    assert isinstance(einsum_node, ad.EinsumNode)
    # einsum_node is a fused einsum
    for node in einsum_node.inputs:
        assert not isinstance(node, ad.EinsumNode)

    in_subs, out_subs, _ = _parse_einsum_input(
        (einsum_node.einsum_subscripts, *einsum_node.inputs))
    in_subs_list = in_subs.split(',')
    whole_str = out_subs + "".join(in_subs_list)

    uf = UF(list(whole_str))

    p_einsum_node = PseudoNode(node=einsum_node, subscript=out_subs)
    p_in_nodes = []
    for i, node in enumerate(einsum_node.inputs):
        p_in_nodes.append(PseudoNode(node=node, subscript=in_subs_list[i]))

    dsets = inv_disjoint_sets(p_einsum_node, p_in_nodes, uf)

    # if the node cannot be decomposed, just return
    # the input node
    if len(dsets) == 1:
        return inv_node

    new_inputs = []
    for dset in dsets:
        input_decomp_einsum = list(
            filter(
                lambda node: not all(char not in dset
                                     for char in node.subscript), p_in_nodes))
        out_subs = "".join(
            [char for char in p_einsum_node.subscript if char in dset])

        decomp_einsum = generate_new_einsum(input_decomp_einsum, out_subs)
        decomp_einsum.set_in_indices_length(int(len(out_subs) / 2))
        inv_node = PseudoNode(node=ad.tensorinv(decomp_einsum),
                              subscript=out_subs)
        new_inputs.append(inv_node)

    return generate_new_einsum(new_inputs, p_einsum_node.subscript)
