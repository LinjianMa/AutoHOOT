"""
    This file will contains the graph transformations and optimizations for 
    tensor inverse.
"""
import logging
import numpy as np
import autodiff as ad
from utils import PseudoNode

from graph_ops.graph_optimizer import UF

from numpy.core.einsumfunc import _parse_einsum_input

FORMAT = '[%(asctime)-15s %(filename)s:%(lineno)s] %(message)s'

logger = logging.getLogger('optimizer')
logging.basicConfig(format=FORMAT)
logger.setLevel(logging.DEBUG)


def inv_disjoint_sets(p_einsum_node, p_in_nodes):
    """
    Get the disjoint sets for inverse optimization.

    Parameters
    ----------
    p_einsum_node: The PseudoNode for the einsum node (inv node input).
    p_in_nodes: The PseudoNode of inputs for the einsum node.

    Returns
    -------
    The disjoint sets of the subscripts. Each set represents part of the subscripts
    that can be decomposable from other parts for the inverse.

    """
    whole_str = p_einsum_node.subscript + "".join(
        [pnode.subscript for pnode in p_in_nodes])
    uf = UF(list(whole_str))

    # Each input node must be in its own disjoint set
    for node in p_in_nodes:
        for char in node.subscript:
            uf.connect(node.subscript[0], char)

    matrix_dim = int(len(p_einsum_node.node.shape) / 2)

    # For each dim in the matrix col, connect it to the corresponding dim in the matrix row.
    for i in range(matrix_dim):
        uf.connect(p_einsum_node.subscript[i],
                   p_einsum_node.subscript[i + matrix_dim])

    return uf.disjoint_set()


def split_inv_einsum(inv_node):
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

    einsum_node = inv_node.inputs[0]
    assert isinstance(einsum_node, ad.EinsumNode)
    # einsum_node is a fused einsum
    for node in einsum_node.inputs:
        assert not isinstance(node, ad.EinsumNode)

    in_subs, out_subs, _ = _parse_einsum_input(
        (einsum_node.einsum_subscripts, *einsum_node.inputs))
    in_subs_list = in_subs.split(',')

    p_einsum_node = PseudoNode(node=einsum_node, subscript=out_subs)
    p_in_nodes = []
    for i, node in enumerate(einsum_node.inputs):
        p_in_nodes.append(PseudoNode(node=node, subscript=in_subs_list[i]))

    dsets = inv_disjoint_sets(p_einsum_node, p_in_nodes)

    # If the node cannot be decomposed, just return the input node
    if len(dsets) == 1:
        return inv_node

    new_inputs = []
    for dset in dsets:
        input_decomp_einsum = list(
            filter(lambda node: any(char in dset for char in node.subscript),
                   p_in_nodes))
        out_subs = "".join(
            [char for char in p_einsum_node.subscript if char in dset])

        decomp_node = generate_new_einsum(input_decomp_einsum, out_subs)

        decomp_node.set_in_indices_length(int(len(out_subs) / 2))

        input_node = PseudoNode(node=ad.tensorinv(decomp_node),
                                subscript=out_subs)
        new_inputs.append(input_node)

    return generate_new_einsum(new_inputs, p_einsum_node.subscript)


def optimize_inverse(inv_node):
    """
    Optimize the inverse of an einsum expression.

    Parameters
    ----------
    node: The inverse of a fused einsum node

    Returns
    -------
    If the input node cannot be optimized, then return the input node.
    If it can be optimized, return the optimized node.

    """

    assert isinstance(inv_node, ad.TensorInverseNode)
    # Note: currently, the optimization algorithm only works when
    # 1. the matrix row and column has same number of dimension,
    # 2. the matrix is square,
    # 3. each corresponding dimension in row and column has the same size.
    if inv_node.input_indices_length * 2 != len(inv_node.shape):
        logger.info(f"Dimension length doesn't agree, can't optimize inverse")
        return inv_node
    matrix_dim = int(len(inv_node.shape) / 2)

    assert np.prod(inv_node.shape[:matrix_dim]) == np.prod(
        inv_node.shape[matrix_dim:])

    shape_diff_list = [
        inv_node.shape[i] - inv_node.shape[i + matrix_dim]
        for i in range(matrix_dim)
    ]
    if any(shape_diff != 0 for shape_diff in shape_diff_list):
        logger.info(
            f"Each corresponding dimension in row and column doesn't have the same size, can't optimize inverse"
        )
        return inv_node

    input_node = inv_node.inputs[0]

    if isinstance(input_node, ad.EinsumNode):
        return split_inv_einsum(inv_node)

    if isinstance(input_node, ad.AddNode) and (
            input_node.inputs[0].name == input_node.inputs[1].name):

        inverse_node = optimize_inverse(ad.tensorinv(input_node.inputs[0]))
        subscript = "".join(
            [chr(ord('a') + i) for i in range(len(inverse_node.shape))])
        return ad.einsum(f",{subscript}->{subscript}", ad.ScalarNode(0.5),
                         inverse_node)

    return inv_node
