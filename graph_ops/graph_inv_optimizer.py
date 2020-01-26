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
    for i in range(matrix_dim):
        uf.connect(p_einsum_node.subscript[i],
                   p_einsum_node.subscript[i + matrix_dim])

    return uf.disjoint_set()


def optimize_inverse(node):
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

    assert isinstance(node, ad.TensorInverseNode)
    einsum_node = node.inputs[0]
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
        return node

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
