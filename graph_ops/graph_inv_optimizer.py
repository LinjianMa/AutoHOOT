"""
    This file will contains the graph transformations and optimizations for 
    tensor inverse.
"""
import logging
import autodiff as ad

from graph_ops.graph_optimizer import UF

from numpy.core.einsumfunc import _parse_einsum_input

FORMAT = '[%(asctime)-15s %(filename)s:%(lineno)s] %(message)s'

logger = logging.getLogger('optimizer')
logging.basicConfig(format=FORMAT)
logger.setLevel(logging.DEBUG)


def inv_disjoint_sets(einsum_node, uf):
    """
    Get the disjoint sets for inverse optimization.
    """

    for node in einsum_node.inputs:
        for char in node.subscripts:
            uf.connect(node.subscripts[0], char)

    matrix_dim = int(len(einsum_node.shape) / 2)
    for i in range(matrix_dim):
        uf.connect(einsum_node.subscripts[i],
                   einsum_node.subscripts[i + matrix_dim])

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

    def generate_new_einsum(inputs, out_subs):
        new_input_subs = [node.subscripts for node in inputs]
        new_input_subs = ','.join(new_input_subs)
        new_subscripts = new_input_subs + '->' + out_subs
        new_einsum = ad.einsum(new_subscripts, *inputs)
        return new_einsum

    assert isinstance(node, ad.TensorInverseNode)
    einsum_node = node.inputs[0]
    assert isinstance(einsum_node, ad.EinsumNode)
    # einsum_node is a fused einsum
    for node in einsum_node.inputs:
        assert not isinstance(node, ad.EinsumNode)

    # TODO: currently only supports the case where
    # input nodes are different.
    in_subs, out_subs, _ = _parse_einsum_input(
        (einsum_node.einsum_subscripts, *einsum_node.inputs))
    in_subs_list = in_subs.split(',')
    whole_str = out_subs + "".join(in_subs_list)

    for i, node in enumerate(einsum_node.inputs):
        node.subscripts = in_subs_list[i]
    einsum_node.subscripts = out_subs

    uf = UF(list(whole_str))

    dsets = inv_disjoint_sets(einsum_node, uf)

    # if the node cannot be decomposed, just return
    # the input node
    if len(dsets) == 1:
        return node

    new_inputs = []
    for dset in dsets:
        input_decomp_einsum = list(
            filter(
                lambda node: not all(char not in dset
                                     for char in node.subscripts),
                einsum_node.inputs))
        out_subs = "".join(
            [char for char in einsum_node.subscripts if char in dset])

        decomp_einsum = generate_new_einsum(input_decomp_einsum, out_subs)
        decomp_einsum.set_in_indices_length(int(len(out_subs) / 2))
        inv_node = ad.tensorinv(decomp_einsum)
        inv_node.subscripts = out_subs

        new_inputs.append(inv_node)

    return generate_new_einsum(new_inputs, einsum_node.subscripts)
