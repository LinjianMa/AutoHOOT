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


def generate_new_einsum(p_inputs, out_subs):
    new_input_subs = [node.subscript for node in p_inputs]
    new_input_subs = ','.join(new_input_subs)
    new_subscripts = new_input_subs + '->' + out_subs
    inputs = [p_node.node for p_node in p_inputs]
    new_einsum = ad.einsum(new_subscripts, *inputs)
    return new_einsum


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

    if isinstance(input_node, ad.AddNode) and (input_node.inputs[0].name
                                               == input_node.inputs[1].name):

        inverse_node = optimize_inverse(ad.tensorinv(input_node.inputs[0]))
        subscript = "".join(
            [chr(ord('a') + i) for i in range(len(inverse_node.shape))])
        return ad.einsum(f",{subscript}->{subscript}", ad.ScalarNode(0.5),
                         inverse_node)

    return inv_node


def prune_single_inv_node(einsum_node, inv_node):
    """
    Prune the inv_node in the einsum node if condition mets.

    Note:
    1. can only optimize the node when the input of inv is an einsum node.
    2. only supports the case when the splitted nodes are different from the remaining ones.
        For example: ad.einsum("ab,bc,cd,de->ae", inv("ab,bc->ac", A, B), A, B, C) will be
        optimzied to ad.einsum("ab,bc->ac", C, ad.identity()),
        but we cannot optimize ad.einsum("ab,bc,cd,de->ae", inv("ab,bc->ac", A, B), A, B, B).

    Parameters
    ----------
    einsum_node: The fused einsum node
    inv_node: the input inv node to be pruned

    Returns
    -------
    If the einsum_node cannot be optimized, then return the input einsum_node.
    If it can be optimized, return the optimized einsum node.

    """
    from graph_ops.graph_transformer import rewrite_einsum_expr
    from graph_ops.graph_generator import split_einsum

    inv_node_input = inv_node.inputs[0]
    if not isinstance(inv_node_input, ad.EinsumNode):
        logger.info(f"inv input is not einsum node, can't prune inv")
        return einsum_node

    if not set(inv_node_input.inputs).issubset(set(einsum_node.inputs)):
        logger.info(
            f"inv inputs is not subset of einsum node inputs, can't prune inv")
        return einsum_node

    einsum_inputs_in_inv = [
        n for n in einsum_node.inputs if n in inv_node_input.inputs
    ]
    if len(einsum_inputs_in_inv) < len(inv_node_input.inputs):
        logger.info(
            f"number of inv inputs is more than corresponding einsum inputs, can't prune inv"
        )
        return einsum_node

    split_einsum_node = split_einsum(
        einsum_node,
        list(set(einsum_node.inputs) - set(inv_node_input.inputs)))

    # Assign pseudo nodes and chars
    in_subs, out_subs, _ = _parse_einsum_input(
        (split_einsum_node.einsum_subscripts, *split_einsum_node.inputs))
    in_subs_list = in_subs.split(',')

    updated_p_in_nodes = []
    for i, node in enumerate(split_einsum_node.inputs):
        if isinstance(node, ad.EinsumNode):
            p_einsum_input = PseudoNode(node=node, subscript=in_subs_list[i])
        elif node is inv_node:
            p_inv_input = PseudoNode(node=node, subscript=in_subs_list[i])
        else:
            updated_p_in_nodes.append(
                PseudoNode(node=node, subscript=in_subs_list[i]))

    contract_char = "".join(
        set(p_einsum_input.subscript) & set(p_inv_input.subscript))
    uncontract_str = "".join(
        set("".join([p_einsum_input.subscript, p_inv_input.subscript])) -
        set(contract_char))

    if not (len(p_einsum_input.subscript) == 2 and len(p_inv_input.subscript)
            == 2 and len(contract_char) == 1 and len(uncontract_str) == 2):
        # this is not a matmul. Just return the initial node
        logger.info(
            f"the op between inv input and the selected einsum is not matmul, can't prune inv"
        )
        return einsum_node

    if p_einsum_input.subscript[0] == p_inv_input.subscript[
            0] or p_einsum_input.subscript[1] == p_inv_input.subscript[1]:
        # the str is like "ab,ac", and one einsum needs to be transposed to compare
        p_in_subs, p_out_subs, _ = _parse_einsum_input(
            (p_einsum_input.node.einsum_subscripts,
             *p_einsum_input.node.inputs))
        einsum_input = ad.einsum(
            f"{p_in_subs}->{p_out_subs[1]}{p_out_subs[0]}",
            *p_einsum_input.node.inputs)
    else:
        einsum_input = p_einsum_input.node

    rewrite_einsum_expr(einsum_input)
    rewrite_einsum_expr(inv_node_input)

    if einsum_input.name != inv_node_input.name:
        logger.info(
            f"inv input and the selected einsum have different expressions, can't prune inv"
        )
        return einsum_node

    # prune the inv node
    updated_p_in_nodes = updated_p_in_nodes + [
        PseudoNode(node=ad.identity(inv_node_input.shape[0]),
                   subscript=uncontract_str)
    ]

    return generate_new_einsum(updated_p_in_nodes, out_subs)


def prune_inv_node(einsum_node):
    """
    Prune input inv nodes in the einsum node if following two conditions met:
        1. inv(A) @ A or similar structures in the einsum
        2. inv(A) where A is identity node

    Parameters
    ----------
    einsum_node: The fused einsum node

    Returns
    -------
    If the einsum_node cannot be optimized, then return the input einsum_node.
    If it can be optimized, return the optimized einsum node.

    """
    assert isinstance(einsum_node, ad.EinsumNode)
    for node in einsum_node.inputs:
        assert not isinstance(node, ad.EinsumNode)

    # condition 1
    inv_inputs_list = list(
        filter(lambda node: isinstance(node, ad.TensorInverseNode),
               einsum_node.inputs))
    if len(inv_inputs_list) == 0:
        logger.info(f"No inv nodes in the inputs, can't prune inv")
        return einsum_node
    for inv_node in inv_inputs_list:
        einsum_node = prune_single_inv_node(einsum_node, inv_node)

    # condition 2
    new_inputs = einsum_node.inputs
    for i, innode in enumerate(new_inputs):
        if isinstance(innode, ad.TensorInverseNode) and isinstance(
                innode.inputs[0], ad.IdentityNode):
            new_inputs[i] = innode.inputs[0]
    einsum_node.set_inputs(new_inputs)

    return einsum_node
