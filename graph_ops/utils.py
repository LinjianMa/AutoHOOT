"""
    This file will contains the utils functions for graph_ops
"""
import logging
from copy import deepcopy
from graph_ops.graph_transformer import rewrite_einsum_expr

FORMAT = '[%(asctime)-15s %(filename)s:%(lineno)s] %(message)s'

logger = logging.getLogger('optimizer')
logging.basicConfig(format=FORMAT)
logger.setLevel(logging.DEBUG)


def einsum_equal(node1, node2):
    """
    Util function to check whether two einsum nodes have the same expression.
    Note that the original nodes are not modified in the function.
    Returns:
        bool variable
    """

    node1_copy = deepcopy(node1)
    node2_copy = deepcopy(node2)
    rewrite_einsum_expr(node1_copy)
    rewrite_einsum_expr(node2_copy)

    input_nodes_1 = sorted(node1.inputs,
                           key=lambda input_node: input_node.name)
    input_nodes_2 = sorted(node2.inputs,
                           key=lambda input_node: input_node.name)

    if (node1_copy.einsum_subscripts == node2_copy.einsum_subscripts
            and input_nodes_1 == input_nodes_2):
        return True
    else:
        return False
