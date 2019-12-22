"""
    This file will contains the utils functions for graph_ops
"""
import logging

from graph_ops.graph_transformer import rewrite_einsum_expr, sort_einsum_inputs

FORMAT = '[%(asctime)-15s %(filename)s:%(lineno)s] %(message)s'

logger = logging.getLogger('optimizer')
logging.basicConfig(format=FORMAT)
logger.setLevel(logging.DEBUG)


def einsum_equal(node1, node2):
    """
    Util function to check whether two einsum nodes have the same expression. 
    Returns:
        bool variable
    """

    sort_einsum_inputs(node1)
    sort_einsum_inputs(node2)
    rewrite_einsum_expr(node1)
    rewrite_einsum_expr(node2)

    if (node1.einsum_subscripts == node2.einsum_subscripts
            and node1.inputs == node2.inputs):
        return True
    else:
        return False
