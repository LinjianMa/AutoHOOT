"""
    This file will contains the graph transformations and optimizations for 
    alternating least squares.
"""
import logging
import autodiff as ad

from utils import get_all_inputs, find_topo_sort, OutputInjectedMode
from graph_ops.graph_generator import split_einsum, get_common_ancestor, generate_optimal_tree, generate_optimal_tree_w_constraint
from graph_ops.graph_dedup import dedup, remove_transposes
from graph_ops.graph_transformer import rewrite_einsum_expr
from numpy.core.einsumfunc import _parse_einsum_input

FORMAT = '[%(asctime)-15s %(filename)s:%(lineno)s] %(message)s'

logger = logging.getLogger('optimizer')
logging.basicConfig(format=FORMAT)
logger.setLevel(logging.DEBUG)


def generate_sequential_optimal_tree(einsum_nodes, input_nodes):
    """
    Regenerating einsum expressions based on the dimension tree.
    Parameters
    ----------
    einsum_nodes : list
        List of einsum nodes to be calculated based on the dimension tree.
    input_nodes : list
        List of input nodes whose contraction in the einsum_nodes obeys
        the sequence from the list end to the list start.

    Returns
    -------
        List of einsum nodes whose results are the same as einsum_nodes,
        while obeys the dimension tree calculation sequence.

    Examples
    --------
    >>> einsum_node_A = ad.einsum("abcd,bm,cm,dm->am", X, B, C, D)
    >>> einsum_node_B = ad.einsum("abcd,am,cm,dm->bm", X, A, C, D)
    >>> einsum_node_C = ad.einsum("abcd,am,bm,dm->cm", X, A, B, D)
    >>> dt = generate_sequential_optimal_tree([einsum_node_A, einsum_node_B, einsum_node_C], [A, B, C])
    >>> dt
    [ad.einsum('bm,abm->am', B, ad.einsum('cm,abcm->abm', C, ad.einsum('abcd,dm->abcm', X, D))),
    ad.einsum('am,abm->bm', A, ad.einsum('cm,abcm->abm', C, ad.einsum('abcd,dm->abcm', X, D))),
    ad.einsum('am,bm,abcm->cm', A, B, ad.einsum('abcd,dm->abcm', X, D)),
    ]
    (einsum strings may be different)
    """

    if len(einsum_nodes) == 1 and len(input_nodes) == 1:
        return einsum_nodes

    new_nodes = []
    for (i, node) in enumerate(einsum_nodes):
        contract_order = input_nodes[i + 1:]
        contract_order.reverse()
        contract_order = contract_order + input_nodes[:i]
        # get the subarray that is the inputs of node
        contract_order = list(
            filter(lambda n: n in node.inputs, contract_order))

        new_nodes.append(
            generate_optimal_tree_w_constraint(node, contract_order))

    # After generate_optimal_tree_w_constraint, some einstrs are not in the canonical format,
    # needs to rewrite again for dedup
    all_nodes = find_topo_sort(new_nodes)
    with OutputInjectedMode(all_nodes):
        for node in all_nodes:
            if isinstance(node, ad.EinsumNode):
                rewrite_einsum_expr(node)
            if node.inputs != []:
                node.set_inputs(node.inputs)

    dedup(*new_nodes)
    remove_transposes(find_topo_sort(new_nodes))
    return new_nodes
