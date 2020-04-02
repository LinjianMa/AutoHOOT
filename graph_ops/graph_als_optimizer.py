"""
    This file will contains the graph transformations and optimizations for 
    alternating least squares.
"""
import logging
from functools import reduce
from operator import add
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


def generate_sequential_optiaml_tree(einsum_nodes):
    """Generates a list of nodes in-order. 
    Args: 
        einsum_node_map: a dict that maps from an output node to a input node.
        first_contract_node: a node that will be contracted before all value nodes
            in einsum_node_map.
     
    Returns:
        einsum_nodes: a list of newly generated einsum nodes.

    All the output nodes are evaluated in order and whenever an output node is
    evaluated, the mapped input node value is invalidated (hence will invalid
    any intermediate cached result that composed of that input node).
    
    Examples:
    >>> einsum_node_A = ad.einsum("bm,cm,dm->am", B, C, D)
    >>> einsum_node_B = ad.einsum("am,cm,dm->bm", A, C, D)
    >>> einsum_node_C = ad.einsum("am,bm,dm->cm", A, B, D)
    >>> dt = generate_sequential_optiaml_tree({einsum_node_A:A, 
    >>>                                        einsum_node_B:B, 
    >>>                                        einsum_node_C:C})    
    
    This will produce an intermediate node that contract (C,D).
    """
    # To collapse the inverse nodes with the same name
    all_nodes = find_topo_sort(einsum_nodes)
    dedup(*all_nodes)

    # form the input_nodes
    all_inputs = [node.inputs for node in einsum_nodes]
    all_inputs_set = set(reduce(add, all_inputs))
    complement_inputs = [
        all_inputs_set - set(node.inputs) for node in einsum_nodes
    ]

    input_nodes = []
    for (i, input_set) in enumerate(complement_inputs):
        remaining_input_nodes_set = set().union(*complement_inputs[i + 1:])
        unique_input_set = input_set - remaining_input_nodes_set

        input_nodes += list(unique_input_set - set(input_nodes))
        input_nodes += list(input_set - unique_input_set)

    # ALS sequence will only depend on the variable nodes.
    input_nodes = list(
        filter(lambda n: isinstance(n, ad.VariableNode), input_nodes))

    dt = dimension_tree(einsum_nodes, input_nodes)

    all_nodes = find_topo_sort(dt)
    with OutputInjectedMode(all_nodes):
        for node in all_nodes:
            if isinstance(node, ad.EinsumNode):
                rewrite_einsum_expr(node)
            if node.inputs != []:
                node.set_inputs(node.inputs)

    # The order of dedup and remove_transposes matters.
    # Remove transposes happen only when the inputs are same nodes.
    dedup(*dt)
    remove_transposes(find_topo_sort(dt))
    return dt


def dimension_tree(einsum_nodes, input_nodes):
    """
    Calculating einsum expressions based on the dimension tree.

    Parameters
    ----------
    einsum_nodes : list
        List of einsum nodes to be calculated based on the dimension tree.
    input_nodes : list
        List of input nodes whose contraction in the einsum_nodes obeys
        the sequence from the list end to the list start.
    first_contract_node: a node that will be contracted before all the input nodes.

    Returns
    -------
        List of einsum nodes whose results are the same as einsum_nodes,
        while obeys the dimension tree calculation sequence.
    Examples
    --------
    >>> einsum_node_A = ad.einsum("abcd,bm,cm,dm->am", X, B, C, D)
    >>> einsum_node_B = ad.einsum("abcd,am,cm,dm->bm", X, A, C, D)
    >>> einsum_node_C = ad.einsum("abcd,am,bm,dm->cm", X, A, B, D)
    >>> dt = dimension_tree([einsum_node_A, einsum_node_B, einsum_node_C], [A, B, C])
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

    return new_nodes
