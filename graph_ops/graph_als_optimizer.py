"""
    This file will contains the graph transformations and optimizations for 
    alternating least squares.
"""
import logging
import autodiff as ad
from utils import get_all_inputs, find_topo_sort, OutputInjectedMode

from graph_ops.graph_generator import split_einsum, get_common_ancestor, generate_optimal_tree
from graph_ops.graph_dedup import dedup, remove_transposes
from graph_ops.graph_transformer import rewrite_einsum_expr

from numpy.core.einsumfunc import _parse_einsum_input

FORMAT = '[%(asctime)-15s %(filename)s:%(lineno)s] %(message)s'

logger = logging.getLogger('optimizer')
logging.basicConfig(format=FORMAT)
logger.setLevel(logging.DEBUG)


def generate_sequential_optiaml_tree(einsum_node_map={},
                                     first_contract_node=None):
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
    dt = dimension_tree(list(einsum_node_map.keys()),
                        list(einsum_node_map.values()), first_contract_node)

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


def dimension_tree(einsum_nodes, input_nodes, first_contract_node=None):
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

    if first_contract_node == None:
        # if first_contract_node is none, the right most tree will not be reused.
        return dimension_tree(einsum_nodes[:-1], input_nodes[:-1],
                              input_nodes[-1]) + [einsum_nodes[-1]]

    assert first_contract_node not in input_nodes

    second_einsums = []
    for einsum_node in einsum_nodes:
        input_node_subset = list(set(input_nodes) & set(einsum_node.inputs))
        input_node_subset = sorted(input_node_subset,
                                   key=lambda node: node.name)

        splitted_einsum, = [
            node
            for node in split_einsum(einsum_node, input_node_subset).inputs
            if isinstance(node, ad.EinsumNode)
        ]

        opt_contract_tree = get_common_ancestor(
            generate_optimal_tree(splitted_einsum), splitted_einsum.inputs,
            first_contract_node)
        opt_contract_tree_leaves = get_all_inputs(opt_contract_tree)

        # Get part of the inputs of splitted_einsum,
        # whose common_ancestor is opt_contract_tree (along the optimal contraction path)
        first_contract_inputs = [
            node for node in splitted_einsum.inputs
            if set(get_all_inputs(node)).issubset(opt_contract_tree_leaves)
        ]

        split_out_inputs = [
            node for node in einsum_node.inputs
            if node not in first_contract_inputs
        ]
        second_einsum = split_einsum(einsum_node, split_out_inputs)
        second_einsums.append(second_einsum)

    # Note that second_einsums[-1] will be directly returned and not be resued.
    # Its inputs will contain one splited_einsum, and other variable nodes.
    return dimension_tree(second_einsums[:-1], input_nodes[:-1],
                          input_nodes[-1]) + [second_einsums[-1]]
