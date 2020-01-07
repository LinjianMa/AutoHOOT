"""
    This file will contains the graph transformations and optimizations for 
    alternating least squares.
"""
import logging
import autodiff as ad

from graph_ops.graph_transformer import rewrite_einsum_expr
from graph_ops.graph_optimizer import fuse_einsums
from graph_ops.utils import einsum_equal
from graph_ops.graph_dedup import dedup

from numpy.core.einsumfunc import _parse_einsum_input

FORMAT = '[%(asctime)-15s %(filename)s:%(lineno)s] %(message)s'

logger = logging.getLogger('optimizer')
logging.basicConfig(format=FORMAT)
logger.setLevel(logging.DEBUG)


def find_contraction(contract_nodes, input_nodes, output_node):
    """
    Finds the contraction output string for a given input array and output node.
    Parameters
    ----------
    positions : list
        two nodes used in the contraction.
    input_nodes : list
        List of nodes that represent the left hand side of the einsum subscript
    output_node : ad.EinsumNode
        Output node
    Returns
    -------
    output_string : string
        The output string of the einsum of the contract_nodes.
    Note
    -----
    The substring of all the nodes need to be set in advance.
    Examples
    --------
    # A simple dot product test case
    >>> contract_nodes = [A, B]
    >>> inodes = [A, B, C, D]
    >>> onode = ad.einsun("ab,bc,cd,de->ae", A,B,C,D)
    >>> find_contraction(contract_nodes, inodes, onode)
    "ac"
    # A more complex case with additional terms in the contraction
    >>> contract_nodes = [A,C]
    >>> inodes = [A, B, C]
    >>> onode = ad.einsum("abd,ac,bdc->ac", A, B, C)
    >>> find_contraction(contract_nodes, inodes, onode)
    "ac"
    """
    if len(input_nodes) == 2:
        return output_node.substring
    idx_contract = set()
    idx_remain = set(output_node.substring).copy()
    for in_node in input_nodes:
        subs_set = set(in_node.substring)
        if in_node in contract_nodes:
            idx_contract |= subs_set
        else:
            idx_remain |= subs_set

    output_set = idx_remain & idx_contract
    # the char order in the output string is based on the name of contract nodes
    contract_nodes = sorted(contract_nodes, key=lambda node: node.name)
    output_list = []
    for contract_node in contract_nodes:
        for char in contract_node.substring:
            if char in output_set:
                output_list.append(char)
                output_set.remove(char)
    return "".join(output_list)


def einsum_partial_contract(contract_nodes, input_nodes, einsum_node):
    """
    Partially contract the input nodes of the einsum node.
    Parameters
    ----------
    contract_nodes : list
        The nodes to be contracted
    input_nodes : list
        The input nodes of the einsum node
    einsum_node : ad.EinsumNode
        Input einsum node
    Returns
    -------
    output_node_fuse : ad.EinsumNode
        Einsum of the contract nodes
    Examples
    --------
    >>> einsum_node = ad.einsum("ab,bc,cd,de->ae", A,B,C,D)
    >>> contract_nodes = [C, D]
    >>> einsum_partial_contract(contract_nodes, [A, B, C, D], einsum_node)
    ad.einsum("cd,de->ce", C,D)
    """
    contract_nodes = sorted(contract_nodes, key=lambda node: node.name)
    output_node = contract_nodes[0]
    for node in contract_nodes[1:]:
        # build the einsum string
        output_str = find_contraction([output_node, node], input_nodes,
                                      einsum_node)
        intermediate = ad.einsum(
            f"{output_node.substring},{node.substring}->{output_str}",
            output_node, node)
        intermediate.substring = output_str
        input_nodes = list(
            set(input_nodes) - {output_node, node} | {intermediate})
        output_node = intermediate
    output_node_fuse = fuse_einsums(output_node, contract_nodes)
    # TODO: this is ugly: fuse_einsums function changes the substring in the
    # einsum node and input nodes because new chars are assigned. However, we
    # need the substring of the output_node_fuse to be the old char so that
    # further partial contracts can be correct.
    output_node_fuse.substring = output_str

    return output_node_fuse


def split_einsum(einsum_node, split_input_nodes):
    """
    Split the einsum node into two einsum nodes.
    Parameters
    ----------
    einsum_node : ad.EinsumNode
        Input einsum node
    split_input_nodes : list
        List of input nodes that are split out from the first einsum contraction
    Returns
    -------
    second_einsum : ad.EinsumNode
        A newly written einsum composed of an intermediate node composed of
        input nodes except split_input_nodes.
    Examples
    --------
    >>> einsum_node = ad.einsum("ab,bc,cd,de->ae", A,B,C,D)
    >>> split_input_nodes = [A, B]
    >>> split_einsum(einsum_node, split_input_nodes)
    ad.einsum("ab,bc,ce->ae", A,B,ad.einsum("cd,de->ce",C,D))
    """

    # set the substring
    in_subs, out_subs, _ = _parse_einsum_input(
        (einsum_node.einsum_subscripts, *einsum_node.inputs))
    in_subs_list = in_subs.split(',')
    for i, input_node in enumerate(einsum_node.inputs):
        input_node.substring = in_subs_list[i]
    einsum_node.substring = out_subs

    assert (isinstance(einsum_node, ad.EinsumNode))
    input_nodes = einsum_node.inputs
    assert set(split_input_nodes).issubset(set(input_nodes))
    first_contract_nodes = list(set(input_nodes) - set(split_input_nodes))

    if first_contract_nodes == []:
        return None, einsum_node

    first_einsum = einsum_partial_contract(first_contract_nodes, input_nodes,
                                           einsum_node)
    second_einsum = einsum_partial_contract(split_input_nodes + [first_einsum],
                                            split_input_nodes + [first_einsum],
                                            einsum_node)
    return second_einsum


def generate_sequential_optiaml_tree(einsum_node_map={}):
    """Generates a list of nodes in-order. 
    Args: 
        einsum_node_map: a dict that maps from an output node to a input node.
     
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
                        list(einsum_node_map.values()))
    dedup(*dt)
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

    second_einsums = []
    for einsum_node in einsum_nodes:
        input_node_subset = list(set(input_nodes) & set(einsum_node.inputs))
        input_node_subset = sorted(input_node_subset,
                                   key=lambda node: node.name)
        second_einsum = split_einsum(einsum_node, input_node_subset)
        second_einsums.append(second_einsum)

    return dimension_tree(second_einsums[:-1],
                          input_nodes[:-1]) + [second_einsums[-1]]
