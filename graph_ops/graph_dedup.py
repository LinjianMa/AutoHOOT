import autodiff as ad
from utils import find_topo_sort, OutputInjectedMode, replace_node
from numpy.core.einsumfunc import _parse_einsum_input
from collections import defaultdict


def dedup(*nodes):
    """Remove the duplicate nodes with same name.
    Args:
        nodes: One or many nodes.
    """
    assert len(nodes) > 0

    topo_order = find_topo_sort(nodes)
    with OutputInjectedMode(topo_order):
        unique_nodes_map = {}
        unique_nodes = set()
        # Use the last occurrence.
        for tmp in topo_order:
            unique_nodes_map[tmp.name] = tmp
        unique_nodes = set(unique_nodes_map.values())

        for tmp in topo_order:
            if tmp not in unique_nodes:
                unique_copy = unique_nodes_map[tmp.name]
                replace_node(tmp, unique_copy)


def declone(o_node):
    """
    Args:
        o_node: An output node.
    Returns:
        o_node: A new node with new name.
    """

    if isinstance(o_node, ad.VariableNode):
        return o_node
    if isinstance(o_node, ad.CloneNode):
        assert len(o_node.inputs) == 1
        return declone(o_node.inputs[0])

    new_inputs = []
    for i_node in o_node.inputs:
        i_node = declone(i_node)
        new_inputs.append(i_node)
    o_node.set_inputs(new_inputs)
    return o_node


def transpose_equivalent(A, B):
    """
    Two nodes are transpose equivalent if following meets:
    1. The contract dims are referring to the same dim.
    2. others inputs dim are permutation of each other.
    
    Args:
        A, B: einsum nodes.
    Returns:
        True/False

    Assume already through rewrite_einsum, a.k.a inputs are sorted.
    """
    if A.inputs != B.inputs:
        return False

    def get_contract_node_dims(node):
        # Returns a nested map.
        # Contract_char -> node -> index
        # e.g {'c': {A: 0, B:1}}
        ret = defaultdict(dict)
        in_subs, out_subs, _ = _parse_einsum_input(
            (node.einsum_subscripts, *node.inputs))
        in_subs_list = in_subs.split(',')
        contract_dims = set("".join(in_subs_list)) - set(out_subs)
        for contract_dim in contract_dims:
            for inode, in_sub in zip(node.inputs, in_subs_list):
                innode_index = in_sub.index(contract_dim)
                ret[contract_dim][inode] = innode_index
        return ret

    def get_node_chars(node):
        # Returns a map:
        # node -> indices
        # e.g {A: set('abc'), B: set('cd')}
        in_subs, out_subs, _ = _parse_einsum_input(
            (node.einsum_subscripts, *node.inputs))
        in_subs_list = in_subs.split(',')
        in_subs_list = [set(isl) for isl in in_subs_list]
        return dict(zip(node.inputs, (in_subs_list)))

    # Check if all the contracted char refers to the same dim.
    if (get_contract_node_dims(A)) != get_contract_node_dims(B):
        return False

    # Check if nodes contain the same chars.
    if get_node_chars(A) != get_node_chars(B):
        return False

    return True


def detranspose(o_node):
    """
    Identifies the equivalent transpose nodes and transform one to the other by propogating changes to the output nodes.
    """
    pass
    pass
