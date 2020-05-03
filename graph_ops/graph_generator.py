from utils import get_all_einsum_descendants, get_all_inputs, find_topo_sort, get_all_nodes

import autodiff as ad
import copy
import numpy as np


def generate_optimal_tree(node, path=None):
    """Generates the descendants of the optimal path.
    
    Args:
        node: The einsum node we are interested about.
        path: If specified, will be used to generate tree.
    Returns:
        final_node: The newly generated node.
    """
    assert isinstance(node, ad.EinsumNode)
    leaves = get_all_inputs(node)
    for leaf in leaves:
        assert (not isinstance(leaf, ad.EinsumNode))

    # Need to call with numpy array with same shape.
    # This is a numpy specific tweak because this function relies on numpy
    # implementation of parse_einsum_input.
    np_inputs = [np.zeros(i_node.shape) for i_node in node.inputs]

    if path is None:
        _, contract_list = np.einsum_path(node.einsum_subscripts,
                                          *np_inputs,
                                          einsum_call=True)
    else:
        assert len(path) > 0
        _, contract_list = np.einsum_path(node.einsum_subscripts,
                                          *np_inputs,
                                          optimize=path,
                                          einsum_call=True)

    original_inputs = [i for i in node.inputs]
    final_node = None
    for contract in contract_list:
        indices, _, subscript, _, _ = contract
        input_nodes = [original_inputs[i] for i in indices]
        new_node = ad.einsum(subscript, *input_nodes)
        original_inputs.append(new_node)
        for i_node in input_nodes:
            original_inputs.remove(i_node)
        final_node = new_node
    return final_node


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
    indices = [
        i for (i, node) in enumerate(einsum_node.inputs)
        if node in (set(einsum_node.inputs) - set(split_input_nodes))
    ]

    merge = tuple(range(len(einsum_node.inputs) - len(indices) + 1))
    return generate_optimal_tree(einsum_node,
                                 path=['einsum_path', indices, merge])


def get_common_ancestor(root, leaves, in_node):
    """
    Get in_node's common ancestor of a tree(defined by root and leaves).
    Here our tree may let a leaf in_node has multiple parents.

    Parameters
    ----------
    root: Tree root.
    leaves: A list of leaf nodes define the inputs of the subtree.
    in_node: one of the node in leaves such that multiple intermediate nodes can have it as children.

    Returns
    ----------
    ancestor: A ancestor that covers all the in_node(s) in the tree.
    """

    assert in_node in leaves

    num_in_nodes = len(list(filter(lambda n: n is in_node, leaves)))
    topo_order_list = find_topo_sort([root], leaves)

    for node in topo_order_list:
        # We want to get the smallest subtree whose inputs contain all the in_node(s).
        if isinstance(node, ad.EinsumNode):
            subtree_leaves = [
                n for n in get_all_nodes([node], leaves) if n in leaves
            ]
            num_in_nodes_subtree = len(
                list(filter(lambda n: n is in_node, subtree_leaves)))
            if num_in_nodes == num_in_nodes_subtree:
                return node


def generate_optimal_tree_w_constraint(einsum_node, contract_order):
    """Generates the optimal path with constraint.

    Args:
        einsum_node: The einsum node we are generating the path wrt.
        contract_order: A list containing the contraction order constraint.
                        Nodes in contract_order will be contracted based on the order
                        from list start to the end.
    Returns:
        out_node: The newly generated node.
    """
    assert set(contract_order).issubset(set(einsum_node.inputs))
    # TODO: currently doesn't support the case where every input is in the contract_order
    assert len(set(einsum_node.inputs)) - len(set(contract_order)) > 0

    out_node = einsum_node

    for i in range(len(contract_order)):

        uncontracted_nodes = contract_order[i + 1:]

        if uncontracted_nodes == []:
            splitted_einsum = out_node
        else:
            splitted_einsum, = [
                node
                for node in split_einsum(out_node, uncontracted_nodes).inputs
                if isinstance(node, ad.EinsumNode)
            ]

        splitted_einsum_opt_path = generate_optimal_tree(splitted_einsum)
        opt_contract_tree = get_common_ancestor(splitted_einsum_opt_path,
                                                splitted_einsum.inputs,
                                                contract_order[i])

        first_contract_inputs = [
            n
            for n in get_all_nodes([opt_contract_tree], splitted_einsum.inputs)
            if n in splitted_einsum.inputs
        ]

        split_out_inputs = [
            node for node in out_node.inputs
            if node not in first_contract_inputs
        ]
        out_node = split_einsum(out_node, split_out_inputs)

    return out_node
