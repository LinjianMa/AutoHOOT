from utils import get_all_einsum_descendants, get_leaves

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
    leaves = get_leaves(get_all_einsum_descendants(node))
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
