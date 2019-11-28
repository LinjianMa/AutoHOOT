import autodiff as ad

from utils import get_all_einsum_descdants, get_leaves


def generate_optimal_tree(node):
    """Generates the descendants of the optimal path.
    
    Update in-place.
    Args:
        node: The einsum node we are interested about.
    Returns:
        None.
    """
    assert isinstance(node, ad.EinsumNode)
    leaves = get_leaves(get_all_einsum_descdants(node))
    for leaf in leaves:
        assert isinstance(leaf, ad.EinsumNode)
