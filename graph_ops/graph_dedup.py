import autodiff as ad
from utils import find_topo_sort, OutputInjectedMode, replace_node


def dedup(node):
    """Remove the duplicate nodes with same name.
    """

    topo_order = find_topo_sort([node])
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
