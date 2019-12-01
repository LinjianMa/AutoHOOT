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
