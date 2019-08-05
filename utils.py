from functools import reduce
import re
import autodiff as ad

##############################
####### Helper Methods #######
##############################


def einsum_grad_subscripts(subscripts, left=True):
    match = re.search(r'^(.*),(.*)->(.*)', subscripts)
    if left:
        return match.group(3) + ',' + match.group(2) + '->' + match.group(1)
    else:
        return match.group(1) + ',' + match.group(3) + '->' + match.group(2)


def find_topo_sort(node_list):
    """Given a list of nodes, return a topological sort list of nodes ending in them.

    A simple algorithm is to do a post-order DFS traversal on the given nodes,
    going backwards based on input edges. Since a node is added to the ordering
    after all its predecessors are traversed due to post-order DFS, we get a topological
    sort.

    """
    visited = set()
    topo_order = []
    for node in node_list:
        topo_sort_dfs(node, visited, topo_order)
    return topo_order


def topo_sort_dfs(node, visited, topo_order):
    """Post-order DFS"""
    if node in visited:
        return
    visited.add(node)
    for n in node.inputs:
        topo_sort_dfs(n, visited, topo_order)
    topo_order.append(node)


def sum_node_list(node_list):
    """Custom sum function in order to avoid create redundant nodes in Python sum implementation."""
    from operator import add
    from functools import reduce
    return reduce(add, node_list)


def inner_product(vector_list, gradient_list):
    assert len(vector_list) == len(gradient_list)
    assert len(vector_list) >= 1
    inner_product_node = ad.sum(vector_list[0] * gradient_list[0])
    for i in range(1, len(vector_list)):
        inner_product_node = inner_product_node + sum(
            vector_list[i] * gradient_list[i])
    return inner_product_node