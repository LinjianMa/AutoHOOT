from functools import reduce
import re
import autodiff as ad
import backend as T

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
        inner_product_node = inner_product_node + ad.sum(
            vector_list[i] * gradient_list[i])
    return inner_product_node


#####################################################
####### Helper Methods for Conjugate gradient #######
#####################################################


def group_minus(xs, ys):
    assert len(xs) == len(ys)
    return [x - y for (x, y) in zip(xs, ys)]


def group_add(xs, ys):
    assert len(xs) == len(ys)
    return [x + y for (x, y) in zip(xs, ys)]


def group_negative(xs):
    return [-x for x in xs]


def group_dot(xs, ys):
    assert len(xs) == len(ys)
    return sum([T.sum(x * y) for (x, y) in zip(xs, ys)])


def group_product(alpha, xs):
    return [alpha * x for x in xs]


def conjugate_gradient(hess_fn, grads, error_tol, max_iters=250, x0=None):
    '''
        This solves the following problem:
        hess_fn(x) = grad
        return: (x)
    '''
    if not x0:
        x0 = [T.ones(grad.shape) for grad in grads]
    hvps = hess_fn(x0)
    r = group_minus(hvps, grads)
    p = group_negative(r)
    r_k_norm = group_dot(r, r)

    i = 0
    while True:
        Ap = hess_fn(p)
        alpha = r_k_norm / group_dot(p, Ap)
        x0 = group_add(x0, group_product(alpha, p))
        r = group_add(r, group_product(alpha, Ap))
        r_kplus1_norm = group_dot(r, r)
        beta = r_kplus1_norm / r_k_norm
        r_k_norm = r_kplus1_norm
        if float(r_kplus1_norm) < error_tol:
            break
        p = group_minus(group_product(beta, p), r)
        if i > max_iters:
            print(f'CG max iter reached.')
            break
        i += 1
    return x0
