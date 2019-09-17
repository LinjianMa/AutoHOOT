from functools import reduce
import re
import autodiff as ad
import backend as T
import scipy.linalg as sla
import time
import numpy as np
import numpy.linalg as la

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
    inner_product_nodes = [
        ad.sum(v * g) for v, g in zip(vector_list, gradient_list)
    ]
    sum_node = sum_node_list(inner_product_nodes)
    return sum_node


#####################################################
####### Helper Methods for Conjugate gradient #######
#####################################################


def group_minus(xs, ys):
    assert len(xs) == len(ys)
    return [x - y for (x, y) in zip(xs, ys)]


def inplace_group_add(xs, ys):
    assert len(xs) == len(ys)
    for x, y in zip(xs, ys):
        x += y


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
        inplace_group_add(x0, group_product(alpha, p))
        inplace_group_add(r, group_product(alpha, Ap))
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


def list_vecnorm(list_A):
    l = [i**2 for i in list_A]
    s = 0
    for i in range(len(l)):
        s += np.sum(l[i])

    return s**0.5


def solve_tri(A, B, lower=True, from_left=True, transp_L=False):
    if not from_left:
        B = B.T
        A = A.T
        llower = not lower
        X = sla.solve_triangular(A, B, trans=transp_L, lower=llower)
        return X.T
    else:
        return sla.solve_triangular(A, B, trans=transp_L, lower=lower)


def fast_block_diag_precondition(X, P):
    N = len(X)
    ret = []
    for i in range(N):
        Y = solve_tri(P[i], X[i], True, False, True)
        Y = solve_tri(P[i], Y, True, False, False)
        ret.append(Y)
    return ret


class cp_nls_optimizer():
    def __init__(self, input_tensor, A, cg_tol=1e-3):
        self.input_tensor = input_tensor
        self.A = A
        self.maxiter = len(A) * (A[0].shape[0]) * (A[0].shape[1])
        self.cg_tol = cg_tol
        self.atol = 0
        self.total_iters = 0
        self.num = 0

    def step(self, hess_fn, grads, regularization):
        A = self.A
        self.gamma = []
        self.gamma.append(
            (T.transpose(A[1]) @ A[1]) * (T.transpose(A[2]) @ A[2]))
        self.gamma.append(
            (T.transpose(A[0]) @ A[0]) * (T.transpose(A[2]) @ A[2]))
        self.gamma.append(
            (T.transpose(A[0]) @ A[0]) * (T.transpose(A[1]) @ A[1]))

        P = self.compute_block_diag_preconditioner(regularization)
        delta, counter = self.fast_precond_conjugate_gradient(
            hess_fn, grads, P, regularization)

        self.total_iters += counter

        self.atol = self.num * list_vecnorm(delta)
        print("cg iterations: ", counter)
        print("total cg iterations: ", self.total_iters)

        self.A[0] += delta[0]
        self.A[1] += delta[1]
        self.A[2] += delta[2]

        return self.A

    def compute_block_diag_preconditioner(self, regularization):
        P = []
        for i in range(len(self.A)):
            n = self.A[i].shape[1]
            P.append(
                la.cholesky(self.gamma[i]) +
                regularization * np.diag(self.gamma[i].diagonal()))
        return P

    def fast_hessian_contract(self, delta, regularization, hvps):
        N = len(self.A)
        ret = []
        for n in range(N):
            ret.append(T.zeros(self.A[n].shape))
            ret[n] = ret[n] + regularization * T.einsum(
                'jj,ij->ij', self.gamma[n], delta[n]) + hvps[n]
        return ret

    def fast_precond_conjugate_gradient(self, hess_fn, grads, P,
                                        regularization):
        start = time.time()

        x = [T.zeros(A.shape) for A in grads]
        tol = np.max([self.atol, self.cg_tol * list_vecnorm(grads)])
        hvps = hess_fn(x)
        r = group_minus(
            group_negative(self.fast_hessian_contract(x, regularization,
                                                      hvps)), grads)
        if list_vecnorm(r) < tol:
            return x, 0
        z = fast_block_diag_precondition(r, P)
        p = z
        counter = 0
        while True:
            hvps = hess_fn(p)
            mv = self.fast_hessian_contract(p, regularization, hvps)
            mul = group_dot(r, z)
            alpha = mul / group_dot(p, mv)
            inplace_group_add(x, group_product(alpha, p))
            r_new = group_minus(r, group_product(alpha, mv))

            if list_vecnorm(r_new) < tol:
                counter += 1
                end = time.time()
                break

            z_new = fast_block_diag_precondition(r_new, P)
            beta = group_dot(r_new, z_new) / mul
            p = group_minus(z_new, group_negative(group_product(beta, p)))
            r = r_new
            z = z_new
            counter += 1

        end = time.time()
        print("cg took: ", end - start)
        return x, counter
