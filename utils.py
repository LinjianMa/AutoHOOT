from functools import reduce
import re
import autodiff as ad
import backend as T
import scipy.linalg as sla
import time
import numpy as np
import numpy.linalg as la


def jit_decorator(forward):
    from jax import jit

    def wrapper_jit(*args, **kwargs):
        jit_forward = jit(forward)
        jax_np_array = jit_forward(*args, **kwargs)
        return [np.asarray(element) for element in jax_np_array]

    return wrapper_jit


def indices_to_subscripts(in_indices, out_index, dim_size):
    """ Produce an einsum subscript based on the input indices notations.
        Args:
            in_indices: an array with each element being the index array of
                that input tensor.
            out_index: The index array of the output tensor.
            dim_size: number of dimensions involved in the einsum.
        Returns:
            new_subscripts: the einsum subscripts corresponding to the index notation.
        Examples:
            For the einsum E[0,1,2,3,4,5,6,7] = A[0,4]*B[1,5]*C[2,6]*D[3,7]
            where E is a 8-d tensor, and A B C D are 2-d tensors, each dimension of
            E is corresponding to one dimension of the inputs with the same number.
            The string based einsum will be 'ae,bf,cg,dh->abcdefgh'.
    """
    cg = CharacterGetter()
    chars = {}
    for i in range(dim_size):
        chars[i] = cg.getchar()
    # Assign literals
    input_subs = []
    for in_index in in_indices:
        input_subs.append("".join([chars[i] for i in in_index]))
    output_sub = "".join([chars[i] for i in out_index])
    new_subscripts = ",".join(input_subs) + "->" + output_sub
    return new_subscripts


##############################
####### Helper Methods #######
##############################
class CharacterGetter():
    """ Return a character and increment"""
    def __init__(self):
        self.char = 'a'

    def getchar(self):
        """
            Returns a single character. Increment after return.
        """
        previous_char = self.char
        if self.char == 'z':
            logging.info('Run out of characters.')
            raise NotImplementedError
        self.char = chr(ord(self.char) + 1)
        return previous_char


class IntGetter():
    """ Return a int and increment"""
    def __init__(self):
        self.val = 0

    def getint(self):
        """
            Returns an stringlized int. Increment after return.
        """
        previous_val = self.val
        self.val = self.val + 1
        return str(previous_val)


class OutputInjectedMode:
    def __init__(self, nodes):
        self.nodes = nodes

    def __enter__(self):
        for n in self.nodes:
            n.outputs = []
        for n in self.nodes:
            for n_i in n.inputs:
                n_i.outputs.append(n)

    def __exit__(self, type, value, traceback):
        for n in self.nodes:
            n.outputs = []


def get_root(nodes):
    """
        Returns the root of a set of nodes. Nodes is connected and a tree.
        Args:
            Nodes is a list of graph nodes.
        Returns:
            one single node that is the root.
        Idea: The root must not any nodes input.
        Complexity: O(N^2)
    """
    all_inputs = set()
    for n in nodes:
        for n_i in n.inputs:
            all_inputs.add(n_i)
    for n in nodes:
        if n not in all_inputs:
            return n


def get_leaves(nodes):
    """
        Returns all the inputs of the nodes formed as a tree. The returned node
        must not be in the tree.
        Args:
            Nodes is a list of graph nodes.
        Returns:
            a set of nodes.
        
        For further illustration:
            A
           / \
          B  I3
         / \
        I1  I2
        
        I1, I2, I3 are the returned nodes. inputs is {A,B} 
    """
    all_inputs = set()
    for n in nodes:
        for n_i in n.inputs:
            if n_i not in nodes:
                all_inputs.add(n_i)
    return all_inputs


def replace_node(prev, new):
    """Replaces the previous node with the new node.

    Need OutputInjectedMode to be enabled.
    It will replace all the inputs reference to prev node to new node.
    Note that this is a mutation operation on the graph which is irreversible.

    Args:
        prev: A node in the graph.
        new: Another node in the graph.
    Returns:
        None
    """
    assert prev.outputs != None
    assert new.outputs != None
    for n_o in prev.outputs:
        n_o.set_inputs(
            [tmp if tmp.name != prev.name else new for tmp in n_o.inputs])


def find_topo_sort(node_list, input_node_list=[]):
    """Given a list of nodes, return a topological sort list of nodes ending in them.
    The input_node_list are used to stop. If ever met a input node, stop probing the graph.
    A simple algorithm is to do a post-order DFS traversal on the given nodes,
    going backwards based on input edges. Since a node is added to the ordering
    after all its predecessors are traversed due to post-order DFS, we get a topological
    sort.
    """
    visited = set()
    topo_order = []
    for node in node_list:
        topo_sort_dfs(node, visited, topo_order, input_node_list)
    return topo_order


def topo_sort_dfs(node, visited, topo_order, input_node_list):
    """Post-order DFS"""
    if node in visited:
        return
    visited.add(node)
    if node not in input_node_list:
        for n in node.inputs:
            topo_sort_dfs(n, visited, topo_order, input_node_list)
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


def group_vecnorm(xs):
    s = sum([T.sum(x * x) for x in xs])
    return s**0.5


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


# TODO: the wrapper needs to be refactored
def solve_tri(A, B, lower=True, from_left=True, trans_left=False):
    if not from_left:
        return sla.solve_triangular(A.T,
                                    B.T,
                                    trans=trans_left,
                                    lower=not lower).T
    else:
        return sla.solve_triangular(A, B, trans=trans_left, lower=lower)


def fast_block_diag_precondition(X, P):
    ret = []
    for i in range(len(X)):
        Y = solve_tri(P[i], X[i], lower=True, from_left=False, trans_left=True)
        Y = solve_tri(P[i], Y, lower=True, from_left=False, trans_left=False)
        ret.append(Y)
    return ret


# TODO: this class now only supports numpy.
class cp_nls_optimizer():
    def __init__(self, input_tensor, A, cg_tol=1e-3):
        self.input_tensor = input_tensor
        self.A = A
        self.maxiter = len(A) * (A[0].shape[0]) * (A[0].shape[1])
        self.cg_tol = cg_tol
        self.atol = 0
        self.total_iters = 0
        self.total_cg_time = 0.
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

        self.atol = self.num * group_vecnorm(delta)
        print(f"cg iterations: {counter}")
        print(f"total cg iterations: {self.total_iters}")
        print(f"total cg time: {self.total_cg_time}")

        self.A[0] += delta[0]
        self.A[1] += delta[1]
        self.A[2] += delta[2]

        return self.A, self.total_cg_time

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
        tol = np.max([self.atol, self.cg_tol * group_vecnorm(grads)])
        hvps = hess_fn(x)
        r = group_minus(
            group_negative(self.fast_hessian_contract(x, regularization,
                                                      hvps)), grads)
        if group_vecnorm(r) < tol:
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

            if group_vecnorm(r_new) < tol:
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
        print(f"cg took: {end - start}")
        self.total_cg_time += end - start
        return x, counter
