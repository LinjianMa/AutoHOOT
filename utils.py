from functools import reduce

import attr
import autodiff as ad
import backend as T
import logging
import numpy as np
import time
from sympy import symbols, simplify
from name_parser import AutodiffParser

FORMAT = '[%(asctime)-15s %(filename)s:%(lineno)s] %(message)s'

logger = logging.getLogger('optimizer')
logging.basicConfig(format=FORMAT)
logger.setLevel(logging.DEBUG)


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


def get_all_inputs(out):
    all_inputs = []
    if len(out.inputs) == 0:
        all_inputs.append(out)
    else:
        for i in out.inputs:
            all_inputs += get_all_inputs(i)
    return all_inputs


def get_tree(sinks, sources=[]):
    """
    Get all the nodes in the tree defined by root node.
    Args:
        sinks: A list of nodes defines the sink.
        sources: Stop expanding at provided sources.
    """
    return [
        pnode.node
        for pnode in find_topo_sort_p([PseudoNode(sink)
                                       for sink in sinks], sources)
    ]


def sympy_simplify(out, inputs):
    """
    Parameters
    ------------------
    out: The to-be simplified node.
    inputs: The nodes that are symbols.
    ------------------
    Returns 
    ------------------
    out: The simplified output.
    ------------------
    """
    # Retrieve all the leaf nodes to establish parser table.
    parser_input = get_all_inputs(out)
    cg = CharacterGetter()
    input_to_chars = {i: '__' + cg.getchar() for i in inputs}
    chars_to_input = dict(zip(input_to_chars.values(), input_to_chars.keys()))
    ss_name = f', '.join([input_to_chars[i] for i in inputs])
    ss = symbols(ss_name)
    formula = out.name
    # Visit the inputs in reverse topological order.
    all_nodes = reversed(find_topo_sort([out]))
    for i in all_nodes:
        if i in inputs:
            formula = formula.replace(i.name, input_to_chars[i])

    ret = str(simplify(formula))

    all_nodes = reversed(find_topo_sort([out]))
    for i in all_nodes:
        if i in inputs:
            ret = ret.replace(input_to_chars[i], i.name)
    return AutodiffParser.parse(ret, parser_input)


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
            self.char = 'A'
        elif self.char == 'Z':
            logging.info('Run out of characters.')
            raise NotImplementedError
        else:
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


class OutputInjectedModeP:
    def __init__(self, pnodes):
        self.pnodes = pnodes

    def __enter__(self):
        for n in self.pnodes:
            n.node.outputs = []
        for n in self.pnodes:
            for n_i in n.node.inputs:
                n_i.outputs.append(n.node)

    def __exit__(self, type, value, traceback):
        for n in self.pnodes:
            n.node.outputs = []


class StandardEinsumExprMode:
    """
    Change the einsum node to its stardard format.

    Within the mode, we provide subscripts and generated
    union-find data structure for optimization purposes.
    Input and einsum subscripts will not change after exiting the mode.
    """
    def __init__(self, node):
        self.node = node

    def __enter__(self):
        from graph_ops.graph_transformer import generate_einsum_info
        uf, p_outnode, p_innodes = generate_einsum_info(self.node)
        self.node.uf = uf
        self.p_outnode = p_outnode
        self.p_innodes = p_innodes
        return self

    def __exit__(self, type, value, traceback):
        self.node.subscripts = None
        for n in self.node.inputs:
            n.subscripts = None
        self.node.uf = None
        self.p_outnode = None
        self.p_innodes = None


@attr.s(eq=False)
class PseudoNode(object):
    node = attr.ib()
    literals = attr.ib(default=[])
    subscript = attr.ib(default='')

    def generate_subscript(self, uf):
        self.subscript = "".join(
            [uf.rootval(literal_name) for literal_name in self.literals])


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


def get_all_einsum_descendants(node):
    """Returns all the einsum descendants including himself.
    Args:
        A node in the graph.
    Returns:
        A list of all connected einsum nodes in the graph.
    """
    assert isinstance(node, ad.EinsumNode)
    tree_nodes = [node]
    for i_node in node.inputs:
        if isinstance(i_node, ad.EinsumNode):
            nodes = get_all_einsum_descendants(i_node)
            tree_nodes += nodes
    return tree_nodes


def replace_node(prev, new):
    """Replaces the previous node with the new node.

    Need OutputInjectedMode to be enabled.
    It will replace all the inputs reference to prev node to new node.
    Note that this is a mutation operation on the graph which is irreversible.

    Args:
        prev: A Pesudo node in the graph.
        new: Another node in the graph.
    Returns:
        None
    Note: A pesudo node represents a node pointer so that the value can be
    updated without discard the reference.
    """
    assert prev.node.outputs != None
    assert new.outputs != None
    if len(prev.node.outputs) == 0:
        prev.node = new
        return
    prev_node = prev.node
    for n_o in prev_node.outputs:
        n_o.set_inputs(
            [tmp if tmp.name != prev_node.name else new for tmp in n_o.inputs])


def update_variables(out_nodes, variables):
    """
    Inplace update the variable nodes for out_nodes.
    Args:
        out_nodes: A list of nodes whose input variable nodes will be updated.
        variables: A list of new variables
    """
    variable_dict = {node.name: node for node in variables}
    pnodes = [PseudoNode(node) for node in out_nodes]
    all_pnodes = find_topo_sort_p(pnodes)

    with OutputInjectedModeP(all_pnodes):
        for pnode in all_pnodes:
            node = pnode.node
            if node.inputs != []:
                node.set_inputs(node.inputs)
            if isinstance(node,
                          ad.VariableNode) and node.name in variable_dict:
                new_node = variable_dict[node.name]
                replace_node(pnode, new_node)


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


def find_topo_sort_p(pnode_list, input_node_list=[]):
    """Given a list of nodes, return a topological sort list of nodes ending in them.
    The input_node_list are used to stop. If ever met a input node, stop probing the graph.
    A simple algorithm is to do a post-order DFS traversal on the given nodes,
    going backwards based on input edges. Since a node is added to the ordering
    after all its predecessors are traversed due to post-order DFS, we get a topological
    sort.
    """
    visited = set()
    topo_order = []
    for node in pnode_list:
        topo_sort_dfs_p(node, visited, topo_order, input_node_list)
    return topo_order


def topo_sort_dfs_p(pnode, visited, topo_order, input_node_list):
    """Post-order DFS"""
    if pnode in visited:
        return
    visited.add(pnode)
    node = pnode.node
    if node not in input_node_list:
        for n in node.inputs:
            topo_sort_dfs_p(PseudoNode(n), visited, topo_order,
                            input_node_list)
    topo_order.append(pnode)


def sum_node_list(node_list):
    """Custom sum function in order to avoid create redundant nodes in Python sum implementation."""
    from operator import add
    return reduce(add, node_list)


def inner_product(vector_list, gradient_list):
    assert len(vector_list) == len(gradient_list)
    assert len(vector_list) >= 1
    inner_product_nodes = [
        ad.tensordot(v, g,
                     [list(range(len(v.shape))),
                      list(range(len(v.shape)))])
        for v, g in zip(vector_list, gradient_list)
    ]
    sum_node = sum_node_list(inner_product_nodes)
    return sum_node


#####################################################
####### Helper Methods for Conjugate gradient #######
#####################################################


def group_minus(xs, ys):
    assert len(xs) == len(ys)
    return [x - y for (x, y) in zip(xs, ys)]


def group_add(xs, ys):
    assert len(xs) == len(ys)
    return [x + y for x, y in zip(xs, ys)]


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


def fast_block_diag_precondition(X, P):
    ret = []
    for i in range(len(X)):
        Y = T.solve_tri(P[i], X[i], lower=True, from_left=False, transp_L=True)
        Y = T.solve_tri(P[i], Y, lower=True, from_left=False, transp_L=False)
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
            self.A[i].shape[1]
            P.append(
                T.cholesky(self.gamma[i]) +
                regularization * T.diag(self.gamma[i].diagonal()))
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
            x = group_add(x, group_product(alpha, p))
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
