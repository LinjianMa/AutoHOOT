import math, copy, attr
import numpy as np
import autodiff as ad
import backend as T
import time
from numpy.core.einsumfunc import _parse_einsum_input
from tensors.quimb_tensors import gauge_transform_mps
from examples.mps import MpsGraph, MpoGraph, DmrgGraph
from graph_ops.graph_generator import split_einsum
import scipy.sparse.linalg as spla
from graph_ops.graph_optimizer import fuse_einsums
from graph_ops.graph_transformer import simplify
from utils import update_variables
from graph_ops.graph_als_optimizer import generate_sequential_optiaml_tree
from graph_ops.graph_generator import generate_optimal_tree_opt_einsum

BACKEND_TYPES = ['numpy']


@attr.s()
class DmrgGraph_implicit_shared_exec(object):

    mpo_inputs = attr.ib()
    mps_inputs = attr.ib()
    executor = attr.ib()
    executor_intermediates = attr.ib()
    intermediates = attr.ib(default=[])
    hes_vecs = attr.ib(default=[])
    v_nodes = attr.ib(default=[])

    def update_graph(self, num, mpo_ranks, mps_ranks, size):
        A_list, _ = MpsGraph.build_inputs_list(num, mps_ranks, size)
        H_list, _, _ = MpoGraph.build_inputs_list(num, mpo_ranks, size)
        self.mpo_inputs = H_list
        self.mps_inputs = A_list

        variable_dict = {node.name: node for node in A_list}
        variable_dict.update({node.name: node for node in H_list})

        update_variables(self.intermediates, variable_dict)

        self.v_nodes = [
            ad.Variable(name=f"v{i}", shape=node.shape)
            for (i, node) in enumerate(self.intermediates)
        ]
        variable_dict.update({node.name: node for node in self.v_nodes})

        update_variables(self.hes_vecs, variable_dict)

    @classmethod
    def create(cls, num, mpo_ranks, mps_ranks, size):
        mpo_graph = MpoGraph.create(num, mpo_ranks, size)
        mps_graph = MpsGraph.create(num, mps_ranks, size)

        intermediates, hvps, vnodes = [], [], []
        for i in range(num - 1):
            intermediate, hvp, vnode = cls.get_sub_hvp(i, mpo_graph, mps_graph)
            hvp = simplify(hvp)
            # TODO: this is a hacky part considering the symmetry
            hvp = hvp.inputs[0]

            intermediates.append(intermediate)
            hvps.append(hvp)
            vnodes.append(vnode)

        hvps = generate_sequential_optiaml_tree(hvps, mps_graph.inputs)
        # hacky part to get optimal tree
        hvps = [generate_optimal_tree_opt_einsum(hvp) for hvp in hvps]
        # from visualizer import print_computation_graph
        # print_computation_graph(hvps)
        executor = ad.Executor(hvps)
        executor_intermediates = ad.Executor(intermediates)

        return cls(mpo_graph.inputs, mps_graph.inputs, executor,
                   executor_intermediates, intermediates, hvps, vnodes)

    @classmethod
    def get_sub_hvp(cls, index, mpo_graph, mps_graph):

        # rebuild mps graph
        intermediate_set = {
            mps_graph.inputs[index], mps_graph.inputs[index + 1]
        }
        split_input_nodes = list(set(mps_graph.inputs) - intermediate_set)
        mps = split_einsum(mps_graph.output, split_input_nodes)

        # get the intermediate node
        intermediate, = [
            node for node in mps.inputs if isinstance(node, ad.EinsumNode)
        ]

        mps_outer_product = ad.tensordot(mps, mps, axes=[[], []])

        mpo_axes = list(range(len(mpo_graph.output.shape)))
        objective = ad.tensordot(mps_outer_product,
                                 mpo_graph.output,
                                 axes=[mpo_axes, mpo_axes])
        v_node = ad.Variable(name=f"v{index}", shape=intermediate.shape)

        hvp, = ad.hvp(objective, [intermediate], [v_node])

        return intermediate, hvp, v_node


class TNLinearOperator(spla.LinearOperator):
    def __init__(self, dmrg_graph, i, feed_dict):

        col_size = np.prod(dmrg_graph.hes_vecs[i].shape)

        super().__init__(dtype=np.float64, shape=(col_size, col_size))

        self.eigvec_shape = dmrg_graph.hes_vecs[i].shape
        self.dmrg_graph = dmrg_graph
        self.index = i
        self.feed_dict = feed_dict
        self.num_call_matvec = 0

    def _matvec(self, vec):

        in_data = T.reshape(vec, self.eigvec_shape)

        i = self.index
        dg = self.dmrg_graph
        feed_dict = self.feed_dict
        feed_dict.update({dg.v_nodes[i]: in_data})

        if self.num_call_matvec == 0:
            if i == 0:
                reset_graph = True
                evicted_inputs = []
            else:
                reset_graph = False
                evicted_inputs = [dg.mps_inputs[i - 1], dg.v_nodes[i]]
        else:
            reset_graph = False
            evicted_inputs = [dg.v_nodes[i]]

        out_data, = dg.executor.run(feed_dict=feed_dict,
                                    reset_graph=reset_graph,
                                    evicted_inputs=evicted_inputs,
                                    out_nodes=[dg.hes_vecs[i]])

        self.num_call_matvec += 1

        return out_data.ravel()


def dmrg_implicit_local_update(intermediate, eigvecs, max_mps_rank):
    """
    Perform local update for DMRG.

    Parameters
    ----------
    intermediate: the input einsum node. Its inputs are two mps sites
    hes_val: the hessian tensor to calculate eigenvalue / gigenvector on
    max_mps_rank: maximum mps tensor rank

    """
    # parse intermediate strings
    inputs = intermediate.inputs
    assert len(inputs) == 2

    # Here input names are formatted as A{i}.
    index_input_0 = int(inputs[0].name[1:])
    index_input_1 = int(inputs[1].name[1:])

    in_subs, out_subs, _ = _parse_einsum_input(
        (intermediate.einsum_subscripts, *intermediate.inputs))

    if index_input_0 > index_input_1:
        # right site appers first
        right_subs, left_subs = in_subs.split(',')
    else:
        left_subs, right_subs = in_subs.split(',')

    map_subs_indices = dict(zip(out_subs,
                                list(range(len(intermediate.shape)))))

    contract_char, = list(set(left_subs) - set(out_subs))

    left_uncontract_chars = list(set(left_subs) - set(contract_char))
    right_uncontract_chars = list(set(right_subs) - set(contract_char))

    left_indices = [map_subs_indices[char] for char in left_uncontract_chars]
    right_indices = [map_subs_indices[char] for char in right_uncontract_chars]

    left_uncontract_str = "".join(left_uncontract_chars)
    right_uncontract_str = "".join(right_uncontract_chars)

    #############################################################
    # perform updates
    eigvec_shape = intermediate.shape
    eigvecs = T.reshape(eigvecs, eigvec_shape)

    # svd decomposition to get updated sites
    eigvecs_mat = T.transpose(eigvecs, left_indices + right_indices)
    eigvecs_mat = T.reshape(eigvecs_mat,
                            (np.prod([eigvec_shape[i]
                                      for i in left_indices]), -1))

    U, s, VT = T.svd(eigvecs_mat)
    rank = min([max_mps_rank, eigvecs_mat.shape[0], eigvecs_mat.shape[1]])
    U, s, VT = U[:, :rank], s[:rank], VT[:rank, :]
    VT = T.diag(s) @ VT

    U = T.reshape(U, [eigvec_shape[i] for i in left_indices] + [rank])
    VT = T.reshape(VT, ([rank] + [eigvec_shape[i] for i in right_indices]))

    left = T.einsum(f"{left_uncontract_str}{contract_char}->{left_subs}", U)
    right = T.einsum(f"{contract_char}{right_uncontract_str}->{right_subs}",
                     VT)

    return left, right


def dmrg_shared_exec_sparse_solve(mpo_tensors,
                                  init_mps_tensors,
                                  max_mps_rank,
                                  num_iter=1,
                                  sequence='R'):
    """
    Perform DMRG iterations with shared execution.

    Parameters
    ----------
    mpo_tensors: an array of mpo tensor data
    init_mps_tensors: an array of mps tensor data
    max_mps_rank: maximum mps rank in the iterations
    num_iter: total number of iterations
    sequence: str, String made of 'L' and 'R' defining the sweep sequence, e.g 'RRL'.
        The sequence will be repeated until num_iter is reached.

    """
    if sequence != "R":
        raise NotImplementedError

    num = len(mpo_tensors)
    size = mpo_tensors[0].shape[1]
    mpo_ranks = [mpo_tensors[i].shape[0] for i in range(1, len(mpo_tensors))]

    mps_tensors = copy.deepcopy(init_mps_tensors)
    mps_ranks = [mps_tensors[i].shape[0] for i in range(1, len(mps_tensors))]

    dg = DmrgGraph_implicit_shared_exec.create(num, mpo_ranks, mps_ranks, size)

    dt, dt2 = 0, 0

    # sequence is R
    for iter in range(num_iter):

        mps_tensors = gauge_transform_mps(mps_tensors, right=True)
        mps_ranks = [
            mps_tensors[i].shape[0] for i in range(1, len(mps_tensors))
        ]

        for i in range(num - 1):

            dg.update_graph(num, mpo_ranks, mps_ranks, size)

            feed_dict = dict(zip(dg.mpo_inputs, mpo_tensors))
            feed_dict.update(dict(zip(dg.mps_inputs, mps_tensors)))

            intermediate = dg.executor_intermediates.run(
                feed_dict=feed_dict, out_nodes=[dg.intermediates[i]])

            t0 = time.time()
            # calculate the eigenvector using sparse solver
            hes_operator = TNLinearOperator(dg, i, feed_dict)
            eig_vals, eigvecs = spla.eigsh(hes_operator,
                                           k=1,
                                           ncv=4,
                                           tol=1e-3,
                                           which='SA',
                                           v0=intermediate)
            eig_val = eig_vals[0]
            eigvec = eigvecs[:, 0]

            dt += time.time() - t0
            print(f"i={i}, and dt for eigen solve ={dt/(iter+1)}")

            # Update the two sites of mps
            mps_tensors[i], mps_tensors[i + 1] = dmrg_implicit_local_update(
                dg.intermediates[i], eigvec, max_mps_rank)

            dt2 += time.time() - t0
            print(f"i={i}, and dt with tensor split ={dt2/(iter+1)}")

            # update the rank
            mps_ranks[i] = mps_tensors[i + 1].shape[0]
            print(f'The smallest eigenvalue is: {eig_val}')

        print(f'At iteration {iter} the smallest eigenvalue is: {eig_val}')

    return mps_tensors, eig_val


def dmrg_shared_exec_hvp(mpo_tensors,
                         init_mps_tensors,
                         max_mps_rank,
                         num_iter=1,
                         num_inner_iter=1,
                         sequence='R'):

    if sequence != "R":
        raise NotImplementedError

    num = len(mpo_tensors)
    size = mpo_tensors[0].shape[1]
    mpo_ranks = [mpo_tensors[i].shape[0] for i in range(1, len(mpo_tensors))]

    mps_tensors = copy.deepcopy(init_mps_tensors)
    mps_ranks = [mps_tensors[i].shape[0] for i in range(1, len(mps_tensors))]

    dg = DmrgGraph_implicit_shared_exec.create(num, mpo_ranks, mps_ranks, size)

    sweep_times = []
    # sequence is R
    for iter in range(num_iter):

        mps_tensors = gauge_transform_mps(mps_tensors, right=True)
        mps_ranks = [
            mps_tensors[i].shape[0] for i in range(1, len(mps_tensors))
        ]

        t0 = time.time()
        for i in range(num - 1):

            feed_dict = dict(zip(dg.mpo_inputs, mpo_tensors))
            feed_dict.update(dict(zip(dg.mps_inputs, mps_tensors)))

            intermediate, = dg.executor_intermediates.run(
                feed_dict=feed_dict, out_nodes=[dg.intermediates[i]])

            feed_dict.update({dg.v_nodes[i]: intermediate})

            for j in range(num_inner_iter):
                if num_inner_iter == 0:
                    if i == 0:
                        reset_graph = True
                        evicted_inputs = []
                    else:
                        reset_graph = False
                        evicted_inputs = [dg.mps_inputs[i - 1], dg.v_nodes[i]]
                else:
                    reset_graph = False
                    evicted_inputs = [dg.v_nodes[i]]

                dg.executor.run(feed_dict=feed_dict,
                                reset_graph=reset_graph,
                                evicted_inputs=evicted_inputs,
                                out_nodes=[dg.hes_vecs[i]])

        sweep_times.append((time.time() - t0) / num_inner_iter)

    return sweep_times


def dmrg_hvp_jax(mpo_tensors,
                 init_mps_tensors,
                 max_mps_rank,
                 num_iter=1,
                 sequence='R'):

    if sequence != "R":
        raise NotImplementedError

    num = len(mpo_tensors)
    size = mpo_tensors[0].shape[1]
    mpo_ranks = [mpo_tensors[i].shape[0] for i in range(1, len(mpo_tensors))]

    mps_tensors = copy.deepcopy(init_mps_tensors)
    mps_ranks = [mps_tensors[i].shape[0] for i in range(1, len(mps_tensors))]

    mpo_graph = MpoGraph.create(num, mpo_ranks, size)
    mps_graph = MpsGraph.create(num, mps_ranks, size)

    intermediates, hvps, vnodes = [], [], []
    for i in range(num - 1):
        intermediate, hvp, vnode = DmrgGraph_implicit_shared_exec.get_sub_hvp(
            i, mpo_graph, mps_graph)
        intermediates.append(intermediate)
        hvps.append(hvp)
        vnodes.append(vnode)

    from source import SourceToSource
    StS = SourceToSource()
    StS.forward(intermediates,
                file=open("examples/jax_dmrg_intermediates.py", "w"),
                function_name='dmrg_intermediates',
                backend='jax')
    StS.forward(hvps,
                file=open("examples/jax_dmrg_hvp.py", "w"),
                function_name='dmrg_hvp',
                backend='jax')

    from examples.jax_dmrg_hvp import dmrg_hvp
    from examples.jax_dmrg_intermediates import dmrg_intermediates

    inputs_intermediate_func = [
        init_mps_tensors[1], init_mps_tensors[0], init_mps_tensors[2],
        init_mps_tensors[3], init_mps_tensors[4], init_mps_tensors[5],
        init_mps_tensors[6], init_mps_tensors[7], init_mps_tensors[8],
        init_mps_tensors[9]
    ]
    intermediate_vals = dmrg_intermediates(inputs_intermediate_func)

    inputs_hvp_func = [
        init_mps_tensors[9], init_mps_tensors[8], init_mps_tensors[7],
        init_mps_tensors[6], init_mps_tensors[5], init_mps_tensors[4],
        init_mps_tensors[3], init_mps_tensors[2]
    ]
    inputs_hvp_func += mpo_tensors
    inputs_hvp_func += [
        intermediate_vals[0], init_mps_tensors[0], intermediate_vals[1],
        init_mps_tensors[1]
    ]
    inputs_hvp_func += intermediate_vals[2:]

    dmrg_hvp(inputs_hvp_func)
