import copy, attr
import numpy as np
import autodiff as ad
import backend as T
import scipy.sparse.linalg as spla

from numpy.core.einsumfunc import _parse_einsum_input
from tensors.quimb_tensors import gauge_transform_mps
from examples.mps import MpsGraph, MpoGraph, dmrg_local_update
from graph_ops.graph_generator import split_einsum
from graph_ops.graph_transformer import simplify
from utils import update_variables
from graph_ops.graph_als_optimizer import generate_sequential_optimal_tree


@attr.s()
class DmrgImplicitUpdateGraph(object):
    """
    Produce a graph representing the DMRG algorithm, where each site is solved via iterative solve.

    Parameters
    ----------
    num: number of tensors in mpo and mps.
    mpo_ranks: an array containing mpo ranks.
    mps_ranks: an array containing mps ranks.
    size: untracted legs dimension size in both mpo and mps.

    Variables
    ---------
    mpo_inputs: inputs of the MpoGraph.
    mps_inputs: inputs the MpsGraph.
    intermediates: an array of einsum nodes taking hvp w.r.t.
    hvps: a list of graphs for hvps.
    vnodes: a list of graphs for the vectors in hvps.
    """
    mpo_inputs = attr.ib()
    mps_inputs = attr.ib()
    intermediates = attr.ib(default=[])
    hvps = attr.ib(default=[])
    vnodes = attr.ib(default=[])

    def update_graph(self, num, mpo_ranks, mps_ranks, size):
        self.mpo_inputs = MpoGraph.create(num, mpo_ranks, size).inputs
        self.mps_inputs = MpsGraph.create(num, mps_ranks, size).inputs
        update_variables(self.intermediates, self.mpo_inputs + self.mps_inputs)

        self.vnodes = [
            ad.Variable(name=f"v{i}", shape=node.shape)
            for (i, node) in enumerate(self.intermediates)
        ]
        update_variables(self.hvps,
                         self.mpo_inputs + self.mps_inputs + self.vnodes)

    @classmethod
    def create(cls, num, mpo_ranks, mps_ranks, size):
        mpo_graph = MpoGraph.create(num, mpo_ranks, size)
        mps_graph = MpsGraph.create(num, mps_ranks, size)

        intermediates, hvps, vnodes = [], [], []
        for i in range(num - 1):
            intermediate, hvp, vnode = cls._get_sub_hvp(
                i, mpo_graph, mps_graph)
            intermediates.append(intermediate)
            hvps.append(hvp)
            vnodes.append(vnode)

        return cls(mpo_graph.inputs, mps_graph.inputs, intermediates, hvps,
                   vnodes)

    @classmethod
    def _get_sub_hvp(cls, index, mpo_graph, mps_graph):

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

        # The 0.5 factor makes sure that the hvp can be written as an einsum
        objective = 0.5 * ad.tensordot(
            mps_outer_product, mpo_graph.output, axes=[mpo_axes, mpo_axes])
        vnode = ad.Variable(name=f"v{index}", shape=intermediate.shape)
        hvp, = ad.hvp(objective, [intermediate], [vnode])

        return intermediate, hvp, vnode


class DMRGLinearOperator(spla.LinearOperator):
    """
    An operator used in the scipy solve.
    """
    def __init__(self, dmrg_graph, executor, index, feed_dict):

        col_size = np.prod(dmrg_graph.hvps[index].shape)
        super().__init__(dtype=np.float64, shape=(col_size, col_size))

        self.eigvec_shape = dmrg_graph.hvps[index].shape
        self.dmrg_graph = dmrg_graph
        self.executor = executor
        self.index = index
        self.feed_dict = feed_dict
        self.initial_matvec = True

    def _matvec(self, vec):
        dg = self.dmrg_graph
        in_data = T.reshape(vec, self.eigvec_shape)
        self.feed_dict.update({dg.vnodes[self.index]: in_data})

        if self.initial_matvec:
            if self.index == 0:
                reset_graph = True
                evicted_inputs = []
            else:
                reset_graph = False
                evicted_inputs = [
                    dg.mps_inputs[self.index - 1], dg.vnodes[self.index]
                ]
            self.initial_matvec = False
        else:
            reset_graph = False
            evicted_inputs = [dg.vnodes[self.index]]

        out_data, = self.executor.run(feed_dict=self.feed_dict,
                                      reset_graph=reset_graph,
                                      evicted_inputs=evicted_inputs,
                                      out_nodes=[dg.hvps[self.index]])
        return out_data.ravel()


def dmrg_shared_exec_iterative_solve(mpo_tensors,
                                     init_mps_tensors,
                                     max_mps_rank,
                                     num_iter=1,
                                     sequence='R'):
    """
    Perform DMRG iterations with shared execution and iterative solve.
    """
    if sequence != "R":
        raise NotImplementedError

    num = len(mpo_tensors)
    size = mpo_tensors[0].shape[1]
    mpo_ranks = [mpo_tensors[i].shape[0] for i in range(1, len(mpo_tensors))]

    mps_tensors = copy.deepcopy(init_mps_tensors)
    mps_ranks = [mps_tensors[i].shape[0] for i in range(1, len(mps_tensors))]

    dg = DmrgImplicitUpdateGraph.create(num, mpo_ranks, mps_ranks, size)
    for i, hvp in enumerate(dg.hvps):
        dg.hvps[i] = simplify(hvp)
        assert isinstance(hvp, ad.EinsumNode)
    dg.hvps = generate_sequential_optimal_tree(dg.hvps, dg.mps_inputs)

    executor_hvps = ad.Executor(dg.hvps)
    executor_intermediates = ad.Executor(dg.intermediates)

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

            intermediate, = executor_intermediates.run(
                feed_dict=feed_dict, out_nodes=[dg.intermediates[i]])

            # Calculate the eigenvector using the implicit solver.
            # Note: This only supports NumPy datatype.
            # TODO: Add a general Lanczos solver that adapts to all the backends.
            operator = DMRGLinearOperator(dg, executor_hvps, i, feed_dict)
            # Reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.eigsh.html
            eig_vals, eigvecs = spla.eigsh(operator,
                                           k=1,
                                           ncv=4,
                                           tol=1e-3,
                                           which='SA',
                                           v0=intermediate.ravel())
            eig_val, eigvec = eig_vals[0], eigvecs[:, 0]
            eigvec = T.reshape(eigvec, dg.intermediates[i].shape)

            # Update the two sites of mps
            mps_tensors[i], mps_tensors[i + 1] = dmrg_local_update(
                dg.intermediates[i], eigvec, max_mps_rank)

            # update the rank
            mps_ranks[i] = mps_tensors[i + 1].shape[0]
            print(f'At site {i}, the smallest eigenvalue is: {eig_val}')

        print(f'At iteration {iter} the smallest eigenvalue is: {eig_val}')
    return mps_tensors, eig_val
