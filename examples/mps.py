import math, copy
import attr
import numpy as np
import autodiff as ad
import backend as T
from utils import CharacterGetter
from tensors.quimb_tensors import rand_mps, ham_heis_mpo, load_quimb_tensors, gauge_transform_mps
from graph_ops.graph_generator import split_einsum
from numpy.core.einsumfunc import _parse_einsum_input

BACKEND_TYPES = ['numpy']


@attr.s()
class MpsGraph(object):
    """
    Produce a graph representing the MPS:

    A-A-A-A-A-A
    | | | | | |

    Each A is a tensor, each line is a leg of the tensor diagram
    representing the contracting index.

    Each tensor is arranged as left leg, right leg, downward leg.
    The left one is arranged as right leg, downward leg, and
    the right one is arranged as left leg, downward leg.

    Variables: 
    -------
    1. a einsum node representing the MPS.
    2. The input nodes of the einsum node.
    
    """
    output = attr.ib()
    inputs = attr.ib(default=[])

    @classmethod
    def create(cls, num, ranks, size=2):
        """
        Parameters
        ----------
        num: Number of sites in the MPS
        size: the size of uncontracted dimensions
        ranks: a list of the size of contracted dimensions.
            The length of the list should be num-1.
        """

        assert len(ranks) == num - 1

        A_left = ad.Variable(name='A0', shape=[ranks[0], size])
        A_right = ad.Variable(name=f'A{num-1}', shape=[ranks[-1], size])

        A_middle_list = []
        for i in range(1, num - 1):
            node = ad.Variable(name=f'A{i}',
                               shape=[ranks[i - 1], ranks[i], size])
            A_middle_list.append(node)

        untracted_subs_list = []
        cg = CharacterGetter()

        # set subscripts for all the node
        A_left.subscripts = f"{cg.getchar()}{cg.getchar()}"
        prev_char = A_left.subscripts[0]
        untracted_subs_list.append(A_left.subscripts[1])

        for node in A_middle_list:
            node.subscripts = f"{prev_char}{cg.getchar()}{cg.getchar()}"
            prev_char = node.subscripts[1]
            untracted_subs_list.append(node.subscripts[2])

        A_right.subscripts = f"{prev_char}{cg.getchar()}"
        untracted_subs_list.append(A_right.subscripts[1])

        # produce output
        A_list = [A_left] + A_middle_list + [A_right]
        out_subs = "".join(untracted_subs_list)
        input_subs = ','.join([node.subscripts for node in A_list])
        einsum_subscripts = input_subs + '->' + out_subs

        # clear all the subscripts
        for node in A_list:
            node.subscripts = None

        return cls(ad.einsum(einsum_subscripts, *A_list), A_list)


def mps_graph(num, ranks, size=2):
    tmp = MpsGraph.create(num, ranks, size)
    return tmp.output, tmp.inputs


@attr.s()
class MpoGraph(object):
    """
    Produce a graph representing the MPO:

    | | | | | |
    H-H-H-H-H-H
    | | | | | |

    Each A is a tensor, each line is a leg of the tensor diagram
    representing the contracting index.

    Each tensor is arranged as left leg, right leg, upward leg, downward leg.
    The left one is arranged as right leg, upward leg, downward leg and
    the right one is arranged as left leg, upward leg, downward leg.

    Returns
    -------
    1. a einsum node representing the MPO.
    2. The input nodes of the einsum node.
    """
    output = attr.ib()
    inputs = attr.ib(default=[])

    @classmethod
    def create(cls, num, ranks, size=2):
        """
        Parameters
        ----------
        num: Number of sites in the MPO
        size: the size of uncontracted dimensions
        ranks: a list of the size of contracted dimensions.
            The length of the list should be num-1.

        """
        assert len(ranks) == num - 1

        H_left = ad.Variable(name='H0', shape=[ranks[0], size, size])
        H_right = ad.Variable(name=f'H{num-1}', shape=[ranks[-1], size, size])

        H_middle_list = []
        for i in range(1, num - 1):
            node = ad.Variable(name=f'H{i}',
                               shape=[ranks[i - 1], ranks[i], size, size])
            H_middle_list.append(node)

        up_subs_list = []
        down_subs_list = []
        cg = CharacterGetter()

        # set subscripts for all the node
        H_left.subscripts = f"{cg.getchar()}{cg.getchar()}{cg.getchar()}"
        prev_char = H_left.subscripts[0]
        up_subs_list.append(H_left.subscripts[1])
        down_subs_list.append(H_left.subscripts[2])

        for node in H_middle_list:
            node.subscripts = f"{prev_char}{cg.getchar()}{cg.getchar()}{cg.getchar()}"
            prev_char = node.subscripts[1]
            up_subs_list.append(node.subscripts[2])
            down_subs_list.append(node.subscripts[3])

        H_right.subscripts = f"{prev_char}{cg.getchar()}{cg.getchar()}"
        up_subs_list.append(H_right.subscripts[1])
        down_subs_list.append(H_right.subscripts[2])

        # produce output
        H_list = [H_left] + H_middle_list + [H_right]
        up_subs = "".join(up_subs_list)
        down_subs = "".join(down_subs_list)
        input_subs = ','.join([node.subscripts for node in H_list])
        einsum_subscripts = input_subs + '->' + up_subs + down_subs

        # clear all the subscripts
        for node in H_list:
            node.subscripts = None

        return cls(ad.einsum(einsum_subscripts, *H_list), H_list)


def mpo_graph(num, ranks, size=2):
    tmp = MpoGraph.create(num, ranks, size)
    return tmp.output, tmp.inputs


class DmrgGraph(object):
    """
    Produce a graph representing the DMRG algorithm.

    Note: here we use hessian calculation to get the contractions
       among the mpo and the mps except the intermediates.

    Parameters
    ----------
    num: number of tensors in mpo and mps
    mpo_ranks: an array containing mpo ranks
    mps_ranks: an array containing mps ranks
    size: untracted legs dimension size in both mpo and mps

    Variables
    ---------
    mpo: the einsum node representing the mpo graph
    mpo_inputs: an array containing mpo input nodes
    mps: the einsum node representing the mps graph
    mps_inputs: an array containing mps input nodes
    intermediates: an array of einsum nodes taking hessian w.r.t.
    executors: an array of executors to calculate hessians

    """
    def __init__(self, num, mpo_ranks, mps_ranks, size):
        self.mpo, self.mpo_inputs = mpo_graph(num, mpo_ranks, size)
        self.mps, self.mps_inputs = mps_graph(num, mps_ranks, size)

        self.intermediates, self.executors = [], []
        for i in range(num - 1):
            intermediate, hes = self._get_sub_hessian(i)
            executor = ad.Executor([hes])
            self.intermediates.append(intermediate)
            self.executors.append(executor)

    def _get_sub_hessian(self, index):

        # rebuild mps graph
        intermediate_set = {self.mps_inputs[index], self.mps_inputs[index + 1]}
        split_input_nodes = list(set(self.mps_inputs) - intermediate_set)
        mps = split_einsum(self.mps, split_input_nodes)

        # get the intermediate node
        intermediate, = [
            node for node in mps.inputs if isinstance(node, ad.EinsumNode)
        ]

        mps_outer_product = ad.tensordot(mps, mps, axes=[[], []])

        mpo_axes = list(range(len(self.mpo.shape)))
        objective = ad.tensordot(mps_outer_product,
                                 self.mpo,
                                 axes=[mpo_axes, mpo_axes])

        hes = ad.hessian(objective, [intermediate])

        return intermediate, hes[0][0]


def dmrg_local_update(intermediate, hes_val, max_mps_rank):
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

    index_input_0 = int(''.join(x for x in inputs[0].name if x.isdigit()))
    index_input_1 = int(''.join(x for x in inputs[1].name if x.isdigit()))

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
    assert len(hes_val.shape) == 2 * len(eigvec_shape)
    assert np.array_equal(eigvec_shape, hes_val.shape[:len(eigvec_shape)])
    assert np.array_equal(eigvec_shape, hes_val.shape[len(eigvec_shape):])

    # get the eigenvector of the hessian matrix
    hes_val_mat = T.reshape(hes_val, (np.prod(eigvec_shape), -1))
    eigvals, eigvecs = T.eigh(hes_val_mat)
    # index for smallest eigenvalue
    # idx = eigvals.argsort()[0]
    idx = T.argmin(eigvals)
    eigvecs = T.reshape(eigvecs[:, idx], eigvec_shape)

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

    outprod = T.einsum(f"{left_subs},{right_subs}->{out_subs}", left, right)
    outprod_indices = list(range(len(eigvec_shape)))
    hvp = T.tensordot(hes_val, outprod, [outprod_indices, outprod_indices])
    eig_val = T.tensordot(hvp, outprod, [outprod_indices, outprod_indices]) / 2

    return left, right, eig_val


def dmrg(mpo_tensors, init_mps_tensors, max_mps_rank, num_iter=1,
         sequence='R'):
    """
    Perform DMRG iterations.

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

    # sequence is R
    for iter in range(num_iter):

        mps_tensors = gauge_transform_mps(mps_tensors, right=True)
        mps_ranks = [
            mps_tensors[i].shape[0] for i in range(1, len(mps_tensors))
        ]

        for i in range(num - 1):

            # TODO: we rebuild the graph every time we update the mps sites.
            # The reason is that every update can change the shape of the
            # mps tensor. Can optimize this step later.
            dg = DmrgGraph(num, mpo_ranks, mps_ranks, size)

            feed_dict = dict(zip(dg.mpo_inputs, mpo_tensors))
            feed_dict.update(dict(zip(dg.mps_inputs, mps_tensors)))

            hes_val, = dg.executors[i].run(feed_dict=feed_dict)

            # Update the two sites of mps
            mps_tensors[i], mps_tensors[i + 1], eig_val = dmrg_local_update(
                dg.intermediates[i], hes_val, max_mps_rank)

            # update the rank
            mps_ranks[i] = mps_tensors[i + 1].shape[0]

        print(f'At iteration {iter} the smallest eigenvalue is: {eig_val}')

    return mps_tensors, eig_val


if __name__ == "__main__":
    # mps = mps_graph(4, 10)
    # mpo = mpo_graph(4, 10)
    mpo_tensors = ham_heis_mpo(num=6)
    mps_tensors = rand_mps(num=6, rank=2, size=2)
    dmrg(mpo_tensors, mps_tensors, max_mps_rank=20, num_iter=2)
