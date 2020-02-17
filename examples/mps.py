import math, copy
import numpy as np
import autodiff as ad
import backend as T
from utils import CharacterGetter
from tensors.quimb_tensors import rand_mps, ham_heis_mpo, load_quimb_tensors, gauge_transform_mps
from graph_ops.graph_generator import split_einsum
from numpy.core.einsumfunc import _parse_einsum_input

BACKEND_TYPES = ['numpy']


def mps_graph(num, ranks, size=2):
    """
    Produce a graph representing the MPS:

    A-A-A-A-A-A
    | | | | | |

    Each A is a tensor, each line is a leg of the tensor diagram
    representing the contracting index.

    Each tensor is arranged as left leg, right leg, downward leg.
    The left one is arranged as right leg, downward leg, and
    the right one is arranged as left leg, downward leg.

    Parameters
    ----------
    num: Number of sites in the MPS
    size: the size of uncontracted dimensions
    ranks: a list of the size of contracted dimensions.
        The length of the list should be num-1.

    Returns
    -------
    1. a einsum node representing the MPS.
    2. The input nodes of the einsum node.
    """
    assert len(ranks) == num - 1

    A_left = ad.Variable(name='A0', shape=[ranks[0], size])
    A_right = ad.Variable(name=f'A{num-1}', shape=[ranks[-1], size])

    A_middle_list = []
    for i in range(1, num - 1):
        node = ad.Variable(name=f'A{i}', shape=[ranks[i - 1], ranks[i], size])
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

    return ad.einsum(einsum_subscripts, *A_list), A_list


def mpo_graph(num, ranks, size=2):
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

    Parameters
    ----------
    num: Number of sites in the MPO
    size: the size of uncontracted dimensions
    ranks: a list of the size of contracted dimensions.
        The length of the list should be num-1.

    Returns
    -------
    1. a einsum node representing the MPO.
    2. The input nodes of the einsum node.
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

    return ad.einsum(einsum_subscripts, *H_list), H_list


class DmrgGraph(object):
    """
    TODO
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

        mpo_shape = list(range(len(self.mpo.shape)))
        objective = ad.tensordot(mps_outer_product,
                                 self.mpo,
                                 axes=[mpo_shape, mpo_shape])

        hes = ad.hessian(objective, [intermediate])

        return intermediate, hes[0][0]


def dmrg_local_update(intermediate, hes_val, max_mps_rank):
    """
    TODO: add comments and refactor
    """
    eigvec_shape = intermediate.shape
    assert len(hes_val.shape) == 2 * len(eigvec_shape)
    assert np.array_equal(eigvec_shape, hes_val.shape[:len(eigvec_shape)])
    assert np.array_equal(eigvec_shape, hes_val.shape[len(eigvec_shape):])

    hes_val = hes_val.reshape(np.prod(eigvec_shape), -1)

    eigvals, eigvecs = T.eigh(hes_val)

    # index for smallest eigenvalue
    idx = eigvals.argsort()[0]

    eigvecs = eigvecs[:, idx].reshape(eigvec_shape)

    in_subs, out_subs, _ = _parse_einsum_input(
        (intermediate.einsum_subscripts, *intermediate.inputs))
    A_subs, B_subs = in_subs.split(',')

    map_subs_indices = dict(zip(out_subs, list(range(len(eigvec_shape)))))

    contract_char, = list(set(A_subs) - set(out_subs))

    A_uncontract_chars = list(set(A_subs) - set(contract_char))
    B_uncontract_chars = list(set(B_subs) - set(contract_char))

    A_indices = [map_subs_indices[char] for char in A_uncontract_chars]
    B_indices = [map_subs_indices[char] for char in B_uncontract_chars]

    eigvecs_mat = T.transpose(eigvecs, A_indices + B_indices).reshape(
        np.prod([eigvec_shape[i] for i in A_indices]), -1)

    U, s, VT = T.svd(eigvecs_mat)

    rank = min([max_mps_rank, eigvecs_mat.shape[0], eigvecs_mat.shape[1]])

    U, s, VT = U[:, :rank], s[:rank], VT[:rank, :]
    U = U @ T.diag(s)

    U = U.reshape([eigvec_shape[i] for i in A_indices] + [rank])
    VT = VT.reshape([rank] + [eigvec_shape[i] for i in B_indices])

    A_uncontract_str = "".join(A_uncontract_chars)
    B_uncontract_str = "".join(B_uncontract_chars)

    A = T.einsum(f"{A_uncontract_str}{contract_char}->{A_subs}", U)
    B = T.einsum(f"{contract_char}{B_uncontract_str}->{B_subs}", VT)

    return B, A


def dmrg(mpo_tensors, init_mps_tensors, max_mps_rank, sequence='R'):
    """
    TODO: add comments and notes
    """
    num = len(mpo_tensors)
    size = mpo_tensors[0].shape[1]
    mpo_ranks = [mpo_tensors[i].shape[0] for i in range(1, len(mpo_tensors))]

    mps_tensors = copy.deepcopy(init_mps_tensors)

    mps_tensors = gauge_transform_mps(mps_tensors, right=True)

    mps_ranks = [mps_tensors[i].shape[0] for i in range(1, len(mps_tensors))]

    # sequence is R
    for i in range(num - 1):

        dg = DmrgGraph(num, mpo_ranks, mps_ranks, size)

        feed_dict = dict(zip(dg.mpo_inputs, mpo_tensors))
        feed_dict.update(dict(zip(dg.mps_inputs, mps_tensors)))

        hes_val, = dg.executors[i].run(feed_dict=feed_dict)

        # TODO: Update the two sites of mps
        mps_tensors[i], mps_tensors[i + 1] = dmrg_local_update(
            dg.intermediates[i], hes_val, max_mps_rank)

        mps_ranks[i] = mps_tensors[i + 1].shape[0]

    return mps_tensors


if __name__ == "__main__":
    # mps = mps_graph(4, 10)
    # mpo = mpo_graph(4, 10)
    mpo_tensors = ham_heis_mpo(num=4)
    mps_tensors = rand_mps(num=4, rank=5, size=2)
    dmrg(mpo_tensors, mps_tensors, max_mps_rank=5)
