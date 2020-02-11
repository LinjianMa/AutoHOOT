import math
import autodiff as ad
import backend as T
from utils import CharacterGetter

BACKEND_TYPES = ['numpy']


def mps_graph(num, rank, size=2):
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
    rank: the size of contracted dimensions.

    Returns
    -------
    1. a einsum node representing the MPS.
    2. The input nodes of the einsum node.
    """
    A_left = ad.Variable(name='A0', shape=[rank, size])
    A_right = ad.Variable(name=f'A{num-1}', shape=[rank, size])

    A_middle_list = []
    for i in range(1, num - 1):
        node = ad.Variable(name=f'A{i}', shape=[rank, rank, size])
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


def mpo_graph(num, rank, size=2):
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
    rank: the size of contracted dimensions.

    Returns
    -------
    1. a einsum node representing the MPO.
    2. The input nodes of the einsum node.
    """
    H_left = ad.Variable(name='H0', shape=[rank, size, size])
    H_right = ad.Variable(name=f'H{num-1}', shape=[rank, size, size])

    H_middle_list = []
    for i in range(1, num - 1):
        node = ad.Variable(name=f'H{i}', shape=[rank, rank, size, size])
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


if __name__ == "__main__":
    mps = mps_graph(4, 10)
    mpo = mpo_graph(4, 10)
