import backend as T
import numpy as np
import quimb.tensor as qtn


def init_rand_3d(s, R):
    A = T.random((s, R))
    B = T.random((s, R))
    C = T.random((s, R))
    input_tensor = T.einsum("ia,ja->ija", A, B)
    input_tensor = T.einsum("ija,ka->ijk", input_tensor, C)
    A = T.random((s, R))
    B = T.random((s, R))
    C = T.random((s, R))
    return [A, B, C, input_tensor]


def init_rand_tucker(dim, size, rank):
    X = T.random([size for _ in range(dim)])
    core = T.random([rank for _ in range(dim)])

    A_list = []
    for i in range(dim):
        A_list.append(T.random((size, rank)))

    return A_list, core, X


def rand_mps(num, rank, size=2):
    """
    Generate random MPS.
    """
    mps = qtn.MPS_rand_state(num, rank, phys_dim=size)
    tensors = []

    for tensor in mps.tensor_map.values():
        tensors.append(T.tensor(tensor.data))

    return tensors


def ham_heis_mpo(num):
    """
    Heisenberg Hamiltonian in MPO form.
    Note: the rank of Heisenberg is set to be 5,
    and size is set to be 2 implicitly.
    """
    mpo = qtn.MPO_ham_heis(num)
    tensors = []

    for tensor in mpo.tensor_map.values():
        tensors.append(T.tensor(tensor.data))

    return tensors
