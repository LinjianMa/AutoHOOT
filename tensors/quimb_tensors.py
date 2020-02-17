import backend as T
import quimb.tensor as qtn

def load_quimb_tensors(network):
    tensors = []

    for tensor in network.tensor_map.values():
        tensors.append(T.tensor(tensor.data))

    return tensors


def rand_mps(num, rank, size=2):
    """
    Generate random MPS.
    """
    mps = qtn.MPS_rand_state(num, rank, phys_dim=size)
    return load_quimb_tensors(mps)


def ham_heis_mpo(num):
    """
    Heisenberg Hamiltonian in MPO form.
    Note: the rank of Heisenberg is set to be 5,
    and size is set to be 2 implicitly.
    """
    mpo = qtn.MPO_ham_heis(num)
    return load_quimb_tensors(mpo)


def gauge_transform_mps(tensors, right=True):
    """
    Perform gause transformation on the MPS

    NOTE: currently this function doesn't support CTF backend.
    Reference: https://tensornetwork.org/mps/#toc_7

    Parameters
    ----------
    tensors: array of tensors representing the MPS
    right: direction of the transformation. If true,
        for the output mps, the diagram for its inner product will be:
                                                
                 o-<-<-<-<-<-<-<-<         o-
                 | | | | | | | | |   =     | | (inner product of o)
                 o-<-<-<-<-<-<-<-<         o-
        if False, the diagram of its inner product will be:

                 >->->->->->->->-o          -o
                 | | | | | | | | |   =     | | (inner product of o)
                 >->->->->->->->-o          -o

        here > or < denotes a tensor that is left / right orthogonal.

    Returns
    -------
    1. An array of tensors representing the MPS
    """
    mps = qtn.MatrixProductState(tensors, shape='lrp')

    if right:
        mps.right_canonize()
    else:
        mps.left_canonize()

    tensors = []
    for tensor in mps.tensor_map.values():
        tensors.append(T.tensor(tensor.data))

    return tensors
