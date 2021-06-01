# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from autohoot import backend as T
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
    np_tensors = [T.to_numpy(tensor) for tensor in tensors]
    mps = qtn.MatrixProductState(np_tensors, shape='lrp')

    if right:
        mps.right_canonize()
    else:
        mps.left_canonize()

    tensors = []
    for tensor in mps.tensor_map.values():
        tensors.append(T.tensor(tensor.data))

    return tensors
