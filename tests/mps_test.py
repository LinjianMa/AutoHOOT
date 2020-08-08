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

import autodiff as ad
import backend as T
import quimb.tensor as qtn

from examples.mps import dmrg, dmrg_shared_exec, MpoGraph, MpsGraph, DmrgGraph
from examples.dmrg_iterative_solve import dmrg_shared_exec_iterative_solve
from tests.test_utils import tree_eq
from tensors.quimb_tensors import rand_mps, gauge_transform_mps, load_quimb_tensors


def test_mps(backendopt):
    for datatype in backendopt:
        T.set_backend(datatype)

        mps_graph = MpsGraph.create(4, ranks=[5, 6, 7])
        executor = ad.Executor([mps_graph.output])

        expect_mps = ad.einsum('ab,acd,cef,eg->bdfg', *mps_graph.inputs)

        assert tree_eq(mps_graph.output, expect_mps, mps_graph.inputs)


def test_mpo(backendopt):
    for datatype in backendopt:
        T.set_backend(datatype)

        mpo_graph = MpoGraph.create(4, ranks=[5, 6, 7])
        executor = ad.Executor([mpo_graph.output])

        expect_mpo = ad.einsum('abc,adef,dghi,gjk->behjcfik',
                               *mpo_graph.inputs)

        assert tree_eq(mpo_graph.output, expect_mpo, mpo_graph.inputs)


def test_gauge_transform_right(backendopt):
    for datatype in backendopt:
        T.set_backend(datatype)

        tensors_input = rand_mps(num=4, rank=4, size=2)
        tensors = gauge_transform_mps(tensors_input)

        # make sure the transformation will not change the mps results
        mps = T.einsum('ab,acd,cef,eg->bdfg', *tensors_input)
        mps_gauge = T.einsum('ab,acd,cef,eg->bdfg', *tensors)
        assert T.norm(mps - mps_gauge) < 1e-8

        dim = len(tensors_input)

        # test all tensors except the left one's orthogonality
        for i in range(1, dim - 1):
            inner = T.einsum("abc,dbc->ad", tensors[i], tensors[i])
            assert T.norm(inner - T.identity(inner.shape[0])) < 1e-8

        inner = T.einsum("ab,cb->ac", tensors[dim - 1], tensors[dim - 1])
        assert T.norm(inner - T.identity(inner.shape[0])) < 1e-8


def test_gauge_transform_left(backendopt):
    for datatype in backendopt:
        T.set_backend(datatype)

        tensors_input = rand_mps(num=4, rank=4, size=2)
        tensors = gauge_transform_mps(tensors_input, right=False)

        # make sure the transformation will not change the mps results
        mps = T.einsum('ab,acd,cef,eg->bdfg', *tensors_input)
        mps_gauge = T.einsum('ab,acd,cef,eg->bdfg', *tensors)
        assert T.norm(mps - mps_gauge) < 1e-8

        dim = len(tensors_input)

        # test all tensors except the right one's orthogonality
        inner = T.einsum("ab,cb->ac", tensors[0], tensors[0])
        assert T.norm(inner - T.identity(inner.shape[0])) < 1e-8

        for i in range(1, dim - 1):
            inner = T.einsum("abc,adc->bd", tensors[i], tensors[i])
            assert T.norm(inner - T.identity(inner.shape[0])) < 1e-8


def test_dmrg_one_sweep():
    max_mps_rank = 5
    num = 4
    T.set_backend("numpy")

    h = qtn.MPO_ham_heis(num)
    dmrg_quimb = qtn.DMRG2(h, bond_dims=[max_mps_rank])

    h_tensors = load_quimb_tensors(h)
    mps_tensors = load_quimb_tensors(dmrg_quimb.state)

    # dmrg based on ad
    mps_tensors, energy = dmrg(h_tensors,
                               mps_tensors,
                               max_mps_rank=max_mps_rank)

    # dmrg based on quimb
    opts = {'max_bond': max_mps_rank}
    quimb_energy = dmrg_quimb.sweep_right(canonize=True, verbosity=0, **opts)

    # We only test on energy (lowest eigenvalue of h), rather than the output
    # mps (eigenvector), because the eigenvectors can vary a lot while keeping the
    # eigenvalue unchanged.
    assert (abs(energy - quimb_energy) < 1e-8)


def test_dmrg_shared_exec_one_sweep():
    max_mps_rank = 5
    num = 4
    T.set_backend("numpy")

    h = qtn.MPO_ham_heis(num)
    dmrg_quimb = qtn.DMRG2(h, bond_dims=[max_mps_rank])

    h_tensors = load_quimb_tensors(h)
    mps_tensors = load_quimb_tensors(dmrg_quimb.state)

    # dmrg based on ad
    mps_tensors, energy = dmrg_shared_exec(h_tensors,
                                           mps_tensors,
                                           max_mps_rank=max_mps_rank)

    # dmrg based on quimb
    opts = {'max_bond': max_mps_rank}
    quimb_energy = dmrg_quimb.sweep_right(canonize=True, verbosity=0, **opts)

    # We only test on energy (lowest eigenvalue of h), rather than the output
    # mps (eigenvector), because the eigenvectors can vary a lot while keeping the
    # eigenvalue unchanged.
    assert (abs(energy - quimb_energy) < 1e-8)


def test_dmrg_shared_exec_graph():

    from graph_ops.graph_transformer import simplify
    from graph_ops.graph_als_optimizer import generate_sequential_optimal_tree
    from utils import find_topo_sort

    num, rank, size = 4, 3, 2
    mpo_ranks = [rank for i in range(1, num)]
    mps_ranks = [rank for i in range(1, num)]

    dg = DmrgGraph.create(num, mpo_ranks, mps_ranks, size)
    for i, hes in enumerate(dg.hessians):
        dg.hessians[i] = simplify(hes)
        assert isinstance(hes, ad.EinsumNode)
    dg.hessians = generate_sequential_optimal_tree(dg.hessians, dg.mps_inputs)

    # 8 input variables (4 H term in MPO, 4 A term in MPS), 7 einsum nodes
    assert len(find_topo_sort(dg.hessians)) == 15


def test_dmrg_shared_exec_iterative_solve_one_sweep():
    max_mps_rank = 5
    num = 7
    mpo_rank = 5
    size = 5
    num_iter = 10
    T.set_backend("numpy")

    h = qtn.MPO_rand_herm(num, mpo_rank, size)
    dmrg_quimb = qtn.DMRG2(h, bond_dims=[max_mps_rank])

    h_tensors = load_quimb_tensors(h)
    mps_tensors = load_quimb_tensors(dmrg_quimb.state)

    # dmrg based on ad
    mps_tensors, energy = dmrg_shared_exec_iterative_solve(
        h_tensors, mps_tensors, num_iter=num_iter, max_mps_rank=max_mps_rank)

    # dmrg based on quimb
    opts = {'max_bond': max_mps_rank}
    for _ in range(num_iter):
        quimb_energy = dmrg_quimb.sweep_right(canonize=True,
                                              verbosity=0,
                                              **opts)

    # We only test on energy (lowest eigenvalue of h), rather than the output
    # mps (eigenvector), because the eigenvectors can vary a lot while keeping the
    # eigenvalue unchanged.
    assert (abs(energy - quimb_energy) < 1e-5)
