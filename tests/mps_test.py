import autodiff as ad
import backend as T
import quimb.tensor as qtn
from examples.mps import dmrg, dmrg_shared_exec, MpoGraph, MpsGraph
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


def test_dmrg_one_sweep(backendopt):
    max_mps_rank = 5
    num = 4
    for datatype in backendopt:

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
        quimb_energy = dmrg_quimb.sweep_right(canonize=True,
                                              verbosity=0,
                                              **opts)

        # We only test on energy (lowest eigenvalue of h), rather than the output
        # mps (eigenvector), because the eigenvectors can vary a lot while keeping the
        # eigenvalue unchanged.
        assert (abs(energy - quimb_energy) < 1e-8)


def test_dmrg_shared_exec_one_sweep(backendopt):
    max_mps_rank = 5
    num = 4
    for datatype in backendopt:

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
        quimb_energy = dmrg_quimb.sweep_right(canonize=True,
                                              verbosity=0,
                                              **opts)

        # We only test on energy (lowest eigenvalue of h), rather than the output
        # mps (eigenvector), because the eigenvectors can vary a lot while keeping the
        # eigenvalue unchanged.
        assert (abs(energy - quimb_energy) < 1e-8)
