import autodiff as ad
import backend as T
from examples.mps import mps_graph, mpo_graph
from tensors.synthetic_tensors import rand_mps, ham_heis_mpo

BACKEND_TYPES = ['numpy', 'ctf']


def test_mps():
    for datatype in BACKEND_TYPES:
        T.set_backend(datatype)

        mps, inputs = mps_graph(4, 10)
        executor = ad.Executor([mps])

        inputs_val = rand_mps(4, 10)
        mps_val, = executor.run(feed_dict=dict(zip(inputs, inputs_val)))

        expect_mps_val = T.einsum('ab,bcd,def,fg->aceg', inputs_val[0],
                                  inputs_val[1], inputs_val[2], inputs_val[3])

        assert T.norm(mps_val - expect_mps_val) < 1e-8


def test_mpo():
    for datatype in BACKEND_TYPES:
        T.set_backend(datatype)

        mpo, inputs = mpo_graph(4, 5)
        executor = ad.Executor([mpo])

        # Note: the rank of Heisenberg is set to be 5 implicitly.
        inputs_val = ham_heis_mpo(4)
        mpo_val, = executor.run(feed_dict=dict(zip(inputs, inputs_val)))

        expect_mpo_val = T.einsum('abc,cdef,fghi,ijk->adgjbehk', inputs_val[0],
                                  inputs_val[1], inputs_val[2], inputs_val[3])

        assert T.norm(mpo_val - expect_mpo_val) < 1e-8
