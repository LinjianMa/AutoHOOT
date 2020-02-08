import autodiff as ad
import backend as T
from examples.mps import mps_graph, mpo_graph
from tests.test_utils import tree_eq

BACKEND_TYPES = ['numpy', 'ctf']


def test_mps():
    for datatype in BACKEND_TYPES:
        T.set_backend(datatype)

        mps, inputs = mps_graph(4, 10)
        executor = ad.Executor([mps])

        expect_mps = ad.einsum('ab,acd,cef,eg->bdfg', *inputs)

        assert tree_eq(mps, expect_mps, inputs)


def test_mpo():
    for datatype in BACKEND_TYPES:
        T.set_backend(datatype)

        mpo, inputs = mpo_graph(4, 5)
        executor = ad.Executor([mpo])

        expect_mpo = ad.einsum('abc,adef,dghi,gjk->behjcfik', *inputs)

        assert tree_eq(mpo, expect_mpo, inputs)
