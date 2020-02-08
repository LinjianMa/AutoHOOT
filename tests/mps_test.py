import autodiff as ad
import backend as T
from examples.mps import mps_graph, mpo_graph
from tests.test_utils import tree_eq
from tensors.quimb_tensors import rand_mps, gauge_transform_mps

BACKEND_TYPES = ['numpy']


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


def test_gauge_transform_right():
    for datatype in BACKEND_TYPES:
        T.set_backend(datatype)

        tensors = rand_mps(num=4, rank=4, size=2)
        tensors = gauge_transform_mps(tensors)

        # test last two tensors' orthogonality
        inner = T.einsum("abc,dbc->ad", tensors[2], tensors[2])
        assert T.norm(inner- T.identity(inner.shape[0])) < 1e-8

        inner = T.einsum("ab,cb->ac", tensors[3], tensors[3])
        assert T.norm(inner- T.identity(inner.shape[0])) < 1e-8


def test_gauge_transform_left():
    for datatype in BACKEND_TYPES:
        T.set_backend(datatype)

        tensors = rand_mps(num=4, rank=4, size=2)
        tensors = gauge_transform_mps(tensors, right=False)

        # test first two tensors' orthogonality
        inner = T.einsum("ab,cb->ac", tensors[0], tensors[0])
        assert T.norm(inner- T.identity(inner.shape[0])) < 1e-8

        inner = T.einsum("abc,adc->bd", tensors[1], tensors[1])
        assert T.norm(inner- T.identity(inner.shape[0])) < 1e-8
