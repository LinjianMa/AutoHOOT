import autodiff as ad
import numpy as np
import backend as T
from source import SourceToSource

BACKEND_TYPES = ['numpy', 'ctf']
BACKEND_TYPES = ['numpy']


def test_add_jacobian():

    for datatype in BACKEND_TYPES:
        T.set_backend(datatype)

        x1 = ad.Variable(name="x1", shape=[2, 2])
        x2 = ad.Variable(name="x2", shape=[2, 2])
        y = x1 + x2

        jacobian_x2, = ad.jacobians(y, [x2])

        executor = ad.Executor([y, jacobian_x2])

        x1_val = T.tensor([[1, 1], [1, 1]])
        x2_val = T.tensor([[1, 1], [1, 1]])
        y_val, jacobian_x2_val = executor.run(feed_dict={
            x1: x1_val,
            x2: x2_val
        })

        I = T.identity(2)
        expected_jacobian_x2_val = T.einsum("ac,bd->abcd", I, I)

        assert isinstance(y, ad.Node)
        assert isinstance(jacobian_x2, ad.Node)
        assert T.array_equal(y_val, x1_val + x2_val)
        assert T.array_equal(jacobian_x2_val, expected_jacobian_x2_val)


def test_add_jacobian_scalar():

    for datatype in BACKEND_TYPES:
        T.set_backend(datatype)

        x1 = ad.Variable(name="x1", shape=[])
        x2 = ad.Variable(name="x2", shape=[])
        y = x1 + x2

        jacobian_x2, = ad.jacobians(y, [x2])

        executor = ad.Executor([y, jacobian_x2])

        x1_val = T.tensor(1.)
        x2_val = T.tensor(1.)
        y_val, jacobian_x2_val = executor.run(feed_dict={
            x1: x1_val,
            x2: x2_val
        })

        expected_jacobian_x2_val = T.tensor(1.)

        assert isinstance(y, ad.Node)
        assert isinstance(jacobian_x2, ad.Node)
        assert T.array_equal(y_val, x1_val + x2_val)
        assert T.array_equal(jacobian_x2_val, expected_jacobian_x2_val)
