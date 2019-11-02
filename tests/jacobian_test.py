import autodiff as ad
import numpy as np
import backend as T
from source import SourceToSource

BACKEND_TYPES = ['numpy', 'ctf']


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


def test_sub_jacobian():

    for datatype in BACKEND_TYPES:
        T.set_backend(datatype)

        x1 = ad.Variable(name="x1", shape=[2, 2])
        x2 = ad.Variable(name="x2", shape=[2, 2])
        y = x1 - x2

        jacobian_x1, jacobian_x2 = ad.jacobians(y, [x1, x2])

        executor = ad.Executor([y, jacobian_x1, jacobian_x2])

        x1_val = T.tensor([[1, 1], [1, 1]])
        x2_val = T.tensor([[1, 1], [1, 1]])
        y_val, jacobian_x1_val, jacobian_x2_val = executor.run(feed_dict={
            x1: x1_val,
            x2: x2_val
        })

        I = T.identity(2)
        expected_jacobian_x1_val = T.einsum("ac,bd->abcd", I, I)
        expected_jacobian_x2_val = -T.einsum("ac,bd->abcd", I, I)

        assert isinstance(y, ad.Node)
        assert isinstance(jacobian_x2, ad.Node)
        assert T.array_equal(y_val, x1_val - x2_val)
        assert T.array_equal(jacobian_x1_val, expected_jacobian_x1_val)
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

        expected_jacobian_x2_val = T.tensor([[1.]])

        assert isinstance(y, ad.Node)
        assert isinstance(jacobian_x2, ad.Node)
        assert T.array_equal(y_val, x1_val + x2_val)
        assert T.array_equal(jacobian_x2_val, expected_jacobian_x2_val)


def test_add_jacobian_w_chain():

    for datatype in BACKEND_TYPES:
        T.set_backend(datatype)

        x1 = ad.Variable(name="x1", shape=[2, 2])
        x2 = ad.Variable(name="x2", shape=[2, 2])
        x3 = ad.Variable(name="x3", shape=[2, 2])
        y = x1 + x2
        z = y + x3

        jacobian_x2, = ad.jacobians(z, [x2])

        executor = ad.Executor([z, jacobian_x2])

        x1_val = T.tensor([[1, 1], [1, 1]])
        x2_val = T.tensor([[1, 1], [1, 1]])
        x3_val = T.tensor([[1, 1], [1, 1]])
        z_val, jacobian_x2_val = executor.run(feed_dict={
            x1: x1_val,
            x2: x2_val,
            x3: x3_val
        })

        I = T.identity(2)
        expected_jacobian_x2_val = T.einsum("ac,bd->abcd", I, I)

        assert isinstance(z, ad.Node)
        assert isinstance(jacobian_x2, ad.Node)
        assert T.array_equal(z_val, x1_val + x2_val + x3_val)
        assert T.array_equal(jacobian_x2_val, expected_jacobian_x2_val)


def test_add_jacobian_scalar_w_chain():

    for datatype in BACKEND_TYPES:
        T.set_backend(datatype)

        x1 = ad.Variable(name="x1", shape=[])
        x2 = ad.Variable(name="x2", shape=[])
        x3 = ad.Variable(name="x3", shape=[])
        y = x1 + x2
        z = y + x3

        jacobian_x2, = ad.jacobians(z, [x2])

        executor = ad.Executor([z, jacobian_x2])

        x1_val = T.tensor(1.)
        x2_val = T.tensor(1.)
        x3_val = T.tensor(1.)
        z_val, jacobian_x2_val = executor.run(feed_dict={
            x1: x1_val,
            x2: x2_val,
            x3: x3_val
        })

        expected_jacobian_x2_val = T.tensor([[1.]])

        assert isinstance(z, ad.Node)
        assert isinstance(jacobian_x2, ad.Node)
        assert T.array_equal(z_val, x1_val + x2_val + x3_val)
        assert T.array_equal(jacobian_x2_val, expected_jacobian_x2_val)


def test_sub_jacobian_w_chain():

    for datatype in BACKEND_TYPES:
        T.set_backend(datatype)

        x1 = ad.Variable(name="x1", shape=[2, 2])
        x2 = ad.Variable(name="x2", shape=[2, 2])
        x3 = ad.Variable(name="x3", shape=[2, 2])
        y = x1 - x2
        z = x3 - y

        jacobian_x2, = ad.jacobians(z, [x2])

        executor = ad.Executor([z, jacobian_x2])

        x1_val = T.tensor([[1, 1], [1, 1]])
        x2_val = T.tensor([[1, 1], [1, 1]])
        x3_val = T.tensor([[1, 1], [1, 1]])
        z_val, jacobian_x2_val = executor.run(feed_dict={
            x1: x1_val,
            x2: x2_val,
            x3: x3_val
        })

        I = T.identity(2)
        expected_jacobian_x2_val = T.einsum("ac,bd->abcd", I, I)

        assert isinstance(z, ad.Node)
        assert isinstance(jacobian_x2, ad.Node)
        assert T.array_equal(z_val, x3_val - x1_val + x2_val)
        assert T.array_equal(jacobian_x2_val, expected_jacobian_x2_val)
