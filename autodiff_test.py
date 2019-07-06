import autodiff as ad
import numpy as np
import backend as T


def test_identity():

    for datatype in ['numpy', 'ctf']:
        T.set_backend(datatype)

        x2 = ad.Variable(name="x2")
        y = x2

        grad_x2, = ad.gradients(y, [x2])

        executor = ad.Executor([y, grad_x2])

        x2_val = 2 * T.ones(3)
        y_val, grad_x2_val = executor.run(feed_dict={x2: x2_val})

        assert isinstance(y, ad.Node)
        assert T.array_equal(y_val, x2_val)
        assert T.array_equal(grad_x2_val, T.ones_like(x2_val))


def test_add_by_const():

    for datatype in ['numpy', 'ctf']:
        T.set_backend(datatype)

        x2 = ad.Variable(name="x2")
        y = 5 + x2

        grad_x2, = ad.gradients(y, [x2])

        executor = ad.Executor([y, grad_x2])
        x2_val = 2 * T.ones(3)
        y_val, grad_x2_val = executor.run(feed_dict={x2: x2_val})

        assert isinstance(y, ad.Node)
        assert T.array_equal(y_val, x2_val + 5)
        assert T.array_equal(grad_x2_val, T.ones_like(x2_val))


def test_mul_by_const():

    for datatype in ['numpy', 'ctf']:
        T.set_backend(datatype)

        x2 = ad.Variable(name="x2")
        y = 5 * x2

        grad_x2, = ad.gradients(y, [x2])

        executor = ad.Executor([y, grad_x2])
        x2_val = 2 * T.ones(3)
        y_val, grad_x2_val = executor.run(feed_dict={x2: x2_val})

        assert isinstance(y, ad.Node)
        assert T.array_equal(y_val, x2_val * 5)
        assert T.array_equal(grad_x2_val, T.ones_like(x2_val) * 5)


def test_add_two_vars():

    for datatype in ['numpy', 'ctf']:
        T.set_backend(datatype)

        x2 = ad.Variable(name="x2")
        x3 = ad.Variable(name="x3")
        y = x2 + x3

        grad_x2, grad_x3 = ad.gradients(y, [x2, x3])

        executor = ad.Executor([y, grad_x2, grad_x3])
        x2_val = 2 * T.ones(3)
        x3_val = 3 * T.ones(3)
        y_val, grad_x2_val, grad_x3_val = executor.run(
            feed_dict={x2: x2_val, x3: x3_val})

        assert isinstance(y, ad.Node)
        assert T.array_equal(y_val, x2_val + x3_val)
        assert T.array_equal(grad_x2_val, T.ones_like(x2_val))
        assert T.array_equal(grad_x3_val, T.ones_like(x3_val))


def test_mul_two_vars():

    for datatype in ['numpy', 'ctf']:
        T.set_backend(datatype)

        x2 = ad.Variable(name="x2")
        x3 = ad.Variable(name="x3")
        y = x2 * x3

        grad_x2, grad_x3 = ad.gradients(y, [x2, x3])

        executor = ad.Executor([y, grad_x2, grad_x3])
        x2_val = 2 * T.ones(3)
        x3_val = 3 * T.ones(3)
        y_val, grad_x2_val, grad_x3_val = executor.run(
            feed_dict={x2: x2_val, x3: x3_val})

        assert isinstance(y, ad.Node)
        assert T.array_equal(y_val, x2_val * x3_val)
        assert T.array_equal(grad_x2_val, x3_val)
        assert T.array_equal(grad_x3_val, x2_val)


def test_add_mul_mix_1():

    for datatype in ['numpy', 'ctf']:
        T.set_backend(datatype)

        x1 = ad.Variable(name="x1")
        x2 = ad.Variable(name="x2")
        x3 = ad.Variable(name="x3")
        y = x1 + x2 * x3 * x1

        grad_x1, grad_x2, grad_x3 = ad.gradients(y, [x1, x2, x3])

        executor = ad.Executor([y, grad_x1, grad_x2, grad_x3])
        x1_val = 1 * T.ones(3)
        x2_val = 2 * T.ones(3)
        x3_val = 3 * T.ones(3)
        y_val, grad_x1_val, grad_x2_val, grad_x3_val = executor.run(
            feed_dict={x1: x1_val, x2: x2_val, x3: x3_val})

        print(grad_x1_val, T.ones_like(x1_val) + x2_val * x3_val)

        assert isinstance(y, ad.Node)
        assert T.array_equal(y_val, x1_val + x2_val * x3_val)
        assert T.array_equal(
            grad_x1_val,
            T.ones_like(x1_val) +
            x2_val *
            x3_val)
        assert T.array_equal(grad_x2_val, x3_val * x1_val)
        assert T.array_equal(grad_x3_val, x2_val * x1_val)


def test_add_mul_mix_2():

    for datatype in ['numpy', 'ctf']:
        T.set_backend(datatype)

        x1 = ad.Variable(name="x1")
        x2 = ad.Variable(name="x2")
        x3 = ad.Variable(name="x3")
        x4 = ad.Variable(name="x4")
        y = x1 + x2 * x3 * x4

        grad_x1, grad_x2, grad_x3, grad_x4 = ad.gradients(y, [x1, x2, x3, x4])

        executor = ad.Executor([y, grad_x1, grad_x2, grad_x3, grad_x4])
        x1_val = 1 * T.ones(3)
        x2_val = 2 * T.ones(3)
        x3_val = 3 * T.ones(3)
        x4_val = 4 * T.ones(3)
        y_val, grad_x1_val, grad_x2_val, grad_x3_val, grad_x4_val = executor.run(
            feed_dict={x1: x1_val, x2: x2_val, x3: x3_val, x4: x4_val})

        assert isinstance(y, ad.Node)
        assert T.array_equal(y_val, x1_val + x2_val * x3_val * x4_val)
        assert T.array_equal(grad_x1_val, T.ones_like(x1_val))
        assert T.array_equal(grad_x2_val, x3_val * x4_val)
        assert T.array_equal(grad_x3_val, x2_val * x4_val)
        assert T.array_equal(grad_x4_val, x2_val * x3_val)


def test_add_mul_mix_3():

    for datatype in ['numpy', 'ctf']:
        T.set_backend(datatype)

        x2 = ad.Variable(name="x2")
        x3 = ad.Variable(name="x3")
        z = x2 * x2 + x2 + x3 + 3
        y = z * z + x3

        grad_x2, grad_x3 = ad.gradients(y, [x2, x3])

        executor = ad.Executor([y, grad_x2, grad_x3])
        x2_val = 2 * T.ones(3)
        x3_val = 3 * T.ones(3)
        y_val, grad_x2_val, grad_x3_val = executor.run(
            feed_dict={x2: x2_val, x3: x3_val})

        z_val = x2_val * x2_val + x2_val + x3_val + 3
        expected_yval = z_val * z_val + x3_val
        expected_grad_x2_val = 2 * \
            (x2_val * x2_val + x2_val + x3_val + 3) * (2 * x2_val + 1)
        expected_grad_x3_val = 2 * (x2_val * x2_val + x2_val + x3_val + 3) + 1
        assert isinstance(y, ad.Node)
        assert T.array_equal(y_val, expected_yval)
        assert T.array_equal(grad_x2_val, expected_grad_x2_val)
        assert T.array_equal(grad_x3_val, expected_grad_x3_val)


def test_grad_of_grad():

    for datatype in ['numpy', 'ctf']:
        T.set_backend(datatype)

        x2 = ad.Variable(name="x2")
        x3 = ad.Variable(name="x3")
        y = x2 * x2 + x2 * x3

        grad_x2, grad_x3 = ad.gradients(y, [x2, x3])
        grad_x2_x2, grad_x2_x3 = ad.gradients(grad_x2, [x2, x3])

        executor = ad.Executor([y, grad_x2, grad_x3, grad_x2_x2, grad_x2_x3])
        x2_val = 2 * T.ones(3)
        x3_val = 3 * T.ones(3)
        y_val, grad_x2_val, grad_x3_val, grad_x2_x2_val, grad_x2_x3_val = executor.run(
            feed_dict={x2: x2_val, x3: x3_val})

        expected_yval = x2_val * x2_val + x2_val * x3_val
        expected_grad_x2_val = 2 * x2_val + x3_val
        expected_grad_x3_val = x2_val
        expected_grad_x2_x2_val = 2 * T.ones_like(x2_val)
        expected_grad_x2_x3_val = 1 * T.ones_like(x2_val)

        assert isinstance(y, ad.Node)
        assert T.array_equal(y_val, expected_yval)
        assert T.array_equal(grad_x2_val, expected_grad_x2_val)
        assert T.array_equal(grad_x3_val, expected_grad_x3_val)
        assert T.array_equal(grad_x2_x2_val, expected_grad_x2_x2_val)
        assert T.array_equal(grad_x2_x3_val, expected_grad_x2_x3_val)


def test_matmul_two_vars():

    for datatype in ['numpy', 'ctf']:
        T.set_backend(datatype)

        x2 = ad.Variable(name="x2")
        x3 = ad.Variable(name="x3")
        y = ad.matmul_op(x2, x3)

        grad_x2, grad_x3 = ad.gradients(y, [x2, x3])

        executor = ad.Executor([y, grad_x2, grad_x3])
        x2_val = T.tensor([[1, 2], [3, 4], [5, 6]])  # 3x2
        x3_val = T.tensor([[7, 8, 9], [10, 11, 12]])  # 2x3

        y_val, grad_x2_val, grad_x3_val = executor.run(
            feed_dict={x2: x2_val, x3: x3_val})

        expected_yval = T.dot(x2_val, x3_val)
        expected_grad_x2_val = T.dot(
            T.ones_like(expected_yval),
            T.transpose(x3_val))
        expected_grad_x3_val = T.dot(
            T.transpose(x2_val),
            T.ones_like(expected_yval))

        assert isinstance(y, ad.Node)
        assert T.array_equal(y_val, expected_yval)
        assert T.array_equal(grad_x2_val, expected_grad_x2_val)
        assert T.array_equal(grad_x3_val, expected_grad_x3_val)
