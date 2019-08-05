import autodiff as ad
import numpy as np
import backend as T

BACKEND_TYPES = ['numpy', 'ctf']


def test_identity():

    for datatype in BACKEND_TYPES:
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

    for datatype in BACKEND_TYPES:
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


def test_sub_by_const():

    for datatype in BACKEND_TYPES:
        T.set_backend(datatype)

        x2 = ad.Variable(name="x2")
        y = x2 - 5

        grad_x2, = ad.gradients(y, [x2])

        executor = ad.Executor([y, grad_x2])
        x2_val = 2 * T.ones(3)
        y_val, grad_x2_val = executor.run(feed_dict={x2: x2_val})

        assert isinstance(y, ad.Node)
        assert T.array_equal(y_val, x2_val - 5)
        assert T.array_equal(grad_x2_val, T.ones_like(x2_val))


def test_sub_by_const_2():

    for datatype in BACKEND_TYPES:
        T.set_backend(datatype)

        x2 = ad.Variable(name="x2")
        y = 5 - x2

        grad_x2, = ad.gradients(y, [x2])

        executor = ad.Executor([y, grad_x2])
        x2_val = 2 * T.ones(3)
        y_val, grad_x2_val = executor.run(feed_dict={x2: x2_val})

        assert isinstance(y, ad.Node)
        assert T.array_equal(y_val, 5 - x2_val)
        assert T.array_equal(grad_x2_val, -T.ones_like(x2_val))


def test_negative():

    for datatype in BACKEND_TYPES:
        T.set_backend(datatype)

        x2 = ad.Variable(name="x2")
        y = -x2

        grad_x2, = ad.gradients(y, [x2])

        executor = ad.Executor([y, grad_x2])
        x2_val = 2 * T.ones(3)
        y_val, grad_x2_val = executor.run(feed_dict={x2: x2_val})

        assert isinstance(y, ad.Node)
        assert T.array_equal(y_val, -x2_val)
        assert T.array_equal(grad_x2_val, -T.ones_like(x2_val))


def test_mul_by_const():

    for datatype in BACKEND_TYPES:
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


def test_power():

    for datatype in BACKEND_TYPES:
        T.set_backend(datatype)

        x2 = ad.Variable(name="x2")
        y = x2**3

        grad_x2, = ad.gradients(y, [x2])

        executor = ad.Executor([y, grad_x2])
        x2_val = 2 * T.ones(3)
        y_val, grad_x2_val = executor.run(feed_dict={x2: x2_val})

        assert isinstance(y, ad.Node)
        assert T.array_equal(y_val, x2_val**3)
        assert T.array_equal(grad_x2_val, 3 * (x2_val**2))


def test_add_two_vars():

    for datatype in BACKEND_TYPES:
        T.set_backend(datatype)

        x2 = ad.Variable(name="x2")
        x3 = ad.Variable(name="x3")
        y = x2 + x3

        grad_x2, grad_x3 = ad.gradients(y, [x2, x3])

        executor = ad.Executor([y, grad_x2, grad_x3])
        x2_val = 2 * T.ones(3)
        x3_val = 3 * T.ones(3)
        y_val, grad_x2_val, grad_x3_val = executor.run(feed_dict={
            x2: x2_val,
            x3: x3_val
        })

        assert isinstance(y, ad.Node)
        assert T.array_equal(y_val, x2_val + x3_val)
        assert T.array_equal(grad_x2_val, T.ones_like(x2_val))
        assert T.array_equal(grad_x3_val, T.ones_like(x3_val))


def test_sub_two_vars():

    for datatype in BACKEND_TYPES:
        T.set_backend(datatype)

        x2 = ad.Variable(name="x2")
        x3 = ad.Variable(name="x3")
        y = x2 - x3

        grad_x2, grad_x3 = ad.gradients(y, [x2, x3])

        executor = ad.Executor([y, grad_x2, grad_x3])
        x2_val = 2 * T.ones(3)
        x3_val = 3 * T.ones(3)
        y_val, grad_x2_val, grad_x3_val = executor.run(feed_dict={
            x2: x2_val,
            x3: x3_val
        })

        assert isinstance(y, ad.Node)
        assert T.array_equal(y_val, x2_val - x3_val)
        assert T.array_equal(grad_x2_val, T.ones_like(x2_val))
        assert T.array_equal(grad_x3_val, -T.ones_like(x3_val))


def test_mul_two_vars():

    for datatype in BACKEND_TYPES:
        T.set_backend(datatype)

        x2 = ad.Variable(name="x2")
        x3 = ad.Variable(name="x3")
        y = x2 * x3

        grad_x2, grad_x3 = ad.gradients(y, [x2, x3])

        executor = ad.Executor([y, grad_x2, grad_x3])
        x2_val = 2 * T.ones(3)
        x3_val = 3 * T.ones(3)
        y_val, grad_x2_val, grad_x3_val = executor.run(feed_dict={
            x2: x2_val,
            x3: x3_val
        })

        assert isinstance(y, ad.Node)
        assert T.array_equal(y_val, x2_val * x3_val)
        assert T.array_equal(grad_x2_val, x3_val)
        assert T.array_equal(grad_x3_val, x2_val)


def test_add_mul_mix_1():

    for datatype in BACKEND_TYPES:
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
        y_val, grad_x1_val, grad_x2_val, grad_x3_val = executor.run(feed_dict={
            x1: x1_val,
            x2: x2_val,
            x3: x3_val
        })

        assert isinstance(y, ad.Node)
        assert T.array_equal(y_val, x1_val + x2_val * x3_val)
        assert T.array_equal(grad_x1_val,
                             T.ones_like(x1_val) + x2_val * x3_val)
        assert T.array_equal(grad_x2_val, x3_val * x1_val)
        assert T.array_equal(grad_x3_val, x2_val * x1_val)


def test_add_mul_mix_2():

    for datatype in BACKEND_TYPES:
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
            feed_dict={
                x1: x1_val,
                x2: x2_val,
                x3: x3_val,
                x4: x4_val
            })
        assert isinstance(y, ad.Node)
        assert T.array_equal(y_val, x1_val + x2_val * x3_val * x4_val)
        assert T.array_equal(grad_x1_val, T.ones_like(x1_val))
        assert T.array_equal(grad_x2_val, x3_val * x4_val)
        assert T.array_equal(grad_x3_val, x2_val * x4_val)
        assert T.array_equal(grad_x4_val, x2_val * x3_val)


def test_add_mul_mix_3():

    for datatype in BACKEND_TYPES:
        T.set_backend(datatype)

        x2 = ad.Variable(name="x2")
        x3 = ad.Variable(name="x3")
        z = x2 * x2 + x2 + x3 + 3
        y = z * z + x3

        grad_x2, grad_x3 = ad.gradients(y, [x2, x3])

        executor = ad.Executor([y, grad_x2, grad_x3])
        x2_val = 2 * T.ones(3)
        x3_val = 3 * T.ones(3)
        y_val, grad_x2_val, grad_x3_val = executor.run(feed_dict={
            x2: x2_val,
            x3: x3_val
        })

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

    for datatype in BACKEND_TYPES:
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
            feed_dict={
                x2: x2_val,
                x3: x3_val
            })

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

    for datatype in BACKEND_TYPES:
        T.set_backend(datatype)

        x2 = ad.Variable(name="x2")
        x3 = ad.Variable(name="x3")
        y = x2 @ x3

        grad_x2, grad_x3 = ad.gradients(y, [x2, x3])

        executor = ad.Executor([y, grad_x2, grad_x3])
        x2_val = T.tensor([[1, 2], [3, 4], [5, 6]])  # 3x2
        x3_val = T.tensor([[7, 8, 9], [10, 11, 12]])  # 2x3

        y_val, grad_x2_val, grad_x3_val = executor.run(feed_dict={
            x2: x2_val,
            x3: x3_val
        })

        expected_yval = T.dot(x2_val, x3_val)
        expected_grad_x2_val = T.dot(T.ones_like(expected_yval),
                                     T.transpose(x3_val))
        expected_grad_x3_val = T.dot(T.transpose(x2_val),
                                     T.ones_like(expected_yval))

        assert isinstance(y, ad.Node)
        assert T.array_equal(y_val, expected_yval)
        assert T.array_equal(grad_x2_val, expected_grad_x2_val)
        assert T.array_equal(grad_x3_val, expected_grad_x3_val)


def test_einsum():

    for datatype in BACKEND_TYPES:
        T.set_backend(datatype)

        x2 = ad.Variable(name="x2")
        x3 = ad.Variable(name="x3")
        y = ad.einsum('ik,kj->ij', x2, x3)

        grad_x2, grad_x3 = ad.gradients(y, [x2, x3])

        executor = ad.Executor([y, grad_x2, grad_x3])
        x2_val = T.tensor([[1, 2], [3, 4], [5, 6]])  # 3x2
        x3_val = T.tensor([[7, 8, 9], [10, 11, 12]])  # 2x3

        y_val, grad_x2_val, grad_x3_val = executor.run(feed_dict={
            x2: x2_val,
            x3: x3_val
        })

        expected_yval = T.dot(x2_val, x3_val)
        expected_grad_x2_val = T.dot(T.ones_like(expected_yval),
                                     T.transpose(x3_val))
        expected_grad_x3_val = T.dot(T.transpose(x2_val),
                                     T.ones_like(expected_yval))

        assert isinstance(y, ad.Node)
        assert T.array_equal(y_val, expected_yval)
        assert T.array_equal(grad_x2_val, expected_grad_x2_val)
        assert T.array_equal(grad_x3_val, expected_grad_x3_val)


def test_norm():

    for datatype in BACKEND_TYPES:
        T.set_backend(datatype)

        x = ad.Variable(name="x")
        y = ad.norm(x)
        z = y**2

        grad_x, = ad.gradients(z, [x])

        executor = ad.Executor([z, grad_x])
        x_val = T.tensor([[1, 2], [3, 4], [5, 6]])  # 3x2

        z_val, grad_x_val = executor.run(feed_dict={x: x_val})

        expected_zval = T.norm(x_val)**2
        expected_grad_x_val = 2 * x_val

        assert isinstance(z, ad.Node)
        assert T.array_equal(z_val, expected_zval)
        assert T.array_equal(grad_x_val, expected_grad_x_val)


def test_sum():

    for datatype in BACKEND_TYPES:
        T.set_backend(datatype)

        x = ad.Variable(name="x")
        y = ad.sum(x)

        grad_x, = ad.gradients(y, [x])

        executor = ad.Executor([y, grad_x])
        x_val = T.tensor([[1, 2], [3, 4], [5, 6]])  # 3x2

        y_val, grad_x_val = executor.run(feed_dict={x: x_val})

        expected_yval = T.sum(x_val)
        expected_grad_x_val = T.ones_like(x_val)

        assert isinstance(y, ad.Node)
        assert T.array_equal(y_val, expected_yval)
        assert T.array_equal(grad_x_val, expected_grad_x_val)


def test_transpose():

    for datatype in BACKEND_TYPES:
        T.set_backend(datatype)

        x = ad.Variable(name="x")
        y = ad.transpose(x)

        grad_x, = ad.gradients(y, [x])

        executor = ad.Executor([y, grad_x])
        x_val = T.tensor([[1, 2], [3, 4], [5, 6]])  # 3x2

        y_val, grad_x_val = executor.run(feed_dict={x: x_val})

        expected_yval = T.transpose(x_val)
        expected_grad_x_val = T.ones_like(x_val)

        assert isinstance(y, ad.Node)
        assert T.array_equal(y_val, expected_yval)
        assert T.array_equal(grad_x_val, expected_grad_x_val)


def test_inner_product():
    for datatype in BACKEND_TYPES:
        T.set_backend(datatype)
        x = ad.Variable(name="x")
        x_inner = x @ ad.transpose(x)

        grad_x, = ad.gradients(x_inner, [x])

        executor = ad.Executor([x_inner, grad_x])
        x_val = T.tensor([[1., 2., 3.]])  # 1x3

        y_val, grad_x_val = executor.run(feed_dict={x: x_val})

        expected_yval = T.norm(x_val) ** 2
        expected_grad_x_val = 2 * x_val

        assert isinstance(x_inner, ad.Node)
        assert T.array_equal(y_val[0][0], expected_yval)
        assert T.array_equal(grad_x_val, expected_grad_x_val)


def test_inner_product_hvp():
    for datatype in BACKEND_TYPES:
        T.set_backend(datatype)
        x = ad.Variable(name="x")
        H = ad.Variable(name="H")
        v = ad.Variable(name="v")
        y = ad.transpose(x) @ x

        grad_x, = ad.gradients(y, [x])
        Hv, = ad.hvp(output_node=y, node_list=[x], vector_list=[v])

        executor = ad.Executor([y, grad_x, Hv])
        x_val = T.tensor([[1.], [2.], [3]])  # 2x1
        v_val = T.tensor([[1.], [2.], [3]])  # 2x1
        y_val, grad_x_val, Hv_val = executor.run(feed_dict={
            x: x_val,
            v: v_val
        })

        expected_yval = T.transpose(x_val) @ x_val
        expected_grad_x_val = 2 * x_val
        expected_hv_val = 2 * v_val

        print(expected_hv_val)
        print(Hv_val)

        assert isinstance(y, ad.Node)
        assert T.array_equal(y_val, expected_yval)
        assert T.array_equal(grad_x_val, expected_grad_x_val)
        assert T.array_equal(Hv_val, expected_hv_val)


def test_hvp1():
    for datatype in BACKEND_TYPES:
        T.set_backend(datatype)
        x = ad.Variable(name="x")
        H = ad.Variable(name="H")
        v = ad.Variable(name="v")
        y = ad.sum(x * (H @ x))

        grad_x, = ad.gradients(y, [x])
        Hv, = ad.hvp(output_node=y, node_list=[x], vector_list=[v])

        executor = ad.Executor([y, grad_x, Hv])
        x_val = T.tensor([[1.], [2.], [3]])  # 2x1
        v_val = T.tensor([[1.], [2.], [3]])  # 2x1
        H_val = T.tensor([[2., 0., 0.], [0., 2., 0.], [0., 0., 2.]])  # 2x2
        y_val, grad_x_val, Hv_val = executor.run(feed_dict={
            x: x_val,
            H: H_val,
            v: v_val
        })

        expected_yval = T.transpose(x_val) @ H_val @ x_val
        expected_grad_x_val = 2 * H_val @ x_val
        expected_hv_val = T.tensor([[4.], [8.], [12.]])

        assert isinstance(y, ad.Node)
        assert T.array_equal(y_val, expected_yval[0][0])
        assert T.array_equal(grad_x_val, expected_grad_x_val)
        assert T.array_equal(Hv_val, expected_hv_val)


def test_hvp2():
    for datatype in BACKEND_TYPES:
        T.set_backend(datatype)
        x = ad.Variable(name="x")
        H = ad.Variable(name="H")
        v = ad.Variable(name="v")
        y = ad.transpose(x) @ H @ x

        grad_x, = ad.gradients(y, [x])
        Hv, = ad.hvp(output_node=y, node_list=[x], vector_list=[v])

        executor = ad.Executor([y, grad_x, Hv])
        x_val = T.tensor([[1.], [2.], [3]])  # 2x1
        v_val = T.tensor([[1.], [2.], [3]])  # 2x1
        H_val = T.tensor([[2., 0., 0.], [0., 2., 0.], [0., 0., 2.]])  # 2x2
        y_val, grad_x_val, Hv_val = executor.run(feed_dict={
            x: x_val,
            H: H_val,
            v: v_val
        })
        expected_yval = T.transpose(x_val) @ H_val @ x_val
        expected_grad_x_val = 2 * H_val @ x_val
        expected_hv_val = T.tensor([[4.], [8.], [12.]])

        print(expected_hv_val)
        print(Hv_val)

        assert isinstance(y, ad.Node)
        assert T.array_equal(y_val, expected_yval)
        assert T.array_equal(grad_x_val, expected_grad_x_val)
        assert T.array_equal(Hv_val, expected_hv_val)

