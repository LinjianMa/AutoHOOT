import autodiff as ad
import backend as T

BACKEND_TYPES = ['numpy', 'ctf']


def test_identity():

    for datatype in BACKEND_TYPES:
        T.set_backend(datatype)

        x2 = ad.Variable(name="x2", shape=[3])
        y = ad.sum(x2)

        grad_x2, = ad.gradients(y, [x2])

        executor = ad.Executor([y, grad_x2])

        x2_val = 2 * T.ones(3)
        y_val, grad_x2_val = executor.run(feed_dict={x2: x2_val})

        assert isinstance(y, ad.Node)
        assert T.array_equal(y_val, T.sum(x2_val))
        assert T.array_equal(grad_x2_val, T.ones_like(x2_val))


def test_add_by_const():

    for datatype in BACKEND_TYPES:
        T.set_backend(datatype)

        x2 = ad.Variable(name="x2", shape=[3])
        y = ad.sum(5 + x2)

        grad_x2, = ad.gradients(y, [x2])

        executor = ad.Executor([y, grad_x2])
        x2_val = 2 * T.ones(3)
        y_val, grad_x2_val = executor.run(feed_dict={x2: x2_val})

        assert isinstance(y, ad.Node)
        assert T.array_equal(y_val, T.sum(x2_val + 5))
        assert T.array_equal(grad_x2_val, T.ones_like(x2_val))


def test_sub_by_const():

    for datatype in BACKEND_TYPES:
        T.set_backend(datatype)

        x2 = ad.Variable(name="x2", shape=[3])
        y = ad.sum(x2 - 5)

        grad_x2, = ad.gradients(y, [x2])

        executor = ad.Executor([y, grad_x2])
        x2_val = 2 * T.ones(3)
        y_val, grad_x2_val = executor.run(feed_dict={x2: x2_val})

        assert isinstance(y, ad.Node)
        assert T.array_equal(y_val, T.sum(x2_val - 5))
        assert T.array_equal(grad_x2_val, T.ones_like(x2_val))


def test_sub_by_const_2():

    for datatype in BACKEND_TYPES:
        T.set_backend(datatype)

        x2 = ad.Variable(name="x2", shape=[3])
        y = ad.sum(5 - x2)

        grad_x2, = ad.gradients(y, [x2])

        executor = ad.Executor([y, grad_x2])
        x2_val = 2 * T.ones(3)
        y_val, grad_x2_val = executor.run(feed_dict={x2: x2_val})

        assert isinstance(y, ad.Node)
        assert T.array_equal(y_val, T.sum(5 - x2_val))
        assert T.array_equal(grad_x2_val, -T.ones_like(x2_val))


def test_negative():

    for datatype in BACKEND_TYPES:
        T.set_backend(datatype)

        x2 = ad.Variable(name="x2", shape=[3])
        y = ad.sum(-x2)

        grad_x2, = ad.gradients(y, [x2])

        executor = ad.Executor([y, grad_x2])
        x2_val = 2 * T.ones(3)
        y_val, grad_x2_val = executor.run(feed_dict={x2: x2_val})

        assert isinstance(y, ad.Node)
        assert T.array_equal(y_val, T.sum(-x2_val))
        assert T.array_equal(grad_x2_val, -T.ones_like(x2_val))


def test_mul_by_const():

    for datatype in BACKEND_TYPES:
        T.set_backend(datatype)

        x2 = ad.Variable(name="x2", shape=[3])
        y = ad.sum(5 * x2)

        grad_x2, = ad.gradients(y, [x2])

        executor = ad.Executor([y, grad_x2])
        x2_val = 2 * T.ones(3)
        y_val, grad_x2_val = executor.run(feed_dict={x2: x2_val})

        assert isinstance(y, ad.Node)
        assert T.array_equal(y_val, T.sum(x2_val * 5))
        assert T.array_equal(grad_x2_val, T.ones_like(x2_val) * 5)


def test_power():

    for datatype in BACKEND_TYPES:
        T.set_backend(datatype)

        x2 = ad.Variable(name="x2", shape=[3])
        y = ad.sum(x2**3)

        grad_x2, = ad.gradients(y, [x2])

        executor = ad.Executor([y, grad_x2])
        x2_val = 2 * T.ones(3)
        y_val, grad_x2_val = executor.run(feed_dict={x2: x2_val})

        assert isinstance(y, ad.Node)
        assert T.array_equal(y_val, T.sum(x2_val**3))
        assert T.array_equal(grad_x2_val, 3 * (x2_val**2))


def test_add_two_vars():

    for datatype in BACKEND_TYPES:
        T.set_backend(datatype)

        x2 = ad.Variable(name="x2", shape=[3])
        x3 = ad.Variable(name="x3", shape=[3])
        y = ad.sum(x2 + x3)

        grad_x2, grad_x3 = ad.gradients(y, [x2, x3])

        executor = ad.Executor([y, grad_x2, grad_x3])
        x2_val = 2 * T.ones(3)
        x3_val = 3 * T.ones(3)
        y_val, grad_x2_val, grad_x3_val = executor.run(feed_dict={
            x2: x2_val,
            x3: x3_val
        })

        assert isinstance(y, ad.Node)
        assert T.array_equal(y_val, T.sum(x2_val + x3_val))
        assert T.array_equal(grad_x2_val, T.ones_like(x2_val))
        assert T.array_equal(grad_x3_val, T.ones_like(x3_val))


def test_sub_two_vars():

    for datatype in BACKEND_TYPES:
        T.set_backend(datatype)

        x2 = ad.Variable(name="x2", shape=[3])
        x3 = ad.Variable(name="x3", shape=[3])
        y = ad.sum(x2 - x3)

        grad_x2, grad_x3 = ad.gradients(y, [x2, x3])

        executor = ad.Executor([y, grad_x2, grad_x3])
        x2_val = 2 * T.ones(3)
        x3_val = 3 * T.ones(3)
        y_val, grad_x2_val, grad_x3_val = executor.run(feed_dict={
            x2: x2_val,
            x3: x3_val
        })

        assert isinstance(y, ad.Node)
        assert T.array_equal(y_val, T.sum(x2_val - x3_val))
        assert T.array_equal(grad_x2_val, T.ones_like(x2_val))
        assert T.array_equal(grad_x3_val, -T.ones_like(x3_val))


def test_mul_two_vars():

    for datatype in BACKEND_TYPES:
        T.set_backend(datatype)

        x2 = ad.Variable(name="x2", shape=[3])
        x3 = ad.Variable(name="x3", shape=[3])
        y = ad.sum(x2 * x3)

        grad_x2, grad_x3 = ad.gradients(y, [x2, x3])

        executor = ad.Executor([y, grad_x2, grad_x3])
        x2_val = 2 * T.ones(3)
        x3_val = 3 * T.ones(3)
        y_val, grad_x2_val, grad_x3_val = executor.run(feed_dict={
            x2: x2_val,
            x3: x3_val
        })

        assert isinstance(y, ad.Node)
        assert T.array_equal(y_val, T.sum(x2_val * x3_val))
        assert T.array_equal(grad_x2_val, x3_val)
        assert T.array_equal(grad_x3_val, x2_val)


def test_add_mul_mix_1():

    for datatype in BACKEND_TYPES:
        T.set_backend(datatype)

        x1 = ad.Variable(name="x1", shape=[3])
        x2 = ad.Variable(name="x2", shape=[3])
        x3 = ad.Variable(name="x3", shape=[3])
        y = ad.sum(x1 + x2 * x3 * x1)

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
        assert T.array_equal(y_val, T.sum(x1_val + x2_val * x3_val))
        assert T.array_equal(grad_x1_val,
                             T.ones_like(x1_val) + x2_val * x3_val)
        assert T.array_equal(grad_x2_val, x3_val * x1_val)
        assert T.array_equal(grad_x3_val, x2_val * x1_val)


def test_add_mul_mix_2():

    for datatype in BACKEND_TYPES:
        T.set_backend(datatype)

        x1 = ad.Variable(name="x1", shape=[3])
        x2 = ad.Variable(name="x2", shape=[3])
        x3 = ad.Variable(name="x3", shape=[3])
        x4 = ad.Variable(name="x4", shape=[3])
        y = ad.sum(x1 + x2 * x3 * x4)

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
        assert T.array_equal(y_val, T.sum(x1_val + x2_val * x3_val * x4_val))
        assert T.array_equal(grad_x1_val, T.ones_like(x1_val))
        assert T.array_equal(grad_x2_val, x3_val * x4_val)
        assert T.array_equal(grad_x3_val, x2_val * x4_val)
        assert T.array_equal(grad_x4_val, x2_val * x3_val)


def test_add_mul_mix_3():

    for datatype in BACKEND_TYPES:
        T.set_backend(datatype)

        x2 = ad.Variable(name="x2", shape=[3])
        x3 = ad.Variable(name="x3", shape=[3])
        z = x2 * x2 + x2 + x3 + 3
        y = ad.sum(z * z + x3)

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
        assert T.array_equal(y_val, T.sum(expected_yval))
        assert T.array_equal(grad_x2_val, expected_grad_x2_val)
        assert T.array_equal(grad_x3_val, expected_grad_x3_val)


def test_matmul_two_vars():

    for datatype in BACKEND_TYPES:
        T.set_backend(datatype)

        x2 = ad.Variable(name="x2", shape=[3, 2])
        x3 = ad.Variable(name="x3", shape=[2, 3])
        y = ad.sum(x2 @ x3)

        grad_x2, grad_x3 = ad.gradients(y, [x2, x3])

        executor = ad.Executor([y, grad_x2, grad_x3])
        x2_val = T.tensor([[1, 2], [3, 4], [5, 6]])  # 3x2
        x3_val = T.tensor([[7, 8, 9], [10, 11, 12]])  # 2x3

        y_val, grad_x2_val, grad_x3_val = executor.run(feed_dict={
            x2: x2_val,
            x3: x3_val
        })

        expected_grad_sum = T.ones_like(T.dot(x2_val, x3_val))
        expected_yval = T.sum(T.dot(x2_val, x3_val))
        expected_grad_x2_val = T.dot(expected_grad_sum, T.transpose(x3_val))
        expected_grad_x3_val = T.dot(T.transpose(x2_val), expected_grad_sum)

        assert isinstance(y, ad.Node)
        assert T.array_equal(y_val, expected_yval)
        assert T.array_equal(grad_x2_val, expected_grad_x2_val)
        assert T.array_equal(grad_x3_val, expected_grad_x3_val)


def test_einsum():

    for datatype in BACKEND_TYPES:
        T.set_backend(datatype)

        x2 = ad.Variable(name="x2", shape=[3, 2])
        x3 = ad.Variable(name="x3", shape=[2, 3])
        matmul = ad.einsum('ik,kj->ij', x2, x3)
        y = ad.sum(matmul)

        grad_x2, grad_x3 = ad.gradients(y, [x2, x3])

        executor = ad.Executor([y, grad_x2, grad_x3])
        x2_val = T.tensor([[1, 2], [3, 4], [5, 6]])  # 3x2
        x3_val = T.tensor([[7, 8, 9], [10, 11, 12]])  # 2x3

        y_val, grad_x2_val, grad_x3_val = executor.run(feed_dict={
            x2: x2_val,
            x3: x3_val
        })

        expected_grad_sum = T.ones_like(T.dot(x2_val, x3_val))
        expected_yval = T.sum(T.dot(x2_val, x3_val))
        expected_grad_x2_val = T.dot(expected_grad_sum, T.transpose(x3_val))
        expected_grad_x3_val = T.dot(T.transpose(x2_val), expected_grad_sum)

        assert isinstance(y, ad.Node)
        assert T.array_equal(y_val, expected_yval)
        assert T.array_equal(grad_x2_val, expected_grad_x2_val)
        assert T.array_equal(grad_x3_val, expected_grad_x3_val)


def test_einsum_3op():

    for datatype in BACKEND_TYPES:
        T.set_backend(datatype)

        x2 = ad.Variable(name="x2", shape=[3, 2])
        x3 = ad.Variable(name="x3", shape=[2, 3])
        x4 = ad.Variable(name="x3", shape=[3, 2])
        matmul = ad.einsum('ik,kj,jl->il', x2, x3, x4)
        y = ad.sum(matmul)

        grad_x2, grad_x3, grad_x4 = ad.gradients(y, [x2, x3, x4])

        executor = ad.Executor([y, grad_x2, grad_x3, grad_x4])
        x2_val = T.tensor([[1, 2], [3, 4], [5, 6]])  # 3x2
        x3_val = T.tensor([[7, 8, 9], [10, 11, 12]])  # 2x3
        x4_val = T.tensor([[1, 2], [3, 4], [5, 6]])  # 3x2

        y_val, grad_x2_val, grad_x3_val, grad_x4_val = executor.run(feed_dict={
            x2: x2_val,
            x3: x3_val,
            x4: x4_val
        })

        expected_grad_sum = T.ones_like(T.dot(T.dot(x2_val, x3_val), x4_val))
        expected_yval = T.sum(T.dot(T.dot(x2_val, x3_val), x4_val))
        expected_grad_x2_val = T.einsum("il, kj, jl->ik", expected_grad_sum,
                                        x3_val, x4_val)
        expected_grad_x3_val = T.einsum("ik, il, jl->kj", x2_val,
                                        expected_grad_sum, x4_val)
        expected_grad_x4_val = T.einsum("ik, kj, il->jl", x2_val, x3_val,
                                        expected_grad_sum)

        assert isinstance(y, ad.Node)
        assert T.array_equal(y_val, expected_yval)
        assert T.array_equal(grad_x2_val, expected_grad_x2_val)
        assert T.array_equal(grad_x3_val, expected_grad_x3_val)
        assert T.array_equal(grad_x4_val, expected_grad_x4_val)


def test_norm():

    for datatype in BACKEND_TYPES:
        T.set_backend(datatype)

        x = ad.Variable(name="x", shape=[3, 2])
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

        x = ad.Variable(name="x", shape=[3, 2])
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

        x = ad.Variable(name="x", shape=[3, 2])
        y = ad.sum(ad.transpose(x))

        grad_x, = ad.gradients(y, [x])

        executor = ad.Executor([y, grad_x])
        x_val = T.tensor([[1, 2], [3, 4], [5, 6]])  # 3x2

        y_val, grad_x_val = executor.run(feed_dict={x: x_val})

        expected_yval = T.sum(T.transpose(x_val))
        expected_grad_x_val = T.ones_like(x_val)

        assert isinstance(y, ad.Node)
        assert T.array_equal(y_val, expected_yval)
        assert T.array_equal(grad_x_val, expected_grad_x_val)


def test_transpose_einsum():

    for datatype in BACKEND_TYPES:
        T.set_backend(datatype)

        x = ad.Variable(name="x", shape=[3, 2])
        y = ad.sum(ad.einsum("ij->ji", x))

        grad_x, = ad.gradients(y, [x])

        executor = ad.Executor([y, grad_x])
        x_val = T.tensor([[1, 2], [3, 4], [5, 6]])  # 3x2

        y_val, grad_x_val = executor.run(feed_dict={x: x_val})

        expected_yval = T.sum(T.transpose(x_val))
        expected_grad_x_val = T.ones_like(x_val)

        assert isinstance(y, ad.Node)
        assert T.array_equal(y_val, expected_yval)
        assert T.array_equal(grad_x_val, expected_grad_x_val)


def test_tensor_transpose_einsum():

    for datatype in BACKEND_TYPES:
        T.set_backend(datatype)

        x = ad.Variable(name="x", shape=[2, 2, 2])
        y = ad.einsum("kij->jik", x)

        v = ad.Variable(name="v", shape=[2, 2, 2])
        v_val = T.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])  # 2 x 2 x 2
        grad_x, = ad.transposed_vjps(y, [x], v)

        executor = ad.Executor([y, grad_x])
        x_val = T.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])  # 2 x 2 x 2

        y_val, grad_x_val = executor.run(feed_dict={x: x_val, v: v_val})

        expected_yval = T.einsum("kij->jik", x_val)
        expected_grad_x_val = T.einsum("kij->jik", v_val)

        assert isinstance(y, ad.Node)
        assert T.array_equal(y_val, expected_yval)
        assert T.array_equal(grad_x_val, expected_grad_x_val)


def test_inner_product():
    for datatype in BACKEND_TYPES:
        T.set_backend(datatype)
        x = ad.Variable(name="x", shape=[1, 3])
        x_inner = ad.sum(x @ ad.transpose(x))

        grad_x, = ad.gradients(x_inner, [x])

        executor = ad.Executor([x_inner, grad_x])
        x_val = T.tensor([[1., 2., 3.]])  # 1x3

        y_val, grad_x_val = executor.run(feed_dict={x: x_val})

        expected_yval = T.norm(x_val)**2
        expected_grad_x_val = 2 * x_val

        assert isinstance(x_inner, ad.Node)
        assert T.array_equal(y_val, expected_yval)
        assert T.array_equal(grad_x_val, expected_grad_x_val)


def test_inner_product_einsum():
    for datatype in BACKEND_TYPES:
        T.set_backend(datatype)
        x = ad.Variable(name="x", shape=[3])
        x_inner = ad.einsum('i,i->', x, x)

        grad_x, = ad.gradients(x_inner, [x])

        executor = ad.Executor([x_inner, grad_x])
        x_val = T.tensor([1., 2., 3.])  # 1x3

        y_val, grad_x_val = executor.run(feed_dict={x: x_val})

        expected_yval = T.norm(x_val)**2
        expected_grad_x_val = 2 * x_val

        assert isinstance(x_inner, ad.Node)
        assert T.array_equal(y_val, expected_yval)
        assert T.array_equal(grad_x_val, expected_grad_x_val)


def test_vjps():
    for datatype in BACKEND_TYPES:
        T.set_backend(datatype)
        x = ad.Variable(name="x", shape=[2])
        A = ad.Variable(name="A", shape=[3, 2])
        v = ad.Variable(name="v", shape=[3])
        y = A @ x

        transposed_vjp_x, = ad.transposed_vjps(y, [x], v)

        executor = ad.Executor([y, transposed_vjp_x])
        x_val = T.tensor([1., 2.])  # 1x3
        A_val = T.tensor([[1., 2.], [3., 4.], [5, 6]])
        v_val = T.tensor([1, 2, 3])

        y_val, transposed_vjp_x_val = executor.run(feed_dict={
            x: x_val,
            A: A_val,
            v: v_val
        })

        expected_yval = A_val @ x_val
        expected_transposed_vjp_x_val = v_val @ A_val

        assert isinstance(transposed_vjp_x, ad.Node)
        assert T.array_equal(y_val, expected_yval)
        assert T.array_equal(transposed_vjp_x_val,
                             expected_transposed_vjp_x_val)


def test_jvps():
    for datatype in BACKEND_TYPES:
        T.set_backend(datatype)
        x1 = ad.Variable(name="x1", shape=[2])
        A1 = ad.Variable(name="A1", shape=[3, 2])
        x2 = ad.Variable(name="x2", shape=[2])
        A2 = ad.Variable(name="A2", shape=[3, 2])
        v1 = ad.Variable(name="v1", shape=[2])
        v2 = ad.Variable(name="v2", shape=[2])
        y = A1 @ x1 + A2 @ x2

        transposed_vjp_x = ad.jvps(y, [x1, x2], [v1, v2])

        executor = ad.Executor([y, transposed_vjp_x])
        x1_val = T.tensor([1., 2.])
        A1_val = T.tensor([[1., 2.], [3., 4.], [5, 6]])
        v1_val = T.tensor([3., 4.])
        x2_val = T.tensor([1., 2.])
        A2_val = T.tensor([[1., 2.], [3., 4.], [5, 6]])
        v2_val = T.tensor([3., 4.])

        y_val, transposed_vjp_x_val = executor.run(feed_dict={
            x1: x1_val,
            A1: A1_val,
            v1: v1_val,
            x2: x2_val,
            A2: A2_val,
            v2: v2_val
        })

        expected_yval = A1_val @ x1_val + A2_val @ x2_val
        expected_transposed_vjp_x_val = A1_val @ v1_val + A2_val @ v2_val

        assert isinstance(transposed_vjp_x, ad.Node)
        assert T.array_equal(y_val, expected_yval)
        assert T.array_equal(transposed_vjp_x_val,
                             expected_transposed_vjp_x_val)


def test_jtjvps():
    for datatype in BACKEND_TYPES:
        T.set_backend(datatype)
        x = ad.Variable(name="x", shape=[2])
        A = ad.Variable(name="A", shape=[3, 2])
        v = ad.Variable(name="v", shape=[2])
        y = A @ x

        jtjvp_x, = ad.jtjvps(y, [x], [v])

        executor = ad.Executor([y, jtjvp_x])
        x_val = T.tensor([1., 2.])
        A_val = T.tensor([[1., 2.], [3., 4.], [5, 6]])
        v_val = T.tensor([3, 4])

        y_val, jtjvp_x_val = executor.run(feed_dict={
            x: x_val,
            A: A_val,
            v: v_val
        })

        expected_yval = A_val @ x_val
        expected_jtjvp_x_val = T.transpose(A_val) @ A_val @ v_val

        assert isinstance(jtjvp_x, ad.Node)
        assert T.array_equal(y_val, expected_yval)
        assert T.array_equal(jtjvp_x_val, expected_jtjvp_x_val)


def test_inner_product_hvp():
    for datatype in BACKEND_TYPES:
        T.set_backend(datatype)
        x = ad.Variable(name="x", shape=[3, 1])
        v = ad.Variable(name="v", shape=[3, 1])
        y = ad.sum(ad.transpose(x) @ x)

        grad_x, = ad.gradients(y, [x])
        Hv, = ad.hvp(output_node=y, node_list=[x], vector_list=[v])

        executor = ad.Executor([y, grad_x, Hv])
        x_val = T.tensor([[1.], [2.], [3]])  # 3x1
        v_val = T.tensor([[1.], [2.], [3]])  # 3x1
        y_val, grad_x_val, Hv_val = executor.run(feed_dict={
            x: x_val,
            v: v_val
        })

        expected_yval = T.sum(T.transpose(x_val) @ x_val)
        expected_grad_x_val = 2 * x_val
        expected_hv_val = 2 * v_val

        assert isinstance(y, ad.Node)
        assert T.array_equal(y_val, expected_yval)
        assert T.array_equal(grad_x_val, expected_grad_x_val)
        assert T.array_equal(Hv_val, expected_hv_val)


def test_hvp1():
    for datatype in BACKEND_TYPES:
        T.set_backend(datatype)
        x = ad.Variable(name="x", shape=[3, 1])
        H = ad.Variable(name="H", shape=[3, 3])
        v = ad.Variable(name="v", shape=[3, 1])
        y = ad.sum(x * (H @ x))

        grad_x, = ad.gradients(y, [x])
        Hv, = ad.hvp(output_node=y, node_list=[x], vector_list=[v])

        executor = ad.Executor([y, grad_x, Hv])
        x_val = T.tensor([[1.], [2.], [3]])  # 3x1
        v_val = T.tensor([[1.], [2.], [3]])  # 3x1
        H_val = T.tensor([[2., 0., 0.], [0., 2., 0.], [0., 0., 2.]])  # 3x3
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
        x = ad.Variable(name="x", shape=[3, 1])
        H = ad.Variable(name="H", shape=[3, 3])
        v = ad.Variable(name="v", shape=[3, 1])
        y = ad.sum(ad.transpose(x) @ H @ x)

        grad_x, = ad.gradients(y, [x])
        Hv, = ad.hvp(output_node=y, node_list=[x], vector_list=[v])

        executor = ad.Executor([y, grad_x, Hv])
        x_val = T.tensor([[1.], [2.], [3]])  # 3x1
        v_val = T.tensor([[1.], [2.], [3]])  # 3x1
        H_val = T.tensor([[2., 0., 0.], [0., 2., 0.], [0., 0., 2.]])  # 3x3
        y_val, grad_x_val, Hv_val = executor.run(feed_dict={
            x: x_val,
            H: H_val,
            v: v_val
        })
        expected_yval = T.sum(T.transpose(x_val) @ H_val @ x_val)
        expected_grad_x_val = 2 * H_val @ x_val
        expected_hv_val = T.tensor([[4.], [8.], [12.]])

        assert isinstance(y, ad.Node)
        assert T.array_equal(y_val, expected_yval)
        assert T.array_equal(grad_x_val, expected_grad_x_val)
        assert T.array_equal(Hv_val, expected_hv_val)
