import autodiff as ad
import backend as T

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


def test_chainjacobian():

    for datatype in BACKEND_TYPES:
        T.set_backend(datatype)

        x1 = ad.Variable(name="x1", shape=[2, 2, 2])
        x2 = ad.Variable(name="x2", shape=[2, 2, 2])
        x1.set_in_indices_length(1)
        x2.set_in_indices_length(2)

        y = ad.chainjacobian(x1, x2)

        executor = ad.Executor([y])

        x1_val = T.tensor([[[1, 1], [1, 1]], [[1, 1], [1, 1]]])
        x2_val = T.tensor([[[1, 1], [1, 1]], [[1, 1], [1, 1]]])
        y_val, = executor.run(feed_dict={x1: x1_val, x2: x2_val})

        expected_y_val = T.einsum("abc,bcd->ad", x1_val, x2_val)

        assert isinstance(y, ad.Node)
        assert T.array_equal(y_val, expected_y_val)


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
        # jacobian_z_y = T.einsum("ae,bf->abef", I, I)
        # jacobian_y_x2 = T.einsum("ec,fd->efcd", I, I)
        # jacobian_z_x2 = T.einsum("abef,efcd->abcd", jacobian_z_y, jacobian_y_x2)
        #               = T.einsum("ae,bf,ec,fd->abcd", I, I, I, I)
        #               = T.einsum("ac,bd->abcd", I, I)
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

        expected_jacobian_x2_val = T.tensor(1.)

        assert isinstance(z, ad.Node)
        assert isinstance(jacobian_x2, ad.Node)
        assert T.array_equal(z_val, x1_val + x2_val + x3_val)
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


def test_mul_jacobian():

    for datatype in BACKEND_TYPES:
        T.set_backend(datatype)

        x1 = ad.Variable(name="x1", shape=[2, 2])
        x2 = ad.Variable(name="x2", shape=[2, 2])
        y = x1 * x2

        jacobian_x1, jacobian_x2 = ad.jacobians(y, [x1, x2])
        executor = ad.Executor([y, jacobian_x1, jacobian_x2])

        x1_val = T.tensor([[1, 2], [3, 4]])
        x2_val = T.tensor([[5, 6], [7, 8]])
        y_val, jacobian_x1_val, jacobian_x2_val = executor.run(feed_dict={
            x1: x1_val,
            x2: x2_val
        })

        I = T.identity(2)
        expected_jacobian_x1_val = T.einsum("ai,bj,ij->abij", I, I, x2_val)
        expected_jacobian_x2_val = T.einsum("ai,bj,ij->abij", I, I, x1_val)

        assert isinstance(y, ad.Node)
        assert T.array_equal(y_val, x1_val * x2_val)
        assert T.array_equal(jacobian_x1_val, expected_jacobian_x1_val)
        assert T.array_equal(jacobian_x2_val, expected_jacobian_x2_val)


def test_mul_jacobian_scalars():

    for datatype in BACKEND_TYPES:
        T.set_backend(datatype)

        x1 = ad.Variable(name="x1", shape=[])
        x2 = ad.Variable(name="x2", shape=[])
        y = x1 * x2

        jacobian_x1, jacobian_x2 = ad.jacobians(y, [x1, x2])

        executor = ad.Executor([y, jacobian_x1, jacobian_x2])

        x1_val = T.tensor(1.)
        x2_val = T.tensor(2.)
        y_val, jacobian_x1_val, jacobian_x2_val = executor.run(feed_dict={
            x1: x1_val,
            x2: x2_val
        })

        expected_jacobian_x1_val = x2_val
        expected_jacobian_x2_val = x1_val

        assert isinstance(y, ad.Node)
        assert T.array_equal(y_val, x1_val * x2_val)
        assert T.array_equal(jacobian_x1_val, expected_jacobian_x1_val)
        assert T.array_equal(jacobian_x2_val, expected_jacobian_x2_val)


def test_mul_jacobian_one_scalar():

    for datatype in BACKEND_TYPES:
        T.set_backend(datatype)

        x1 = ad.Variable(name="x1", shape=[])
        x2 = ad.Variable(name="x2", shape=[2, 2])

        # test both cases of left and right multiply a scalar
        for y in [x1 * x2, x2 * x1]:

            jacobian_x1, jacobian_x2 = ad.jacobians(y, [x1, x2])
            executor = ad.Executor([y, jacobian_x1, jacobian_x2])

            x1_val = T.tensor(2.)
            x2_val = T.tensor([[5, 6], [7, 8]])
            y_val, jacobian_x1_val, jacobian_x2_val = executor.run(feed_dict={
                x1: x1_val,
                x2: x2_val
            })

            I = T.identity(2)
            expected_jacobian_x1_val = T.einsum("ai,bj,ij->ab", I, I, x2_val)
            expected_jacobian_x2_val = x1_val * T.einsum("ai,bj->abij", I, I)

            assert isinstance(y, ad.Node)
            assert T.array_equal(y_val, x1_val * x2_val)
            assert T.array_equal(jacobian_x1_val, expected_jacobian_x1_val)
            assert T.array_equal(jacobian_x2_val, expected_jacobian_x2_val)


def test_jacobian_einsum():

    for datatype in BACKEND_TYPES:
        T.set_backend(datatype)

        x1 = ad.Variable(name="x1", shape=[3, 3, 3])
        x2 = ad.Variable(name="x2", shape=[3, 3, 3])
        y = ad.einsum("ikl,jkl->ijk", x1, x2)

        jacobian_x1, jacobian_x2 = ad.jacobians(y, [x1, x2])
        executor = ad.Executor([y, jacobian_x1, jacobian_x2])

        x1_val = T.random((3, 3, 3))
        x2_val = T.random((3, 3, 3))
        y_val, jacobian_x1_val, jacobian_x2_val = executor.run(feed_dict={
            x1: x1_val,
            x2: x2_val,
        })

        I = T.identity(3)
        expected_jacobian_x1_val = T.einsum("im,kn,jno->ijkmno", I, I, x2_val)
        expected_jacobian_x2_val = T.einsum("jm,kn,ino->ijkmno", I, I, x1_val)

        assert isinstance(y, ad.Node)
        assert T.array_equal(y_val, T.einsum("ikl,jkl->ijk", x1_val, x2_val))
        assert T.array_equal(jacobian_x1_val, expected_jacobian_x1_val)
        assert T.array_equal(jacobian_x2_val, expected_jacobian_x2_val)


def test_jacobian_summation_einsum():
    for datatype in BACKEND_TYPES:
        T.set_backend(datatype)
        x = ad.Variable(name="x", shape=[2, 2])
        x_sum = ad.einsum('ij->', x)

        grad_x, = ad.jacobians(x_sum, [x])

        executor = ad.Executor([x_sum, grad_x])
        x_val = T.tensor([[1., 2.], [3., 4.]])

        x_sum_val, grad_x_val = executor.run(feed_dict={x: x_val})

        expected_x_sum_val = T.sum(x_val)
        expected_grad_x_val = T.ones_like(x_val)

        assert T.array_equal(x_sum_val, expected_x_sum_val)
        assert T.array_equal(grad_x_val, expected_grad_x_val)


def test_jacobian_summation_einsum_2():
    for datatype in BACKEND_TYPES:
        T.set_backend(datatype)
        x = ad.Variable(name="x", shape=[2, 2])
        y = ad.Variable(name="y", shape=[2, 2])
        out = ad.einsum('ij,ab->ab', x, y)

        grad_x, = ad.jacobians(out, [x])
        executor = ad.Executor([out, grad_x])
        x_val = T.tensor([[1., 2.], [3., 4.]])
        y_val = T.tensor([[5., 6.], [7., 8.]])

        out_val, grad_x_val = executor.run(feed_dict={x: x_val, y: y_val})

        expected_out_val = T.einsum('ij,ab->ab', x_val, y_val)
        expected_grad_x_val = T.einsum('ij,ab->abij', T.ones(x_val.shape),
                                       y_val)

        assert T.array_equal(out_val, expected_out_val)
        assert T.array_equal(grad_x_val, expected_grad_x_val)


def test_jacobian_trace_einsum():
    for datatype in BACKEND_TYPES:
        T.set_backend(datatype)
        x = ad.Variable(name="x", shape=[2, 2])
        trace = ad.einsum('ii->', x)

        grad_x, = ad.jacobians(trace, [x])

        executor = ad.Executor([trace, grad_x])
        x_val = T.tensor([[1., 2.], [3., 4.]])

        trace_val, grad_x_val = executor.run(feed_dict={x: x_val})

        expected_trace_val = T.einsum('ii->', x_val)
        expected_grad_x_val = T.eye(2)

        assert T.array_equal(trace_val, expected_trace_val)
        assert T.array_equal(grad_x_val, expected_grad_x_val)


def test_hessian_quadratic():

    for datatype in BACKEND_TYPES:
        T.set_backend(datatype)

        x = ad.Variable(name="x", shape=[3])
        H = ad.Variable(name="H", shape=[3, 3])
        y = ad.einsum("i,ij,j->", x, H, x)

        hessian = ad.hessian(y, [x])
        executor = ad.Executor([hessian[0][0]])

        x_val = T.random(3)
        H_val = T.random((3, 3))
        hessian_val, = executor.run(feed_dict={x: x_val, H: H_val})

        assert T.array_equal(hessian_val, H_val + T.transpose(H_val))
