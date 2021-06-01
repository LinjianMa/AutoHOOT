# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import autohoot.autodiff as ad
import autohoot.backend as T

backends = ["numpy"]


def test_einsum():

    for datatype in backends:
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


def test_vjps():
    for datatype in backends:
        T.set_backend(datatype)
        x = ad.Variable(name="x", shape=[2])
        A = ad.Variable(name="A", shape=[3, 2])
        v = ad.Variable(name="v", shape=[3])
        y = ad.einsum('ab, b->a', A, x)

        transposed_vjp_x, = ad.transposed_vjps(y, [x], v)

        executor = ad.Executor([y, transposed_vjp_x])
        x_val = T.tensor([1., 2.])  # 1x3
        A_val = T.tensor([[1., 2.], [3., 4.], [5, 6]])
        v_val = T.tensor([1., 2., 3.])

        y_val, transposed_vjp_x_val = executor.run(feed_dict={
            x: x_val,
            A: A_val,
            v: v_val
        })

        expected_yval = T.einsum('ab, b->a', A_val, x_val)
        expected_transposed_vjp_x_val = T.einsum('b, ba->a', v_val, A_val)

        assert isinstance(transposed_vjp_x, ad.Node)
        assert T.array_equal(y_val, expected_yval)
        assert T.array_equal(transposed_vjp_x_val,
                             expected_transposed_vjp_x_val)


def test_jvps():
    for datatype in backends:
        T.set_backend(datatype)
        x1 = ad.Variable(name="x1", shape=[2])
        A1 = ad.Variable(name="A1", shape=[3, 2])
        x2 = ad.Variable(name="x2", shape=[2])
        A2 = ad.Variable(name="A2", shape=[3, 2])
        v1 = ad.Variable(name="v1", shape=[2])
        v2 = ad.Variable(name="v2", shape=[2])
        y = ad.einsum('ab, b->a', A1, x1) + ad.einsum('ab, b->a', A2, x2)

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

        expected_yval = T.einsum('ab, b->a', A1_val, x1_val) + T.einsum(
            'ab, b->a', A2_val, x2_val)

        expected_transposed_vjp_x_val = T.einsum(
            'ab, b->a', A1_val, v1_val) + T.einsum('ab, b->a', A2_val, v2_val)

        assert isinstance(transposed_vjp_x, ad.Node)
        assert T.array_equal(y_val, expected_yval)
        assert T.array_equal(transposed_vjp_x_val,
                             expected_transposed_vjp_x_val)


def test_jtjvps():
    for datatype in backends:
        T.set_backend(datatype)
        x = ad.Variable(name="x", shape=[2])
        A = ad.Variable(name="A", shape=[3, 2])
        v = ad.Variable(name="v", shape=[2])
        y = ad.einsum('ab, b->a', A, x)

        jtjvp_x, = ad.jtjvps(y, [x], [v])

        executor = ad.Executor([y, jtjvp_x])
        x_val = T.tensor([1., 2.])
        A_val = T.tensor([[1., 2.], [3., 4.], [5, 6]])
        v_val = T.tensor([3., 4.])

        y_val, jtjvp_x_val = executor.run(feed_dict={
            x: x_val,
            A: A_val,
            v: v_val
        })

        expected_yval = T.einsum('ab, b->a', A_val, x_val)
        expected_jtjvp_x_val = T.einsum('ba, ac->bc', T.transpose(A_val),
                                        A_val)
        expected_jtjvp_x_val = T.einsum('ab, b->a', expected_jtjvp_x_val,
                                        v_val)

        assert isinstance(jtjvp_x, ad.Node)
        assert T.array_equal(y_val, expected_yval)
        assert T.array_equal(jtjvp_x_val, expected_jtjvp_x_val)


def test_inner_product_hvp():
    for datatype in backends:
        T.set_backend(datatype)
        x = ad.Variable(name="x", shape=[3, 1])
        v = ad.Variable(name="v", shape=[3, 1])
        y = ad.sum(ad.einsum("ab,bc->ac", ad.transpose(x), x))

        grad_x, = ad.gradients(y, [x])
        Hv, = ad.hvp(output_node=y, node_list=[x], vector_list=[v])

        executor = ad.Executor([y, grad_x, Hv])
        x_val = T.tensor([[1.], [2.], [3]])  # 3x1
        v_val = T.tensor([[1.], [2.], [3]])  # 3x1
        y_val, grad_x_val, Hv_val = executor.run(feed_dict={
            x: x_val,
            v: v_val
        })

        expected_yval = T.sum(T.dot(T.transpose(x_val), x_val))
        expected_grad_x_val = 2 * x_val
        expected_hv_val = 2 * v_val

        assert isinstance(y, ad.Node)
        assert T.array_equal(y_val, expected_yval)
        assert T.array_equal(grad_x_val, expected_grad_x_val)
        assert T.array_equal(Hv_val, expected_hv_val)


def test_tensorinv_matrix():
    for datatype in backends:
        T.set_backend(datatype)
        x = ad.Variable(name="x", shape=[3, 3])
        inv_x = ad.tensorinv(x)
        executor = ad.Executor([inv_x])

        x_val = T.random([3, 3])
        inv_x_val, = executor.run(feed_dict={x: x_val})
        assert T.array_equal(inv_x_val, T.inv(x_val))
