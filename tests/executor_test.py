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

import autodiff as ad
import backend as T
from tests.test_utils import tree_eq, gen_dict


def test_executor_retain(backendopt):

    for datatype in backendopt:
        T.set_backend(datatype)

        x2 = ad.Variable(name="x2", shape=[3, 3])
        y = ad.sum(x2)
        z = y * 2

        x2_val = T.identity(3)
        executor = ad.Executor([y, z])
        y_val, = executor.run(feed_dict={x2: x2_val},
                              reset_graph=False,
                              out_nodes=[y])

        # This can only be run if y values are retained.
        z_val, = executor.run(feed_dict={}, reset_graph=False, out_nodes=[z])


def test_executor_dependent(backendopt):

    for datatype in backendopt:
        T.set_backend(datatype)

        A = ad.Variable(name="A", shape=[3, 3])
        B = ad.Variable(name="B", shape=[3, 3])
        AA = ad.einsum('ab,ab->', A, A)
        BB = ad.einsum('ab,ab->', B, B)
        AB = ad.einsum('ab,ab->', A, B)

        out_A = AA + AB
        out_B = AB + AA

        executor = ad.Executor({out_A, out_B})

        data = gen_dict([A, B])
        A_val, = executor.run(feed_dict=data,
                              reset_graph=False,
                              out_nodes=[out_A])
        data2 = gen_dict([A])
        data2.update({B: data[B]})
        B_val, = executor.run(feed_dict=data2, out_nodes=[out_B])
        # This is checking A's val is not reused in B_val computationA.
        assert A_val != B_val


def test_executor_debug_symmetry(backendopt):
    for datatype in backendopt:
        T.set_backend(datatype)

        A = ad.Variable(name="A", shape=[3, 3], symmetry=[[0, 1]])
        out = ad.einsum("ab,bc->ac", A, A)
        A_val = T.random((3, 3))
        A_val += T.transpose(A_val)

        executor = ad.Executor([out])
        executor.run(feed_dict={A: A_val}, debug=True)


def test_executor_debug_orthonormal(backendopt):
    for datatype in backendopt:
        T.set_backend(datatype)

        A = ad.Matrix(name="A", shape=[3, 3], orthonormal='row')
        out = ad.einsum("ab,bc->ac", A, A)
        A_val, _, _ = T.svd(T.random((3, 3)))

        executor = ad.Executor([out])
        executor.run(feed_dict={A: A_val}, debug=True)
