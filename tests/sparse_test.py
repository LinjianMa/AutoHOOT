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
import sparse
import pytaco as pt
import numpy as np
import formats
from tests.test_utils import float_eq


def test_sparse_ops():

    T.set_backend("numpy")
    size = 5

    x_sparse = T.random([size, size], format='coo', density=0.1)
    x_dense = T.to_numpy(x_sparse)
    assert isinstance(x_sparse, sparse._coo.core.COO)
    assert isinstance(x_dense, np.ndarray)

    y_dense = T.random([size, size])
    y_sparse = T.tensor(y_dense, format="coo")
    assert isinstance(y_sparse, sparse._coo.core.COO)
    assert isinstance(y_dense, np.ndarray)

    # test einsum
    einsum1 = T.einsum("ab,bc->ac", x_dense, y_dense)
    einsum2 = T.einsum("ab,bc->ac", x_dense, y_sparse)
    einsum3 = T.einsum("ab,bc->ac", x_sparse, y_sparse)
    assert float_eq(einsum1, einsum2)
    assert float_eq(einsum1, einsum3)

    # test solve_tri, first change matrices to full-rank ones
    x_sparse += T.tensor(T.identity(size), format="coo")
    y_sparse += T.tensor(T.identity(size), format="coo")
    x_dense += T.identity(size)
    y_dense += T.identity(size)
    out1 = T.solve_tri(x_sparse, y_sparse)
    out2 = T.solve_tri(x_dense, y_dense)
    assert float_eq(out1, out2)


def test_sparse_einsum_graph():

    T.set_backend("taco")
    size = 5
    coo = formats.SparseFormat([formats.compressed, formats.compressed])
    csc = formats.SparseFormat([formats.compressed, formats.dense])

    x1 = ad.Variable(name="x1", shape=[size, size], format=coo)
    x2 = ad.Variable(name="x2", shape=[size, size])
    y = ad.einsum('ik,kj->ij', x1, x2, out_format=csc)
    executor = ad.Executor([y])

    x1_val = T.random([size, size], format='coo', density=0.1)
    x2_val = T.random([size, size])

    y_val, = executor.run(feed_dict={x1: x1_val, x2: x2_val}, debug=True)

    expected_yval = T.einsum("ab,bc->ac", x1_val, x2_val)
    assert float_eq(y_val, expected_yval)
    assert isinstance(y_val, pt.pytensor.taco_tensor.tensor)


def test_sparse_optimize():

    T.set_backend("taco")
    size = 1000
    coo = formats.SparseFormat([formats.compressed, formats.compressed])

    X = ad.Variable(name="X", shape=[size, size], format=coo)
    A = ad.Variable(name="A", shape=[size, size])
    B = ad.Variable(name="B", shape=[size, size])
    y = ad.einsum('ij,is,js->ij', X, A, B, out_format=coo)
    executor = ad.Executor([y])

    X_val = T.random([size, size], format='coo', density=0.1)
    B_val = T.random([size, size])
    A_val = T.random([size, size])

    y_val, = executor.run(feed_dict={A: A_val, B: B_val, X: X_val}, debug=True)

    expected_yval = T.einsum("ij, is, js->ij", X_val, A_val, B_val)
    assert float_eq(y_val, expected_yval)
    assert isinstance(y_val, pt.pytensor.taco_tensor.tensor)
