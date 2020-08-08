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
import jax.numpy as np
import backend as T

from frontend.jax_frontend import make_graph


def test_simpledot():
    def testfunc(w, b, x):
        return np.dot(w, x) + b + np.ones(5), x

    T.set_backend('jax')
    w = T.random((5, 10))
    b = T.random((5, ))
    x = T.random((10, ))
    inputs = [w, b, x]

    out_nodes, variables = make_graph(testfunc, *inputs)
    executor = ad.Executor(out_nodes)
    feed_dict = dict(zip(variables, inputs))

    outvals = executor.run(feed_dict=feed_dict)
    expect_outvals = testfunc(*inputs)

    for outval, expect_outval in zip(outvals, expect_outvals):
        assert T.norm(outval - expect_outval) < 1e-6


def test_mul():
    def testfunc(w, b, x):
        # Note: because our executor output is always a list, here a list is also
        # returned to make them consistent.
        return w * x + b,

    T.set_backend('jax')
    w = T.random((5, 10))
    b = T.random((5, 10))
    x = T.random((5, 10))
    inputs = [w, b, x]

    out_nodes, variables = make_graph(testfunc, *inputs)
    executor = ad.Executor(out_nodes)
    feed_dict = dict(zip(variables, inputs))

    outvals = executor.run(feed_dict=feed_dict)
    expect_outvals = testfunc(*inputs)

    for outval, expect_outval in zip(outvals, expect_outvals):
        assert T.norm(outval - expect_outval) < 1e-6
