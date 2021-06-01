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
import numpy as np  # This is used for generate random numbers.


def gen_dict(input_nodes):
    """Generates a random dict for executor to use.
    """
    feed_dict = {}
    for i_node in input_nodes:
        feed_dict[i_node] = T.random(i_node.shape)
    return feed_dict


def float_eq(A, B, tol=1e-8):
    return (abs(T.to_numpy(A) - T.to_numpy(B)) < tol).all()


def tree_eq(out, new_out, input_nodes, tol=1e-8):
    """Compares whether two output (based on the same set of inputs are equal.
    """
    feed_dict = gen_dict(input_nodes)

    executor = ad.Executor([out])
    out_val, = executor.run(feed_dict=feed_dict)

    executor = ad.Executor([new_out])
    new_out_val, = executor.run(feed_dict=feed_dict)
    return float_eq(out_val, new_out_val, tol)
