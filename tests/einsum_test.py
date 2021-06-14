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
from autohoot.graph_ops.graph_transformer import optimize
from autohoot.utils import CharacterGetter
from tests.test_utils import float_eq


def test_large_matmul_chain(backendopt):
    n = 60
    size = 3
    for datatype in backendopt:
        T.set_backend(datatype)

        # build the graph of x_1 @ ... @ x_n
        x_list = [
            ad.Variable(name=f"x{i}", shape=[size, size]) for i in range(n)
        ]
        cg = CharacterGetter()
        prev_char = cg.getchar()
        left_char = prev_char
        for i in range(n):
            x_list[i].subscripts = f"{prev_char}{cg.getchar()}"
            prev_char = x_list[i].subscripts[1]
        right_char = prev_char
        input_subs = ','.join([node.subscripts for node in x_list])
        einsum_subscripts = input_subs + '->' + left_char + right_char

        out = ad.einsum(einsum_subscripts, *x_list)
        # decompose the large einsum, and rewrite the einsum expression of the
        # generated einsum tree so there's no unicode character
        out = optimize(out)
        executor = ad.Executor([out])

        x_val_list = [T.random([size, size]) for _ in range(n)]
        out_val, = executor.run(feed_dict=dict(zip(x_list, x_val_list)))

        out_val_matmul = x_val_list[0]
        for i in range(1, n):
            out_val_matmul = out_val_matmul @ x_val_list[i]
        assert float_eq(out_val, out_val_matmul, tol=1e-2)
