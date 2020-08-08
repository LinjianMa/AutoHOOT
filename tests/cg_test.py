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

import backend as T
from utils import conjugate_gradient

def test_HinverseG(backendopt):
    for datatype in backendopt:
        T.set_backend(datatype)

        N = 10
        T.seed(1224)

        A = T.random([N, N])
        A = T.transpose(A) @ A
        A = A + T.identity(N)
        b = T.random([N])

        def hess_fn(x):
            return [T.einsum("ab,b->a", A, x[0])]

        error_tol = 1e-9
        x, = conjugate_gradient(hess_fn, [b], error_tol)
        assert (T.norm(T.abs(T.einsum("ab,b->a", A, x) - b)) <= 1e-4)
