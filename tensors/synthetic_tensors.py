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


def init_rand_cp(dim, size, rank):

    X = T.random([size for _ in range(dim)])

    A_list = []
    for i in range(dim):
        A_list.append(T.random((size, rank)))

    return A_list, X


def init_rand_tucker(dim, size, rank):
    assert size > rank

    X = T.random([size for _ in range(dim)])
    core = T.random([rank for _ in range(dim)])

    A_list = []
    for i in range(dim):
        # for Tucker, factor matrices are orthogonal
        mat, _, _ = T.svd(T.random((size, rank)))
        A_list.append(mat[:, :rank])

    return A_list, core, X
