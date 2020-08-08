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


def compute_coefficient_matrix(n1, n2, G):
    ret = T.ones(G[0].shape)
    for i in range(len(G)):
        if i != n1 and i != n2:
            ret = T.einsum("ij,ij->ij", ret, G[i])
    return ret


def jtjvp(inputs):

    v_A = inputs[0]
    B0 = inputs[1]
    C0 = inputs[2]
    v_B = inputs[3]
    A0 = inputs[4]
    v_C = inputs[5]
    A = [A0, B0, C0]
    v = [v_A, v_B, v_C]

    # compute G
    G = []
    for i in range(3):
        G.append(T.einsum("ij,ik->jk", A[i], A[i]))

    # compute gamma
    gamma = []
    for i in range(3):
        gamma.append([])
        for j in range(3):
            if j >= i:
                M = compute_coefficient_matrix(i, j, G)
                gamma[i].append(M)
            else:
                M = gamma[j][i]
                gamma[i].append(M)

    # fast hessian contract
    ret = []
    for n in range(3):
        ret.append(T.zeros(A[n].shape))
        for p in range(3):
            M = gamma[n][p]
            if n == p:
                ret[n] += T.einsum("iz,zr->ir", v[p], M)
            else:
                B = T.einsum("jr,jz->rz", A[p], v[p])
                ret[n] += T.einsum("iz,zr,rz->ir", A[n], M, B)

    return [ret[0], ret[1], ret[2]]
