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

import time
import autodiff as ad
import backend as T

from examples.cpd import cpd_graph
from tensors.synthetic_tensors import init_rand_cp
from graph_ops.graph_transformer import optimize, simplify

BACKEND_TYPES = ['numpy']


def cpd_als_orthogonal_constraint(dim,
                                  size,
                                  rank,
                                  num_iter,
                                  lamda,
                                  input_val=[]):

    A_list, input_tensor, loss, residual = cpd_graph(dim, size, rank)

    for i in range(dim):
        ortho_res = ad.einsum("ab,ac->bc", A_list[i], A_list[i]) - ad.identity(
            A_list[i].shape[1])
        loss += lamda * ad.einsum("ab,ab->", ortho_res, ortho_res)

    full_hessian = ad.hessian(loss, A_list)
    hessians = [simplify(full_hessian[i][i]) for i in range(len(full_hessian))]

    grads = ad.gradients(loss, A_list)
    grads = [simplify(grad) for grad in grads]

    updates = [
        ad.tensordot(ad.tensorinv(hes), grad, [[2, 3], [0, 1]])
        for (hes, grad) in zip(hessians, grads)
    ]

    new_A_list = [A - update for (A, update) in zip(A_list, updates)]

    executor = ad.Executor(new_A_list)
    executor_loss = ad.Executor([simplify(loss)])

    if input_val == []:
        A_val_list, input_tensor_val = init_rand_cp(dim, size, rank)
    else:
        A_val_list, input_tensor_val = input_val

    for iter in range(num_iter):
        # als iterations
        for i in range(len(A_list)):

            feed_dict = dict(zip(A_list, A_val_list))
            feed_dict.update({input_tensor: input_tensor_val})
            A_val_list[i], = executor.run(feed_dict=feed_dict,
                                          out_nodes=[new_A_list[i]])

        feed_dict = dict(zip(A_list, A_val_list))
        feed_dict.update({input_tensor: input_tensor_val})
        loss_val, = executor_loss.run(feed_dict=feed_dict)
        print(f'At iteration {iter} the loss is: {loss_val}')
        # print(A_val_list[0].T @ A_val_list[0])

    return A_val_list


if __name__ == "__main__":
    cpd_als_orthogonal_constraint(dim=3,
                                  size=10,
                                  rank=5,
                                  num_iter=100,
                                  lamda=0.1)
