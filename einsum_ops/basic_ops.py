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

einsum = ad.EinsumNode.create
from utils import indices_to_subscripts


def transpose(node):
    # TODO: let it handle general tensor transpose
    assert len(node.shape) == 2
    return einsum("ab->ba", node)


def sum(node, axis=None):
    if axis != None:
        raise Exception(f"Sum with axis {axis} is not implemented.")

    subscripts = indices_to_subscripts([list(range(len(node.shape)))], [],
                                       len(node.shape))
    return einsum(subscripts, node)


def tensordot(node_A, node_B, axes):
    """
    Compute tensor dot product along specified axes.

    Given node_A and node_B, and an array_like object containing two array_like objects,
    (a_axes, b_axes), sum the products of node_A’s and node_B’s elements over the axes specified.

    Example: for 4-d tensors node_A and node_B,
    tensordot(node_A, node_B, axes=[[2,3], [0,1]]) is same as
    einsum("abcd,cdef->abef", node_A, node_B).
    """
    assert len(axes) == 2
    assert len(axes[0]) == len(axes[1])

    dim = len(node_A.shape) + len(node_B.shape) - len(axes[0])
    input_indices_A = list(range(len(node_A.shape)))

    index_acc = len(node_A.shape)
    input_indices_B = [0] * len(node_B.shape)

    for i in range(len(node_B.shape)):
        if i not in axes[1]:
            input_indices_B[i] = index_acc
            index_acc += 1
    for i in range(len(axes[1])):
        input_indices_B[axes[1][i]] = input_indices_A[axes[0][i]]

    assert index_acc == dim
    out_indices = [
        v for (i, v) in enumerate(input_indices_A) if i not in axes[0]
    ]
    out_indices += [
        v for (i, v) in enumerate(input_indices_B) if i not in axes[1]
    ]

    subscripts = indices_to_subscripts([input_indices_A, input_indices_B],
                                       out_indices, dim)
    return einsum(subscripts, node_A, node_B)
