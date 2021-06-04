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
import copy
from collections import deque
from autohoot import autodiff as ad


def get_leaves(nodes):
    """
        Returns all the inputs of the nodes formed as a tree. The returned node
        must not be in the tree.
        Args:
            Nodes is a list of graph nodes.
        Returns:
            a set of nodes.
        
        For further illustration:
            A
           / \
          B  I3
         / \
        I1  I2
        
        I1, I2, I3 are the returned nodes. inputs is {A,B} 
    """
    all_inputs = set()
    for n in nodes:
        for n_i in n.inputs:
            if n_i not in nodes:
                all_inputs.add(n_i)
    return all_inputs


def copy_tree(node):
    """
        Copies a tree, creating new nodes for each one in the tree.
    """
    node_map = {}
    visited = set()
    q = deque()
    q.append(node)
    while len(q) > 0:
        tmp = q.popleft()
        if tmp in visited:
            continue
        visited.add(tmp)
        if tmp not in node_map:
            node_map[tmp] = copy.deepcopy(tmp)
        new_tmp = node_map[tmp]
        new_inputs = []

        if not isinstance(tmp, ad.OpNode):
            node_map[tmp] = copy.deepcopy(tmp)
            continue

        for t in tmp.inputs:
            if t not in node_map:
                node_map[t] = copy.deepcopy(t)
            new_inputs.append(node_map[t])
            q.append(t)
        new_tmp.set_inputs(new_inputs)
    return node_map[node]
