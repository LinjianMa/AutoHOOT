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
from autohoot.utils import CharacterGetter
from autohoot.einsum_graph.union_find import UFBase


class DimInfo(object):
    """The einsum node dimension information"""
    def __init__(self, node, dim_index, node_index=None, temp_node_name=None):
        """
        dim_index: the index of the dimension in the node.
        node_index: the index of the node in a specific node list (can be None if not specified).
        temp_node_name: the temporary name used to override the node name (can be None if not specified).
        """
        self.node = node
        self.dim_index = dim_index
        self.node_index = node_index
        if temp_node_name != None:
            self.node_name = temp_node_name
        else:
            self.node_name = node.name

        self.name = f"{self.node_name}-{self.node_index}-{self.dim_index}"

    def __str__(self):
        return self.name


### Assign each UF group a character val.
class UF(UFBase):
    def __init__(self, dims_info):
        super().__init__(dims_info)
        self.cg = CharacterGetter()

    def assign(self):
        """
            Get all parent and assign a character for each group.
            Should be only called once.
        """
        assert len(self.roots) == 0
        dsets = {}
        for value in self.parent_map.keys():
            rootval = self.root(value)
            if rootval in dsets:
                dsets[rootval].add(value)
            else:
                dsets[rootval] = {value}
        # We want to fix the character assignment order.
        # The hash is dependent on the output dim index and connection index.
        def sort_hash(pair):
            k, dset = pair
            hash_strings = [
                f'{dim_info.node_name}-{dim_info.dim_index}'
                for dim_info in dset
            ]
            return '+'.join(sorted(hash_strings))

        for k, v in sorted(list(dsets.items()), key=sort_hash):
            self.roots[k] = self.cg.getchar()
