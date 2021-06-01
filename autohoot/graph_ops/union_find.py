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

###############################################################################
# Defines a union find datastruct.
# Not optimized.
# User must inherit this class and override assign method.
###############################################################################


class UFBase():
    def __init__(self, values):
        self.parent_map = {}
        self.roots = {}
        for value in values:
            self.parent_map[value] = value

    def assign():
        """
            Assigns a distinct value for each group.
        """
        raise NotImplementedError

    def root(self, n1):
        """
            Returns the root of the given node.
        """
        n = n1
        while self.parent_map[n] != n:
            n = self.parent_map[n]
        return n

    def disjoint_set(self):
        """
            Returns all the disjoint sets.
        """
        dsets = {}
        for value in self.parent_map.keys():
            rootval = self.root(value)
            if rootval in dsets:
                dsets[rootval].add(value)
            else:
                dsets[rootval] = {value}
        return dsets.values()

    def connect(self, n1, n2):
        """
            Union two nodes.
        """
        rootn1 = self.root(n1)
        rootn2 = self.root(n2)
        if rootn1 == rootn2:
            # Already connected.
            return
        self.parent_map[rootn1] = rootn2

    # Must be called after assign
    def rootval(self, n1):
        """
            Returns the assigned character of the given node's root.
        """
        return self.roots[self.root(n1)]
