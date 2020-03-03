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
