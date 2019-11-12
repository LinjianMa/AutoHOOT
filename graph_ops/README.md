This folder includes the optimization operations on the computation graphs.

Definition:

* Einsum Tree: A tree with all intermediate nodes being Einsum Nodes. Leaf
nodes are non-einsum nodes.

Several Core Algorithms:

1. *Linearization*: Copies a node if it is used by >2 nodes, copies the whole
subtree if the node is in the tree. 

2. *Fusion*: Fuse a pure einsum tree into a long expression.

3. *Find Fusable Tree*: For a graph, find multiple fusable einsum trees.

4. *Distribute*: For a graph with + and einsums, make the + execute at last.
