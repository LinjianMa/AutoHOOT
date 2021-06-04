This folder includes the optimization operations on the computation graphs.

Files included:

1. *optimal_tree*: generate (constrained) optimized contraction path.

2. *graph_transformer*: transform graphs. Detailed are presented below.

3. *graph_pruning*: prune unnecessary nodes in the graph.

4. *graph_inv_optimizer*: optimize (generalized) inverse based on symbolic rules.

5. *graph_dedup*: remove duplicated nodes in the graph.

6. *graph_als_optimizer*: optimzie the graph for alternating least squares optimization algorithms.

7. *graph_utils*: utility functions used in the optimization.

Definition:

* Einsum Tree: A tree with all intermediate nodes being Einsum Nodes. Leaf
nodes are non-einsum nodes.

Several Core Algorithms in graph_transformer:

1. *Linearization*: Copies a node if it is used by >2 nodes, copies the whole
subtree if the node is in the tree. 

2. *Fusion*: Fuse a pure einsum tree into a long expression.

3. *Find Fusable Tree*: For a graph, find multiple fusable einsum trees.

4. *Distribute*: For a graph with + and einsums, make the + execute at last.
