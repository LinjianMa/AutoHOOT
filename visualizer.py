from graphviz import Digraph
import autodiff as ad
import numpy as np
from utils import einsum_grad_subscripts, find_topo_sort, topo_sort_dfs, sum_node_list, inner_product
from utils import OutputInjectedMode

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# Specify the graph here.
# a1 = ad.Variable(name="a1", shape=[3, 2])
# a2 = ad.Variable(name="a2", shape=[2, 3])
#
# b1 = ad.Variable(name="b1", shape=[3, 2])
# b2 = ad.Variable(name="b2", shape=[2, 3])
#
# x = ad.einsum('ik,kj->ij', a1, a2)
# y = ad.einsum('jl,ls->js', b1, b2)
#
# z = ad.einsum('ij, js->is', x, y)
#
# executor = ad.Executor([z])
# print_computation_graph(z)
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


def print_computation_graph(output_node):
    topo_order = find_topo_sort([output_node])

    inputs = filter(lambda x: isinstance(x, ad.VariableNode), topo_order)
    with OutputInjectedMode(topo_order):
        outputs = filter(lambda x: len(x.outputs) == 0, topo_order)

        dot = Digraph(comment='Poorman Computation Graph')
        with dot.subgraph() as s:
            s.attr(rank='same')
            for n in inputs:
                s.node(n.name, color='blue')
        with dot.subgraph() as s:
            s.attr(rank='same')
            for n in outputs:
                s.node(n.name, color='red')
        for node in topo_order:
            dot.node(node.name, node.name)
            for node_i in node.inputs:
                dot.edge(node_i.name, node.name)

        print(dot.source)


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# Please visit a online digraph visualizer like
# https://dreampuf.github.io/GraphvizOnline/
