from graphviz import Digraph
import autodiff as ad
from utils import find_topo_sort
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
# print_computation_graph([z])
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


def graph_name(node):
    if isinstance(node, ad.CloneNode):
        return "Clone"
    elif isinstance(node, ad.AddNode) or isinstance(node, ad.AddByConstNode):
        return "Add"
    elif isinstance(node, ad.SubNode) or isinstance(node, ad.SubByConstNode):
        return "Sub"
    elif isinstance(node, ad.MulNode) or isinstance(node, ad.MulByConstNode):
        return "Mul"
    elif isinstance(node, ad.PowerNode):
        return "Power"
    elif isinstance(node, ad.MatMulNode):
        return "Matmul"
    elif isinstance(node, ad.EinsumNode):
        return f"Einsum(\"{node.einsum_subscripts}\")"
    elif isinstance(node, ad.NormNode):
        return "Norm"
    elif isinstance(node, ad.SumNode):
        return "Sum"
    elif isinstance(node, ad.TransposeNode):
        return "Transpose"
    else:
        return node.name


def print_computation_graph(output_node_list, input_nodes=[]):
    """
        ouput_node_list: a list of output nodes.
    """
    assert len(output_node_list) > 0

    topo_order = find_topo_sort(output_node_list, input_nodes)

    inputs = list(filter(lambda x: isinstance(x, ad.VariableNode), topo_order))
    with OutputInjectedMode(topo_order):

        dot = Digraph(comment='Poorman Computation Graph')

        with dot.subgraph() as s:
            s.attr(rank='same')
            for n in inputs:
                s.node(n.name, style='filled', color='aquamarine3')
        with dot.subgraph() as s:
            s.attr(rank='same')
            for n in output_node_list:
                s.node(n.name, style='filled', color='thistle')
        with dot.subgraph() as s:
            for n in topo_order:
                if (n not in output_node_list and n not in inputs):
                    s.node(n.name, style='filled', color='lightblue')

        for node in topo_order:
            dot.node(node.name, graph_name(node))
            for node_i in node.inputs:
                dot.edge(node_i.name, node.name)

        print(dot.source)


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# Please visit a online digraph visualizer like
# https://dreampuf.github.io/GraphvizOnline/
