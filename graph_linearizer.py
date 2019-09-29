from utils import find_topo_sort


class OutputInjectedMode:
    def __init__(self, nodes):
        self.nodes = nodes

    def __enter__(self):
        for n in self.nodes:
            for n_i in n.inputs:
                n_i.outputs.append(n)

    def __exit__(self, type, value, traceback):
        for n in self.nodes:
            n.outputs = []


def linearize(output_node, input_nodes):
    # Need to create new nodes for whichever node that has 2 or more outgoing edges.
    all_nodes = find_topo_sort([output_node])
    # Inject outpus relationship.
    with OutputInjectedMode(all_nodes):
        for n in all_nodes:
            if len(n.outputs) > 1:
                for n_o in n.outputs:
                    n_new = n.clone()
                    # Find n_o's input that correspond to previous name, delete.
                    # Add new cloned node as the input.
                # Redo the link.
    return output_node, input_nodes
