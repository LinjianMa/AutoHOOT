from utils import find_topo_sort, OutputInjectedMode


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
