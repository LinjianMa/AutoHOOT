from utils import find_topo_sort, OutputInjectedMode
from visualizer import print_computation_graph


def linearize(output_nodes, input_nodes):
    """
        Linearize a graph by adding clone nodes for computation optimization.

        NOTE: If you ever need to debug this function, the generated name is 
            inconsistent becasue of the added edges.
    """
    # Need to create new nodes for whichever node that has 2 or more outgoing edges.
    # Note that
    assert len(output_nodes) > 0
    assert len(input_nodes) > 0
    all_nodes = find_topo_sort(output_nodes)
    # Inject outpus relationship.
    with OutputInjectedMode(all_nodes):
        for n in all_nodes:
            if len(n.outputs) > 1:
                for n_o in n.outputs:
                    n_new = n.clone()
                    n_o.set_inputs(*[
                        tmp if tmp.name != n.name else n_new
                        for tmp in n_o.inputs
                    ])

    return output_nodes, input_nodes
