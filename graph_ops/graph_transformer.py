"""
    This file will contains the equivalent graph transformations.
    These are not optimizations but equivalent transforms.

    Currently it includes 
        * Linearization
"""
from utils import find_topo_sort, OutputInjectedMode
import autodiff as ad


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


def distribute(binary_op_node, output):
    """ Distribute the operations. E.g (A + B) * C = A * C + B * C 

    Currently only consider the case where the binary_op is plus node.

    Args:
        binary_op_node: This is the plus node
        output: This is the (A + B) * C node.
    Return:
        The new output node that already distribute the computation.
    """
    assert isinstance(binary_op_node, ad.AddNode)
    assert isinstance(output, ad.EinsumNode)
    assert binary_op_node in output.inputs

    other_inputs = filter(lambda x: x != binary_op_node, output.inputs)

    # Then find the childs, the binary op should only have two.
    A, B = binary_op_node.inputs
    AC_seq = [
        tmp if tmp.name != binary_op_node.name else A for tmp in output.inputs
    ]
    BC_seq = [
        tmp if tmp.name != binary_op_node.name else B for tmp in output.inputs
    ]
    AC = ad.einsum(output.einsum_subscripts, *AC_seq)
    BC = ad.einsum(output.einsum_subscripts, *BC_seq)
    return AC + BC
