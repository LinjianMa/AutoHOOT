"""
    This file will contains the equivalent graph transformations.
    These are not optimizations but equivalent transforms.

    Currently it includes 
        * Linearization
"""
import logging

from utils import find_topo_sort, OutputInjectedMode
from utils import replace_node
from graph_ops.graph_optimizer import find_sub_einsumtree, fuse_einsums, UF, cross_einsum_connect
from graph_ops.graph_generator import generate_optimal_tree
from graph_ops.graph_dedup import dedup, declone
import autodiff as ad
import copy

FORMAT = '[%(asctime)-15s %(filename)s:%(lineno)s] %(message)s'

logger = logging.getLogger('optimizer')
logging.basicConfig(format=FORMAT)
logger.setLevel(logging.DEBUG)


def linearize(output_node):
    """Linearize a graph by adding clone nodes for computation optimization.

    Args:
        output_node: A single node.
    Returns: 
        None. Update is inplace. 

    NOTE: If you ever need to debug this function, the generated name is 
        inconsistent becasue of the added edges.

    """
    # Need to create new nodes for whichever node that has 2 or more outgoing edges.
    all_nodes = find_topo_sort([output_node])
    # Inject outpus relationship.
    with OutputInjectedMode(all_nodes):
        for n in all_nodes:
            if len(n.outputs) > 1:
                for n_o in n.outputs:
                    n_new = copy_tree(n)
                    n_o.set_inputs([
                        tmp if tmp.name != n.name else n_new
                        for tmp in n_o.inputs
                    ])


def _distribute(binary_op_node, output):
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


def distribute_tree(output):
    """ Distribute a tree of einsum and add nodes.

    Behavior undefined if there are other kind of nodes.

    Args:
        output: The output of a tree.

    Returns:
        output: a newly generated einsum tree with operands distributed. 
    
    Idea:
        1. Construct the output tree.
        2. Find binary op.
        3. Apply distribute.
        4. Iterate 1->3
    """
    def get_first_binary_op(nodes):
        for node in nodes:
            if isinstance(node, ad.AddNode) and len(node.outputs) >= 1:
                has_einsum_nodes = all(
                    [isinstance(x, ad.EinsumNode) for x in node.outputs])
                if has_einsum_nodes:
                    return node
        return None

    assert isinstance(output, ad.EinsumNode)

    while 1:
        all_nodes = find_topo_sort([output])
        with OutputInjectedMode(all_nodes):
            first_binary_op = get_first_binary_op(all_nodes)
            if first_binary_op is None:
                break
            for einsum_node in first_binary_op.outputs:
                if isinstance(einsum_node, ad.AddNode):
                    continue
                assert isinstance(einsum_node, ad.EinsumNode)
                new_node = _distribute(first_binary_op, einsum_node)
                replace_node(einsum_node, new_node)
                if einsum_node == output:
                    output = new_node
    # This is need for source generation.
    output.set_inputs(output.inputs)
    return output


def copy_tree(node):
    """
        Copies a tree, creating new nodes for each one in the tree.
    """
    # Track back the original Variable node.
    if isinstance(node, ad.CloneNode):
        assert len(node.inputs) == 1
        return copy_tree(node.inputs[0])
    if isinstance(node, ad.VariableNode):
        return node.clone()
    new_inputs = []
    for i_node in node.inputs:
        new_i_node = copy_tree(i_node)
        new_inputs.append(new_i_node)
    new_node = copy.deepcopy(node)
    new_node.set_inputs(new_inputs)
    return new_node


def rewrite_einsum_expr(einsum_node):
    """
        Rewrites the einsum expression of a node.

        Inplace update.

    Returns:
        uf (type: graph_ops.graph_optimizer.UF): the union_find set of the input
        
    """
    assert (isinstance(einsum_node, ad.EinsumNode))
    input_nodes = einsum_node.inputs

    # TODO: Get all the einsum nodes in the computation graph.
    # Note that the order doesn't matter!
    all_nodes = [einsum_node] + input_nodes

    # Create a map
    for node in all_nodes:
        node.literals = [node.name + str(i) for i in range(len(node.shape))]

    literal_names = []
    for node in all_nodes:
        literal_names += node.literals

    # For any literal that are the same, get their pos and connect.
    uf = UF(literal_names)
    cross_einsum_connect(uf, einsum_node)

    uf.assign()
    # Assign literals
    for node in all_nodes:
        node.subscripts = "".join(
            [uf.rootval(literal_name) for literal_name in node.literals])

    new_input_subs = [node.subscripts for node in input_nodes]
    new_subscripts = ",".join(new_input_subs) + "->" + einsum_node.subscripts
    einsum_node.einsum_subscripts = new_subscripts
    logger.info(f"Rewrite to new subscript: {new_subscripts}")

    return uf


def optimize(node):
    """Optimize a graph with a single output node.

    Args:
        node: The output node.
    Returns:
        node: The newly generated node.
    """
    node = distribute_tree(node)
    linearize(node)
    all_nodes = find_topo_sort([node])
    with OutputInjectedMode(all_nodes):
        trees = find_sub_einsumtree(node)
        for tree in trees:
            out_node, in_nodes = tree
            new_z = fuse_einsums(out_node, in_nodes)
            new_z = generate_optimal_tree(new_z)
            replace_node(out_node, new_z)
    node = declone(node)
    dedup(node)
    return node
