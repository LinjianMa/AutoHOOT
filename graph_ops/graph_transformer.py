"""
    This file will contains the equivalent graph transformations.
    These are not optimizations but equivalent transforms.

    Currently it includes 
        * Linearization
"""
import logging

import copy
import numpy as np
from collections import deque

import autodiff as ad
from graph_ops.graph_dedup import dedup, declone
from graph_ops.graph_generator import generate_optimal_tree
from graph_ops.graph_inv_optimizer import optimize_inverse
from graph_ops.graph_optimizer import find_sub_einsumtree, fuse_einsums, UF, cross_einsum_connect
from numpy.core.einsumfunc import _parse_einsum_input
from utils import find_topo_sort, OutputInjectedMode, PseudoNode
from utils import replace_node

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
    # Inject outputs relationship.
    with OutputInjectedMode(all_nodes):
        for n in all_nodes:
            if len(n.outputs) > 1:
                for n_o in set(n.outputs):
                    n_o.set_inputs([
                        tmp if tmp.name != n.name else copy_tree(n)
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
        output: a newly generated node with add operands distributed.
    
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
    node_map = {}
    visited = set()
    q = deque()
    q.append(node)
    while len(q) > 0:
        tmp = q.popleft()
        if tmp in visited:
            continue
        visited.add(tmp)
        if tmp not in node_map:
            node_map[tmp] = copy.deepcopy(tmp)
        new_tmp = node_map[tmp]
        new_inputs = []

        if not isinstance(tmp, ad.OpNode):
            node_map[tmp] = copy.deepcopy(tmp)
            continue

        for t in tmp.inputs:
            if t not in node_map:
                node_map[t] = copy.deepcopy(t)
            new_inputs.append(node_map[t])
            q.append(t)
        new_tmp.set_inputs(new_inputs)
    return node_map[node]


def generate_einsum_info(einsum_node):
    """
        Generates the einsum information for easier programming.

        Args:
            einsum_node: All inputs must be unique.

        Returns:
            uf (type: graph_ops.graph_optimizer.UF): 
            the union_find set of the input

        Updates the subscript of the graph nodes affected.
        
    """
    assert (isinstance(einsum_node, ad.EinsumNode))

    pseudo_nodes = []
    einsum_node_literals = [
        f'{einsum_node.name}-{i}' for i in range(len(einsum_node.shape))
    ]
    p_outnode = PseudoNode(node=einsum_node, literals=einsum_node_literals)
    pseudo_nodes.append(p_outnode)

    p_innodes = []
    for k, node in enumerate(einsum_node.inputs):
        literals = [f'{node.name}-{k}-{i}' for i in range(len(node.shape))]
        p_innode = PseudoNode(node=node, literals=literals)
        pseudo_nodes.append(p_innode)
        p_innodes.append(p_innode)

    node_literals = []

    all_literals = sum([node.literals for node in pseudo_nodes], [])

    # For any literal that are the same, get their pos and connect.
    uf = UF(all_literals)
    cross_einsum_connect(uf, einsum_node, all_literals)

    uf.assign()
    # Assign literals
    for node in pseudo_nodes:
        node.generate_subscript(uf)
        # TODO(yejiayu): Remove this after cleaning up the callers.
        node.node.subscripts = node.subscript

    return uf, p_outnode, p_innodes


def rewrite_einsum_expr(einsum_node):
    """
        Rewrites the einsum expression of a node.
        Inplace update.

        Args:
            einsum_node: Allow duplicate inputs of the einsum node.

        Returns:
            uf (type: graph_ops.graph_optimizer.UF): 
            the union_find set of the input
        
    """
    assert (isinstance(einsum_node, ad.EinsumNode))
    input_nodes = einsum_node.inputs

    # TODO: Get all the einsum nodes in the computation graph.
    # Note that the order matters!

    pseudo_nodes = []
    # Here einsum node has a temporary name so that the character assignment
    # order is consistent.
    einsum_node_literals = [(f'_temp_einsum', i)
                            for i in range(len(einsum_node.shape))]
    pseudo_nodes.append(
        PseudoNode(node=einsum_node, literals=einsum_node_literals))

    for k, node in enumerate(einsum_node.inputs):
        literals = [(f'{node.name}', i, k) for i in range(len(node.shape))]
        pseudo_nodes.append(PseudoNode(node=node, literals=literals))

    node_literals = []

    all_literals = sum([node.literals for node in pseudo_nodes], [])

    # For any literal that are the same, get their pos and connect.
    uf = UF(all_literals)
    cross_einsum_connect(uf, einsum_node, all_literals)

    uf.assign()
    # Assign literals
    for node in pseudo_nodes:
        node.generate_subscript(uf)

    einsum_node_subscript = pseudo_nodes[0].subscript

    # Remove the einsum node.
    pseudo_nodes.pop(0)

    # Sort based on both the node name and subscript.
    pseudo_nodes = sorted(pseudo_nodes,
                          key=lambda pnode: pnode.node.name + pnode.subscript)

    new_input_subs = [pnode.subscript for pnode in pseudo_nodes]
    new_subscripts = ",".join(new_input_subs) + "->" + einsum_node_subscript
    einsum_node.einsum_subscripts = new_subscripts
    einsum_node.set_inputs([pnode.node for pnode in pseudo_nodes])
    logger.info(f"Rewrite to new subscript: {new_subscripts}")

    return uf


def prune_identity_nodes(einsum_node):
    """
        reduce the number of identity nodes in the
        einsum_node's inputs. Inplace update.

        Args:
            einsum_node: An fused einsum node.
    """
    assert (isinstance(einsum_node, ad.EinsumNode))
    # used to assign new characters
    uf_str, _, _ = generate_einsum_info(einsum_node)

    in_subs, out_subs, _ = _parse_einsum_input(
        (einsum_node.einsum_subscripts, *einsum_node.inputs))
    in_subs_list = in_subs.split(',')
    whole_str = out_subs + "".join(in_subs_list)

    for i, node in enumerate(einsum_node.inputs):
        node.subscripts = in_subs_list[i]

    identity_nodes = list(
        filter(lambda node: isinstance(node, ad.IdentityNode),
               einsum_node.inputs))
    variable_nodes = list(set(einsum_node.inputs) - set(identity_nodes))

    # each disjoint set in uf_identity represents the indices
    # linked by identity node
    uf_identity = UF(list(whole_str))
    for node in identity_nodes:
        uf_identity.connect(node.subscripts[0], node.subscripts[1])

    input_indices_set, output_indices_set = set(), set()
    for node in variable_nodes:
        # replace subscripts by the root chars
        sub_list = [uf_identity.root(char) for char in node.subscripts]
        node.subscripts = "".join(sub_list)
        input_indices_set |= set(sub_list)

    updated_inputs = variable_nodes
    out_sub_list = []
    for i, char in enumerate(out_subs):
        uf_root_char = uf_identity.root(char)
        if uf_root_char in output_indices_set:
            # we cannot assign the same char to two indices in the
            # output. Therefore, assign a new char, and add one
            # identity node to the inputs to show the constraint.
            new_char = uf_str.cg.getchar()
            out_sub_list.append(new_char)
            identity_node = ad.identity(einsum_node.shape[i])
            identity_node.subscripts = f"{uf_root_char}{new_char}"
            updated_inputs.append(identity_node)
        else:
            # directly assign the root char to the subscripts
            out_sub_list.append(uf_root_char)
            output_indices_set.add(uf_root_char)
    einsum_node.subscripts = "".join(out_sub_list)

    new_input_subs = [node.subscripts for node in updated_inputs]
    new_subscripts = ",".join(new_input_subs) + "->" + einsum_node.subscripts
    einsum_node.einsum_subscripts = new_subscripts
    einsum_node.set_inputs(updated_inputs)


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
            prune_identity_nodes(new_z)
            new_z = generate_optimal_tree(new_z)
            replace_node(out_node, new_z)
    node = declone(node)
    all_nodes = find_topo_sort([node])
    for node in all_nodes:
        if isinstance(node, ad.EinsumNode):
            rewrite_einsum_expr(node)

    for node in find_topo_sort([node]):
        if node.inputs != []:
            node.set_inputs(node.inputs)

    dedup(node)
    return node


def simplify(node):
    """Simplify a graph with a single output node.
    The simplified form will distribute selected operations
    (+), and fuse all connected einsums.

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
            prune_identity_nodes(new_z)
            replace_node(out_node, new_z)

    node = declone(node)
    all_nodes = find_topo_sort([node])

    # optimize inverse
    with OutputInjectedMode(all_nodes):
        for node in all_nodes:
            if isinstance(node, ad.EinsumNode):
                # To make sure the same einsum nodes have the same same,
                # so that we can collapse the add node.
                rewrite_einsum_expr(node)
            if node.inputs != []:
                node.set_inputs(node.inputs)
            if isinstance(node, ad.TensorInverseNode):
                new_inv_node = optimize_inverse(node)
                replace_node(node, new_inv_node)

    # if the inverse an einsum whose output is the same as input, just inverse its input
    all_nodes = find_topo_sort([node])
    with OutputInjectedMode(all_nodes):
        for node in all_nodes:
            if node.inputs != []:
                node.set_inputs(node.inputs)
            if isinstance(node, ad.TensorInverseNode) and isinstance(
                    node.inputs[0], ad.EinsumNode):
                einsum_node = node.inputs[0]
                in_subs, out_subs, _ = _parse_einsum_input(
                    (einsum_node.einsum_subscripts, *einsum_node.inputs))
                in_subs_list = in_subs.split(',')
                if len(in_subs_list) == 1 and in_subs_list[0] == out_subs:
                    replace_node(node, ad.tensorinv(einsum_node.inputs[0]))

    return node
