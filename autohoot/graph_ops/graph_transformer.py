# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import itertools
import numpy as np
from autohoot import autodiff as ad

from autohoot.graph_ops.graph_dedup import dedup, declone, collapse_symmetric_expr
from autohoot.graph_ops.optimal_tree import generate_optimal_tree
from autohoot.graph_ops.graph_inv_optimizer import optimize_inverse, prune_inv_node
from autohoot.graph_ops.graph_utils import copy_tree, get_leaves
from autohoot.graph_ops.graph_pruning import prune_identity_nodes, prune_orthonormal_matmuls, prune_scalar_nodes

from autohoot.einsum_graph.graph_structure import UF, DimInfo
from autohoot.einsum_graph.expr_generator import rewrite_einsum_expr
from autohoot.einsum_graph.graph_generator import cross_einsum_connect

from autohoot.utils import find_topo_sort, OutputInjectedMode, find_topo_sort_p, OutputInjectedModeP, PseudoNode
from autohoot.utils import replace_node, sympy_simplify
from autohoot.utils import get_all_einsum_descendants

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
    all_pnodes = find_topo_sort_p([PseudoNode(output_node)])
    # Inject outputs relationship.
    with OutputInjectedModeP(all_pnodes):
        for pn in all_pnodes:
            n = pn.node
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
    assert isinstance(binary_op_node, ad.DistributiveNode)
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
    return type(binary_op_node)(AC, BC)


def distribute_tree(output):
    """ Distribute a tree of einsum and add nodes.

    NOTE: the output node should be a linearized node.
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
    def get_first_binary_op(pnodes):
        for pnode in pnodes:
            node = pnode.node
            if isinstance(node,
                          ad.DistributiveNode) and len(node.outputs) >= 1:
                has_einsum_nodes = all(
                    [isinstance(x, ad.EinsumNode) for x in node.outputs])
                if has_einsum_nodes:
                    return node
        return None

    while 1:
        all_pnodes = find_topo_sort_p([PseudoNode(output)])
        with OutputInjectedModeP(all_pnodes):
            first_binary_op = get_first_binary_op(all_pnodes)
            if first_binary_op is None:
                break
            for einsum_node in first_binary_op.outputs:
                if isinstance(einsum_node, ad.DistributiveNode):
                    continue
                assert isinstance(einsum_node, ad.EinsumNode)
                new_node = _distribute(first_binary_op, einsum_node)
                replace_node(PseudoNode(einsum_node), new_node)
                if einsum_node == output:
                    output = new_node
    # This is need for source generation.
    output.set_inputs(output.inputs)
    return output


def distribute_graph_w_linearize(output):
    linearize(output)
    output = distribute_tree(output)
    output = declone(output)
    return output


def find_sub_einsumtree(output_node_p):
    # TMP Pseudo Mode.
    """
    Finds all the subtrees from the given graph definition.
    There can be overlap of different subtrees.
    Arguments:
        output_node_p: the root of the tree, must be PseudoNode.
        input_nodes: leaf of the tree
    Returns:
        Return many einsum trees of the form 
        [[Pseudo root node, leaf nodes], ... ]
    """
    trees = []
    output_node = output_node_p.node
    if isinstance(output_node, ad.EinsumNode):
        tree_nodes = get_all_einsum_descendants(output_node)
        leaves = get_leaves(tree_nodes)
        for leaf in leaves:
            new_trees = find_sub_einsumtree(PseudoNode(leaf))
            trees += new_trees
        trees.append([output_node_p, leaves])
        return trees
    else:
        for i_node in output_node.inputs:
            new_trees = find_sub_einsumtree(PseudoNode(i_node))
            trees += new_trees
        return trees


def node_dims_info(einsum_node):
    """
        Get node dimensions information.

        Args:
            einsum_node: A Pseudo Node.
    """
    return einsum_node.dims_info + sum(
        [x.dims_info for x in einsum_node.node.inputs], [])


def fuse_einsums(output_node, input_nodes):
    """
    Find and fuse einsums.
        Parameters:
            Each node must have attribute inputs, which makes it a sparse graph
            representation.
        Returns:
            A graph with fused intermediate einsum nodes. Represented by
            output_node.
    Note: inputs of a node can have same node. But one node can't go to two 
    output nodes
    """
    # First assume everything einsum.
    logger.info('Start fusing einsum')

    # Making this automatic.
    # Assume output_node is einsum and their children are einsum of any number
    # of input nodes
    assert (isinstance(output_node, ad.EinsumNode))

    pseudo_nodes = []

    # # Get all the einsum nodes except the input nodes in the computation graph.
    # # Note that the order doesn't matter!
    all_nodes = find_topo_sort([output_node], input_nodes)

    pseudo_input_nodes = []
    pseudo_output_node = None

    # We first represennt each dim as a different character, and then union.
    # Create a map
    for k, node in enumerate(all_nodes):
        node.dims_info = [
            DimInfo(node=node, dim_index=i, node_index=k)
            for i in range(len(node.shape))
        ]
        pnode = PseudoNode(node=node, dims_info=node.dims_info)
        pseudo_nodes.append(pnode)
        if node in input_nodes:
            pseudo_input_nodes.append(pnode)
        if node == output_node:
            pseudo_output_node = pnode

    intermediate_nodes = list(set(pseudo_nodes) - set(pseudo_input_nodes))

    einsum_pseudo_nodes = list(
        filter(lambda x: isinstance(x.node, ad.EinsumNode),
               intermediate_nodes))

    all_dims_info = sum([node.dims_info for node in pseudo_nodes], [])

    # For any two dims with the same literal, get their pos and connect.
    uf = UF(all_dims_info)
    for node in einsum_pseudo_nodes:
        all_dims_info = node_dims_info(node)
        cross_einsum_connect(uf, node.node, all_dims_info)

    uf.assign()
    # Assign literals
    for node in pseudo_nodes:
        node.generate_subscript(uf)

    new_input_subs = [node.subscript for node in pseudo_input_nodes]
    new_subscripts = ",".join(
        new_input_subs) + "->" + pseudo_output_node.subscript
    logger.info(f"Generated new subscript: {new_subscripts}")
    ##########################################
    output_node = ad.einsum(new_subscripts,
                            *[node.node for node in pseudo_input_nodes])

    return output_node


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
    ret_node = PseudoNode(node)
    with OutputInjectedMode(all_nodes):
        trees = find_sub_einsumtree(ret_node)
        for tree in trees:
            out_node_p, in_nodes = tree
            new_z = fuse_einsums(out_node_p.node, in_nodes)
            prune_identity_nodes(new_z)
            new_z = generate_optimal_tree(new_z)
            replace_node(out_node_p, new_z)

    node = declone(ret_node.node)
    all_nodes = find_topo_sort([node])
    for node in all_nodes:
        if isinstance(node, ad.EinsumNode):
            rewrite_einsum_expr(node)

    for node in find_topo_sort([node]):
        if node.inputs != []:
            node.set_inputs(node.inputs)

    dedup(node)
    return node


def simplify(output_node):
    """Simplify a graph with a single output node.
    The simplified form will distribute selected operations
    (+), and fuse all connected einsums.

    Args:
        node: The output node.
    Returns:
        node: The newly generated node.
    """
    def fuse_all_einsums(node):
        linearize(node)
        ret_node = PseudoNode(node)
        all_pnodes = find_topo_sort_p([ret_node])
        with OutputInjectedModeP(all_pnodes):
            trees = find_sub_einsumtree(ret_node)
            for tree in trees:
                out_node_p, in_nodes = tree
                new_z = fuse_einsums(out_node_p.node, in_nodes)
                prune_identity_nodes(new_z)
                replace_node(out_node_p, new_z)

        node = declone(ret_node.node)
        return node

    output_node = distribute_graph_w_linearize(output_node)
    output_node = fuse_all_einsums(output_node)

    output_pnode = PseudoNode(output_node)
    all_pnodes = find_topo_sort_p([output_pnode])
    # optimize inverse
    with OutputInjectedModeP(all_pnodes):
        for pnode in all_pnodes:
            node = pnode.node
            if isinstance(node, ad.EinsumNode):
                # To make sure the same einsum nodes have the same same,
                # so that we can collapse the add node.
                rewrite_einsum_expr(node)
            if node.inputs != []:
                node.set_inputs(node.inputs)
            if isinstance(node, ad.TensorInverseNode):
                new_inv_node = optimize_inverse(node)
                replace_node(pnode, new_inv_node)

    # fuse again
    output_node = output_pnode.node
    output_node = fuse_all_einsums(output_node)

    # prune the orthonormal matmuls
    all_pnodes = find_topo_sort_p([output_pnode])
    with OutputInjectedModeP(all_pnodes):
        for pnode in all_pnodes:
            node = pnode.node
            if node.inputs != []:
                node.set_inputs(node.inputs)
            if isinstance(node, ad.EinsumNode):
                new_node = prune_orthonormal_matmuls(node)
                replace_node(pnode, new_node)

    # prune inverse nodes
    output_pnode = PseudoNode(output_node)
    all_pnodes = find_topo_sort_p([output_pnode])
    with OutputInjectedModeP(all_pnodes):
        for pnode in all_pnodes:
            node = pnode.node
            if node.inputs != []:
                node.set_inputs(node.inputs)
            if isinstance(node, ad.EinsumNode):
                new_node = prune_inv_node(node)
                replace_node(pnode, new_node)

    # prune the scalar nodes and remove unnecessary identity nodes
    all_pnodes = find_topo_sort_p([output_pnode])
    with OutputInjectedModeP(all_pnodes):
        for pnode in all_pnodes:
            node = pnode.node
            if node.inputs != []:
                node.set_inputs(node.inputs)
            if isinstance(node, ad.EinsumNode):
                prune_identity_nodes(node)
                new_node = prune_scalar_nodes(node)
                replace_node(pnode, new_node)

    # collapse symmetric expressions
    all_pnodes = find_topo_sort_p([output_pnode])
    for i in range(len(all_pnodes)):
        for j in range(i):
            collapse_symmetric_expr(all_pnodes[i].node, all_pnodes[j].node)

    sympy_input_types = (ad.DistributiveNode, ad.ScalarNode, ad.MulNode)
    #sympy_simplify the distributed nodes
    if isinstance(output_node, ad.DistributiveNode):
        sympy_inputs = []
        all_nodes = find_topo_sort([output_node])
        for node in all_nodes:
            if isinstance(node, ad.EinsumNode):
                # To make sure the same einsum nodes have the same name,
                # so that they can be reduced by sympy.
                rewrite_einsum_expr(node)
            if node.inputs != []:
                node.set_inputs(node.inputs)
            if not isinstance(node, sympy_input_types):
                sympy_inputs.append(node)
        output_node = sympy_simplify(output_node, sympy_inputs)

    return output_node
