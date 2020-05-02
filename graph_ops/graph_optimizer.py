# This file contains the basic operations that can fuse/prune the graph.
import autodiff as ad
import logging
import copy

from graph_ops.union_find import UFBase
from numpy.core.einsumfunc import _parse_einsum_input
from utils import find_topo_sort, IntGetter, CharacterGetter, PseudoNode
from utils import get_all_einsum_descendants

FORMAT = '[%(asctime)-15s %(filename)s:%(lineno)s] %(message)s'

logger = logging.getLogger('optimizer')
logging.basicConfig(format=FORMAT)
logger.setLevel(logging.DEBUG)


### Assign each UF group a character val.
class UF(UFBase):
    def __init__(self, literal_names):
        super().__init__(literal_names)
        self.cg = CharacterGetter()

    def assign(self):
        """
            Get all parent and assign a character for each group.
            Should be only called once.
        """
        assert len(self.roots) == 0
        dsets = {}
        for value in self.parent_map.keys():
            rootval = self.root(value)
            if rootval in dsets:
                dsets[rootval].add(value)
            else:
                dsets[rootval] = {value}
        # We want to fix the character assignment order.
        # The hash is dependent on the output dim index and connection index.
        def sort_hash(pair):
            k, dset = pair
            # val[0] is the name, val[1] is the shape index.
            hash_strings = [f'{val[0]}-{val[1]}' for val in dset]
            return '+'.join(sorted(hash_strings))

        for k, v in sorted(list(dsets.items()), key=sort_hash):
            self.roots[k] = self.cg.getchar()


### Assign each UF parent a int value for group.
class UFNodes(UFBase):
    def __init__(self, nodes):
        super().__init__(nodes)
        self.ig = IntGetter()

    def assign(self):
        """
            Get all parent and assign a int for each group.
            Should be only called once.
        """
        assert len(self.roots) == 0
        for node in self.parent_map.keys():
            rootnode = self.root(node)
            if rootnode not in self.roots:
                self.roots[rootnode] = self.ig.getint()


def cross_einsum_connect(uf, output_node, literal_names):
    """
        Link the literal relationship for an einsum op.
        
        Args: 
            uf: union find data structure.
            output_node: An einsum node.
            literals: A list of all the literals including the output_node.
        
        Inputs of the einsum node can have duplicates.
    """
    assert (isinstance(output_node, ad.EinsumNode))
    # for child in output_node.inputs:
    #     assert (isinstance(child, ad.EinsumNode))

    in_subs, out_subs, _ = _parse_einsum_input(
        (output_node.einsum_subscripts, *output_node.inputs))
    in_subs_list = in_subs.split(',')
    whole_str = out_subs + "".join(in_subs_list)

    record = {}

    for pos, pair in enumerate(zip(whole_str, literal_names)):
        char, litername = pair
        if char in record:
            # encode
            uf.connect(litername, record[char])
        else:
            record[char] = litername


def node_literals(einsum_node):
    """
        Get node literals.

        Args:
            einsum_node: A Pseudo Node.
    """
    return einsum_node.literals + sum(
        [x.literals for x in einsum_node.node.inputs], [])


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

    # We first treat each literal as a different character, and then union.
    # Create a map
    for k, node in enumerate(all_nodes):
        node.literals = [
            f"{node.name}-{k}-{i}" for i in range(len(node.shape))
        ]
        pnode = PseudoNode(node=node, literals=node.literals)
        pseudo_nodes.append(pnode)
        if node in input_nodes:
            pseudo_input_nodes.append(pnode)
        if node == output_node:
            pseudo_output_node = pnode

    intermediate_nodes = list(set(pseudo_nodes) - set(pseudo_input_nodes))

    einsum_pseudo_nodes = list(
        filter(lambda x: isinstance(x.node, ad.EinsumNode), pseudo_nodes))

    literal_names = sum([node.literals for node in pseudo_nodes], [])

    # For any literal that are the same, get their pos and connect.
    uf = UF(literal_names)
    for node in einsum_pseudo_nodes:
        all_literals = node_literals(node)
        cross_einsum_connect(uf, node.node, all_literals)

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


def get_leaves(nodes):
    """
        Returns all the inputs of the nodes formed as a tree. The returned node
        must not be in the tree.
        Args:
            Nodes is a list of graph nodes.
        Returns:
            a set of nodes.
        
        For further illustration:
            A
           / \
          B  I3
         / \
        I1  I2
        
        I1, I2, I3 are the returned nodes. inputs is {A,B} 
    """
    all_inputs = set()
    for n in nodes:
        for n_i in n.inputs:
            if n_i not in nodes:
                all_inputs.add(n_i)
    return all_inputs


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
