# This file contains the basic operations that can fuse/prune the graph.
import autodiff as ad
import logging
import copy

from graph_ops.union_find import UFBase
from numpy.core.einsumfunc import _parse_einsum_input
from utils import find_topo_sort, IntGetter, CharacterGetter
from utils import get_leaves, get_all_einsum_descendants

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
        for name in self.parent_map.keys():
            rootname = self.root(name)
            if rootname not in self.roots:
                self.roots[rootname] = self.cg.getchar()


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


def cross_einsum_connect(uf, output_node):
    """
        Link the literal relationship for an einsum op.
    """
    assert (isinstance(output_node, ad.EinsumNode))
    # for child in output_node.inputs:
    #     assert (isinstance(child, ad.EinsumNode))

    in_subs, out_subs, _ = _parse_einsum_input(
        (output_node.einsum_subscripts, *output_node.inputs))
    in_subs_list = in_subs.split(',')
    whole_str = out_subs + "".join(in_subs_list)

    record = {}
    literal_names = copy.deepcopy(output_node.literals)
    for node in output_node.inputs:
        literal_names += node.literals

    for pos, pair in enumerate(zip(whole_str, literal_names)):
        char, litername = pair
        if char in record:
            # encode
            uf.connect(litername, record[char])
        else:
            record[char] = litername


def fuse_einsums(output_node, input_nodes):
    """
    Find and fuse einsums.
        Parameters:
            Each node must have attribute inputs, which makes it a sparse graph
            representation.
        Returns:
            A graph with fused intermediate einsum nodes. Represented by
            output_node.
    """
    # First assume everything einsum.
    logger.info('Start fusing einsum')

    # Making this automatic.
    # Assume output_node is einsum and their children are einsum of any number
    # of input nodes
    assert (isinstance(output_node, ad.EinsumNode))

    # Assume the graph is independent of others, while all are einsums.
    # Assume input are variables.
    for node in input_nodes:
        assert (not isinstance(node, ad.EinsumNode))

    # TODO: Get all the einsum nodes in the computation graph.
    # Note that the order doesn't matter!
    all_nodes = find_topo_sort([output_node], input_nodes)
    einsum_nodes = list(
        filter(lambda x: isinstance(x, ad.EinsumNode), all_nodes))

    # We first treat each literal as a different character, and then union.
    # Create a map
    for node in all_nodes:
        node.literals = [node.name + str(i) for i in range(len(node.shape))]

    literal_names = []
    for node in all_nodes:
        literal_names += node.literals

    # For any literal that are the same, get their pos and connect.
    uf = UF(literal_names)
    for node in einsum_nodes:
        cross_einsum_connect(uf, node)

    uf.assign()
    # Assign literals
    for node in [output_node, *input_nodes]:
        node.subscripts = "".join(
            [uf.rootval(literal_name) for literal_name in node.literals])

    new_input_subs = [node.subscripts for node in input_nodes]
    new_subscripts = ",".join(new_input_subs) + "->" + output_node.subscripts
    logger.info(f"Generated new subscript: {new_subscripts}")
    ##########################################
    output_node = ad.einsum(new_subscripts, *input_nodes)

    return output_node


def find_sub_einsumtree(output_node):
    """
    Finds all the subtrees from the given graph definition.

    There can be overlap of different subtrees.

    Arguments:
        output_node: the root of the tree
        input_nodes: leaf of the tree
    Returns:
        Return many einsum trees.
    """
    trees = []
    if isinstance(output_node, ad.EinsumNode):
        tree_nodes = get_all_einsum_descendants(output_node)
        leaves = get_leaves(tree_nodes)
        for leaf in leaves:
            new_trees = find_sub_einsumtree(leaf)
            trees += new_trees
        trees.append([output_node, leaves])
        return trees
    else:
        for i_node in output_node.inputs:
            new_trees = find_sub_einsumtree(i_node)
            trees += new_trees
        return trees
