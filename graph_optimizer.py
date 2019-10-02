# This file contains the basic operations that can fuse/prune the graph.
import autodiff as ad
import logging
import copy
from numpy.core.einsumfunc import _parse_einsum_input
from utils import find_topo_sort

FORMAT = '[%(asctime)-15s %(filename)s:%(lineno)s] %(message)s'

logger = logging.getLogger('optimizer')
logging.basicConfig(format=FORMAT)
logger.setLevel(logging.DEBUG)


class CharacterGetter():
    """ Return a character and increment"""
    def __init__(self):
        self.char = 'a'

    def getchar(self):
        """
            Returns a single character. Increment after return.
        """
        previous_char = self.char
        if self.char == 'z':
            logging.info('Run out of characters.')
            raise NotImplementedError
        self.char = chr(ord(self.char) + 1)
        return previous_char


### Assign each UF parent a character val.
class UF():
    def __init__(self, literal_names):
        self.parent_map = {}
        self.cg = CharacterGetter()
        self.roots = {}
        for name in literal_names:
            self.parent_map[name] = name

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

    def root(self, n1):
        """
            Returns the root of the given node.
        """
        n = n1
        while self.parent_map[n] != n:
            n = self.parent_map[n]
        return n

    def connect(self, n1, n2):
        """
            Union two nodes.
        """
        rootn1 = self.root(n1)
        rootn2 = self.root(n2)
        if rootn1 == rootn2:
            # Already connected.
            return
        self.parent_map[rootn1] = rootn2

    # Must be called after assign
    def rootval(self, n1):
        """
            Returns the assigned character of the given node's root.
        """
        return self.roots[self.root(n1)]


def cross_einsum_connect(uf, output_node):
    """
        Link the literal relationship for an einsum op.
    """
    assert (isinstance(output_node, ad.EinsumNode))
    # for child in output_node.inputs:
    #     assert (isinstance(child, ad.EinsumNode))

    in_subs, out_subs, _ = _parse_einsum_input(
        (output_node.subscripts, *output_node.inputs))
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
            output_node and input_nodes
    TODO(yejiayu): work on extend to multiple dimensions.
    """
    # First assume everything einsum.
    logger.info('Start fusing einsum')
    g = CharacterGetter()

    # Making this automatic.
    # Assume output_node is einsum and their children are einsum of any number
    # of input nodes
    assert (isinstance(output_node, ad.EinsumNode))

    # Assume the graph is independent of others, while all are einsums.
    # Assume input are variables.
    # for node in input_nodes:
    #     assert (isinstance(node, ad.VariableNode))

    # TODO: Get all the einsum nodes in the computation graph.
    # Note that the order doesn't matter!
    all_nodes = find_topo_sort([output_node])
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

    return output_node, input_nodes
