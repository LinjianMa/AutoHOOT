import autodiff as ad
import backend as T
from graph_ops.graph_dedup import dedup
from tests.test_utils import tree_eq, gen_dict
from visualizer import print_computation_graph

BACKEND_TYPES = ['numpy', 'ctf']
BACKEND_TYPES = ['numpy']


def test_dedup():
    """
    Dedup the tree.
    """

    for datatype in BACKEND_TYPES:
        T.set_backend(datatype)

        a = ad.Variable(name="a", shape=[2, 2])
        b = ad.Variable(name="b", shape=[2, 2])

        c = a + b
        d = a + b
        z = c + d

        dedup(z)
        # Assert object level equivalence.
        assert z.inputs[0] == z.inputs[1]
