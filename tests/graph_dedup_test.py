import autodiff as ad
import backend as T
from graph_ops.graph_dedup import dedup, declone
from tests.test_utils import tree_eq, gen_dict
from visualizer import print_computation_graph

BACKEND_TYPES = ['numpy', 'ctf', 'tensorflow']


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


def test_dedup_many():
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
        y = c - d

        dedup(y, z)
        # Assert object level equivalence.
        assert z.inputs[0] == z.inputs[1]
        assert y.inputs[0] == y.inputs[1]


def test_declone():
    """
    Declone the tree.
    """

    for datatype in BACKEND_TYPES:
        T.set_backend(datatype)

        a = ad.Variable(name="a", shape=[2, 2])
        b = ad.Variable(name="b", shape=[2, 2])

        a2 = a.clone()
        c = a2 + b
        c = declone(c)
        assert c.inputs == [a, b]


def test_declone_long():
    """
    Declone the tree.
    """

    for datatype in BACKEND_TYPES:
        T.set_backend(datatype)

        a = ad.Variable(name="a", shape=[2, 2])
        b = ad.Variable(name="b", shape=[2, 2])

        a2 = a.clone()
        a3 = a2.clone()
        c = a3 + b
        c = declone(c)
        assert c.inputs == [a, b]
