import autodiff as ad
from graph_ops.graph_dedup import dedup, declone, transpose_equivalent
from tests.test_utils import tree_eq, gen_dict
from visualizer import print_computation_graph


def test_dedup():
    """
    Dedup the tree.
    """

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

    a = ad.Variable(name="a", shape=[2, 2])
    b = ad.Variable(name="b", shape=[2, 2])

    a2 = a.clone()
    a3 = a2.clone()
    c = a3 + b
    c = declone(c)
    assert c.inputs == [a, b]


def test_transpose_equiv():
    """
    Test if several pattern of transpose equivalence.
    """
    a = ad.Variable(name="a", shape=[2, 2, 2])
    b = ad.Variable(name="b", shape=[2, 2])
    assert not transpose_equivalent(ad.einsum('iii->', a), ad.einsum(
        'ii->', b))
    assert transpose_equivalent(ad.einsum('iii->', a), ad.einsum('iii->', a))
    assert not transpose_equivalent(ad.einsum('adb,cb->adc', a, b),
                                    ad.einsum('dab,bc->dac', a, b))
    assert not transpose_equivalent(ad.einsum('acb,bd->adc', a, b),
                                    ad.einsum('dab,bc->dac', a, b))
    assert transpose_equivalent(ad.einsum('adb,bc->adc', a, b),
                                ad.einsum('dab,bc->dac', a, b))
