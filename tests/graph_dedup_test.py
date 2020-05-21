import autodiff as ad
from graph_ops.graph_dedup import dedup, declone, get_transpose_indices, remove_transposes, collapse_symmetric_expr
from tests.test_utils import tree_eq, gen_dict
from utils import find_topo_sort
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


def test_get_transpose_indices():
    a = ad.Variable(name="a", shape=[2, 2, 2])
    b = ad.Variable(name="b", shape=[2, 2])
    c = ad.Variable(name="a", shape=[2, 2, 2, 2])

    # not transposable
    assert get_transpose_indices(a, b) == None
    assert get_transpose_indices(ad.einsum("abc,cd->abd", a, b), b) == None
    assert get_transpose_indices(ad.einsum('iii->', a), ad.einsum('ii->',
                                                                  b)) == None
    assert get_transpose_indices(ad.einsum('abc,bc->a', a, b),
                                 ad.einsum('abc,bc->ab', a, b)) == None
    assert get_transpose_indices(ad.einsum('adb,cb->adc', a, b),
                                 ad.einsum('dab,bc->dac', a, b)) == None
    assert get_transpose_indices(ad.einsum('abc,bc->ab', a, b),
                                 ad.einsum('abc,bc->ac', a, b)) == None

    # same expression
    assert get_transpose_indices(ad.einsum('iii->', a), ad.einsum('iii->',
                                                                  a)) == None
    assert get_transpose_indices(ad.einsum('adb,bc->adc', a, b),
                                 ad.einsum('dab,bc->dac', a, b)) == None

    # transposable
    assert get_transpose_indices(ad.einsum('acb,bd->adc', a, b),
                                 ad.einsum('dab,bc->dac', a, b)) == [0, 2, 1]
    assert get_transpose_indices(ad.einsum('acje,ie->iacj', c, b),
                                 ad.einsum('jace,ie->iacj', c,
                                           b)) == [0, 2, 3, 1]


def test_remove_transposes():
    a = ad.Variable(name="a", shape=[2, 2, 2, 2])
    b = ad.Variable(name="b", shape=[2, 2])
    c = ad.Variable(name="b", shape=[2, 2])
    d = ad.Variable(name="b", shape=[2, 2])

    ab1 = ad.einsum("abcd,de->abce", a, b)
    ab2 = ad.einsum("abcd,de->ecba", a, b)

    abc1 = ad.einsum("abce,ce->abe", ab1, c)
    abc2 = ad.einsum("ecba,ce->eba", ab2, c)

    abcd1 = ad.einsum("abe,be->ae", abc1, d)
    abcd2 = ad.einsum("eba,be->ae", abc2, d)

    remove_transposes(find_topo_sort([abcd1, abcd2]))

    assert abcd1.name == abcd2.name


def test_remove_transposes_multiple_trans():
    a = ad.Variable(name="a", shape=[2, 2, 2, 2])

    intermediate1 = ad.einsum("abcd->dcba", a)
    intermediate2 = ad.einsum("abcd->abdc", a)

    ret1 = ad.einsum("dcba->badc", intermediate1)
    ret2 = ad.einsum("abdc->badc", intermediate2)

    remove_transposes(find_topo_sort([ret1, ret2]))

    assert ret1.name == ret2.name


def test_collapse_symmetric_expr():
    h = ad.Variable(name="h", shape=[2, 2, 2, 2], symmetry=[[0, 1], [2, 3]])
    a = ad.Variable(name="a", shape=[2, 2])

    out1 = ad.einsum("ijkl,ik->jl", h, a)
    out2 = ad.einsum("ijkl,jl->ik", h, a)

    collapse_symmetric_expr(out1, out2)

    assert out1.name == out2.name


def test_collapse_symmetric_expr_complex():
    """
    out1:
    A1 - a - A2 - b - A3
    |         |        |
    c         d        e
    |         |        |
    H1 - f - H2 - g - H3
    |         |        |
    h         i        j
    out2:
    a         b        c
    |         |        |
    H1 - d - H2 - e - H3
    |         |        |
    f         g        h
    A1 - i - A2 - j - A3
    """
    H1 = ad.Variable(name="H1", shape=[2, 2, 2], symmetry=[[0, 2]])
    H2 = ad.Variable(name="H2", shape=[2, 2, 2, 2], symmetry=[[0, 2]])
    H3 = ad.Variable(name="H3", shape=[2, 2, 2], symmetry=[[0, 1]])

    A1 = ad.Variable(name="H1", shape=[2, 2])
    A2 = ad.Variable(name="H2", shape=[2, 2, 2])
    A3 = ad.Variable(name="H3", shape=[2, 2])

    out1 = ad.einsum("ca,dab,eb,cfh,dgif,ejg->hij", A1, A2, A3, H1, H2, H3)
    out2 = ad.einsum("fi,gij,hj,adf,begd,che->abc", A1, A2, A3, H1, H2, H3)

    collapse_symmetric_expr(out1, out2)

    assert out1.name == out2.name


def test_cannot_collapse_symmetric_expr():
    h = ad.Variable(name="h", shape=[2, 2, 2, 2], symmetry=[[0, 1], [2, 3]])
    a = ad.Variable(name="a", shape=[2, 2])

    # non-einsum node
    collapse_symmetric_expr(h, a)
    assert h.name != a.name

    # different inputs
    out1 = ad.einsum("ijkl,ik->jl", h, a)
    out2 = ad.einsum("jl,ijkl->ik", a, h)
    collapse_symmetric_expr(out1, out2)
    assert out1.name != out2.name

    # different output shape
    out1 = ad.einsum("ijkl,ik->jl", h, a)
    out2 = ad.einsum("ijkl,ik->jkl", h, a)
    collapse_symmetric_expr(out1, out2)
    assert out1.name != out2.name


def test_cannot_collapse_expr():
    h = ad.Variable(name="h", shape=[2, 2, 2, 2])
    a = ad.Variable(name="a", shape=[2, 2])

    out1 = ad.einsum("ijkl,ik->jl", h, a)
    out2 = ad.einsum("ijkl,jl->ik", h, a)

    collapse_symmetric_expr(out1, out2)

    assert out1.name != out2.name
