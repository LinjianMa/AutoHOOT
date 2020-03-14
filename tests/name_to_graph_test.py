import autodiff as ad
from tests.test_utils import tree_eq
from name_parser import AutodiffParser


def test_add():

    A = ad.Variable(name="A", shape=[3])
    B = ad.Variable(name="B", shape=[3])
    y = A + B

    assert AutodiffParser.parse(y.name, [A, B]).name == y.name


def test_minus():

    A = ad.Variable(name="A", shape=[3])
    B = ad.Variable(name="B", shape=[3])
    y = A - B

    assert AutodiffParser.parse(y.name, [A, B]).name == y.name


def test_scalar_mul():

    A = ad.Variable(name="A", shape=[3])
    y = 3.0 * A

    assert AutodiffParser.parse(y.name, [A]).name == y.name


def test_add_3():

    A = ad.Variable(name="A", shape=[3])
    B = ad.Variable(name="B", shape=[3])
    y = A + B + B

    assert AutodiffParser.parse(y.name, [A, B]).name == y.name


def test_einsum():

    A = ad.Variable(name="A", shape=[3, 2])
    B = ad.Variable(name="B", shape=[2, 3])
    y = ad.einsum('ik,kj->ij', A, B)

    assert AutodiffParser.parse(y.name, [A, B]).name == y.name


def test_einsum_mul():

    A = ad.Variable(name="A", shape=[3, 2])
    B = ad.Variable(name="B", shape=[2, 3])
    y = ad.einsum('ik,kj->ij', A, B)
    z = 2 * y

    assert AutodiffParser.parse(z.name, [A, B]).name == z.name


def test_tensorinv():

    A = ad.Variable(name="A", shape=[3, 3])
    y = ad.tensorinv(A)

    assert AutodiffParser.parse(y.name, [A]).name == y.name
