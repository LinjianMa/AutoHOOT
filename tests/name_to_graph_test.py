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


def test_add_3():

    A = ad.Variable(name="A", shape=[3])
    B = ad.Variable(name="B", shape=[3])
    y = A + B + B

    assert AutodiffParser.parse(y.name, [A, B]).name == y.name
