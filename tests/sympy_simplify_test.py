import autodiff as ad
from tests.test_utils import tree_eq
from utils import sympy_simplify


def test_add():

    A = ad.Variable(name="A", shape=[3])
    y = A + A

    inputs = [A]
    assert isinstance(sympy_simplify(y, inputs), ad.MulByConstNode)


def test_minus():

    A = ad.Variable(name="A", shape=[3])
    B = ad.Variable(name="B", shape=[3])
    y = A - (A - B)

    inputs = [A, B]
    assert sympy_simplify(y, inputs).name == 'B'
