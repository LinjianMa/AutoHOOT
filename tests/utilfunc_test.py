import autodiff as ad
import backend as T
from utils import update_variables
from tests.test_utils import tree_eq

BACKEND_TYPES = ['numpy', 'ctf', 'tensorflow']


def test_update_variables():
    for datatype in BACKEND_TYPES:
        T.set_backend(datatype)

        a = ad.Variable(name="a", shape=[2, 2])
        b = ad.Variable(name="b", shape=[2, 2])
        c = ad.Variable(name="c", shape=[2, 2])

        d = ad.einsum("ab,bc->ac", a, b) + c

        a = ad.Variable(name="a", shape=[3, 3])
        b = ad.Variable(name="b", shape=[3, 3])
        c = ad.Variable(name="c", shape=[3, 3])

        update_variables([d], [a, b, c])
        assert (tree_eq(d, ad.einsum("ab,bc->ac", a, b) + c, [a, b, c]))
