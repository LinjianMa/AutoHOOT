import autodiff as ad
from graph_ops.utils import einsum_equal


def test_einsum_equal():

    a1 = ad.Variable(name="a1", shape=[3, 2])
    a2 = ad.Variable(name="a2", shape=[2, 3])

    x = ad.einsum('ik,kj->ij', a1, a2)
    y = ad.einsum('ml,sm->sl', a2, a1)

    assert einsum_equal(x, y) == True
