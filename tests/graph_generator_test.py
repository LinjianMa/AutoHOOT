import autodiff as ad
import backend as T
from graph_ops.graph_generator import generate_optimal_tree

BACKEND_TYPES = ['numpy']


def test_einsum_gen():
    for datatype in BACKEND_TYPES:
        a1 = ad.Variable(name="a1", shape=[3, 2])
        a2 = ad.Variable(name="a2", shape=[2, 3])

        x = ad.einsum('ik,kj->ij', a1, a2)
        generate_optimal_tree(x)
