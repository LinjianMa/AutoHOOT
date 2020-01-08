import autodiff as ad
import backend as T
from graph_ops.graph_generator import generate_optimal_tree
from tests.test_utils import tree_eq
from visualizer import print_computation_graph

BACKEND_TYPES = ['numpy']


def test_einsum_gen():
    for datatype in BACKEND_TYPES:
        N = 10
        C = ad.Variable(name="C", shape=[N, N])
        I = ad.Variable(name="I", shape=[N, N, N, N])

        output = ad.einsum('pi,qj,ijkl,rk,sl->pqrs', C, C, I, C, C)
        new_output = generate_optimal_tree(output)
        assert tree_eq(output, new_output, [C, I])
        assert len(new_output.inputs) == 2


def test_einsum_gen_custom():
    for datatype in BACKEND_TYPES:
        a = ad.Variable(name="a", shape=[2, 2])
        b = ad.Variable(name="b", shape=[2, 5])
        c = ad.Variable(name="c", shape=[5, 2])

        output = ad.einsum('ij,jk,kl->il', a, b, c)
        new_output = generate_optimal_tree(output,
                                           path=[
                                               'einsum_path',
                                               (1, 2),
                                               (0, 1),
                                           ])
        assert tree_eq(output, new_output, [a, b, c])


def test_einsum_gen_custom_3operands():
    for datatype in BACKEND_TYPES:
        a = ad.Variable(name="a", shape=[2, 2])
        b = ad.Variable(name="b", shape=[2, 5])
        c = ad.Variable(name="c", shape=[5, 2])
        d = ad.Variable(name="d", shape=[2, 2])

        output = ad.einsum('ij,jk,kl,lm->im', a, b, c, d)
        new_output = generate_optimal_tree(output,
                                           path=[
                                               'einsum_path',
                                               (1, 2),
                                               (0, 1, 2),
                                           ])
        print_computation_graph([new_output])
        assert False
        assert tree_eq(output, new_output, [a, b, c, d])
