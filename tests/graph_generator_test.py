import autodiff as ad
import backend as T
from graph_ops.graph_generator import generate_optimal_tree, split_einsum
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
        assert tree_eq(output, new_output, [a, b, c, d])


def test_split_einsum():

    A = ad.Variable(name="A", shape=[2, 2])
    B = ad.Variable(name="B", shape=[2, 2])
    C = ad.Variable(name="C", shape=[2, 2])
    D = ad.Variable(name="D", shape=[2, 2])
    E = ad.Variable(name="E", shape=[2, 2])

    einsum_node = ad.einsum("ab,bc,cd,de,ef->af", A, B, C, D, E)
    split_input_nodes = [A, B]
    new_einsum = split_einsum(einsum_node, split_input_nodes)
    assert len(new_einsum.inputs) == 3  # A, B, einsum(C, D, E)
    assert tree_eq(new_einsum, einsum_node, [A, B, C, D, E])
