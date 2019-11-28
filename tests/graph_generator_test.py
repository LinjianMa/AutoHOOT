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
