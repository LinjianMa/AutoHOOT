import autodiff as ad
from graph_ops.graph_als_optimizer import generate_sequential_optiaml_tree
from utils import find_topo_sort
from tests.test_utils import tree_eq
from visualizer import print_computation_graph


def test_dimension_tree():
    A = ad.Variable(name="A", shape=[2, 2])
    B = ad.Variable(name="B", shape=[2, 2])
    C = ad.Variable(name="C", shape=[2, 2])
    D = ad.Variable(name="D", shape=[2, 2])
    X = ad.Variable(name="X", shape=[2, 2, 2, 2])

    einsum_node_A = ad.einsum("abcd,bm,cm,dm->am", X, B, C, D)
    einsum_node_B = ad.einsum("abcd,am,cm,dm->bm", X, A, C, D)
    einsum_node_C = ad.einsum("abcd,am,bm,dm->cm", X, A, B, D)
    einsum_node_D = ad.einsum("abcd,am,bm,cm->dm", X, A, B, C)

    dt = generate_sequential_optiaml_tree({
        einsum_node_A: A,
        einsum_node_B: B,
        einsum_node_C: C
    })

    assert tree_eq(dt[0], einsum_node_A, [A, B, C, D, X])
    assert tree_eq(dt[1], einsum_node_B, [A, B, C, D, X])
    assert tree_eq(dt[2], einsum_node_C, [A, B, C, D, X])
