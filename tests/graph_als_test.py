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

    dt = generate_sequential_optiaml_tree(
        {
            einsum_node_A: A,
            einsum_node_B: B,
            einsum_node_C: C,
            einsum_node_D: D
        })

    assert tree_eq(dt[0], einsum_node_A, [A, B, C, D, X])
    assert tree_eq(dt[1], einsum_node_B, [A, B, C, D, X])
    assert tree_eq(dt[2], einsum_node_C, [A, B, C, D, X])
    assert tree_eq(dt[3], einsum_node_D, [A, B, C, D, X])


def test_dimension_tree_w_identity():
    A = ad.Variable(name="A", shape=[2, 2])
    B = ad.identity(2)
    C = ad.Variable(name="C", shape=[2, 2])
    X = ad.Variable(name="X", shape=[2, 2, 2])

    einsum_node_A = ad.einsum("abc,bm,cm->am", X, B, C)
    einsum_node_B = ad.einsum("abc,am,cm->bm", X, A, C)
    einsum_node_C = ad.einsum("abc,am,bm->cm", X, A, B)

    dt = generate_sequential_optiaml_tree(
        {
            einsum_node_A: A,
            einsum_node_B: B
        },
        first_contract_node=C)

    assert tree_eq(dt[0], einsum_node_A, [A, C, X])
    assert tree_eq(dt[1], einsum_node_B, [A, C, X])


def test_simple_dmrg_tree():
    A1 = ad.Variable(name="A1", shape=[3, 2])
    A2 = ad.Variable(name="A2", shape=[3, 3, 2])
    A3 = ad.Variable(name="A3", shape=[3, 2])

    X1 = ad.Variable(name="X1", shape=[3, 2, 2])
    X2 = ad.Variable(name="X2", shape=[3, 3, 2, 2])
    X3 = ad.Variable(name="X3", shape=[3, 2, 2])
    """
        The network and indices positions are as follows:

        A1 - f - A2 - g - A3
        |        |        |
        c        d        e
        |        |        |
        X1 - a - X2 - b - X3
        |        |        |
        h        i        j
        |        |        |
        A1 - k - A2 - l - A3

    """
    einsum_node_A1 = ad.einsum("ach,abdi,bej,fgd,kli,ge,lj->fckh", X1, X2, X3,
                               A2, A2, A3, A3)
    einsum_node_A2 = ad.einsum("ach,abdi,bej,fc,kh,ge,lj->fgdkli", X1, X2, X3,
                               A1, A1, A3, A3)

    dt = generate_sequential_optiaml_tree(
        {
            einsum_node_A1: A1,
            einsum_node_A2: A2,
        }, first_contract_node=A3)

    assert tree_eq(dt[0], einsum_node_A1, [X1, X2, X3, A1, A1, A2, A2, A3, A3])
    assert tree_eq(dt[1], einsum_node_A2, [X1, X2, X3, A1, A1, A2, A2, A3, A3])

    # In the correct contraction path, only X3 should be contracted with A3,
    # all other X nodes should be contracted later.
    einsum_inputs = list(
        filter(lambda node: isinstance(node, ad.EinsumNode),
               find_topo_sort(dt)))
    assert sorted(einsum_inputs[0].inputs,
                  key=lambda node: node.name) == sorted(
                      [A3, A3, X3], key=lambda node: node.name)
