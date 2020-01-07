import autodiff as ad
from graph_ops.graph_als_optimizer import split_einsum, generate_sequential_optiaml_tree
from utils import find_topo_sort
from graph_ops.utils import einsum_equal
from visualizer import print_computation_graph


def test_split_einsum():

    A = ad.Variable(name="A", shape=[2, 2])
    B = ad.Variable(name="B", shape=[2, 2])
    C = ad.Variable(name="C", shape=[2, 2])
    D = ad.Variable(name="D", shape=[2, 2])
    E = ad.Variable(name="E", shape=[2, 2])

    einsum_node = ad.einsum("ab,bc,cd,de,ef->af", A, B, C, D, E)
    split_input_nodes = [A, B]
    first_einsum, second_einsum = split_einsum(einsum_node, split_input_nodes)

    first_einsum_expected = ad.einsum("cd,de,ef->fc", C, D, E)
    assert einsum_equal(first_einsum, first_einsum_expected)

    second_einsum_expected = ad.einsum("ab,bc,fc->af", A, B, first_einsum)
    assert einsum_equal(second_einsum, second_einsum_expected)


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

    intermediate_nodes = list(
        filter(lambda x: isinstance(x, ad.OpNode),
               find_topo_sort(dt, [A, B, C, D])))

    dt_output_expected_0 = ad.einsum("abcd,dm->mabc", X, D)
    assert einsum_equal(dt_output_expected_0, intermediate_nodes[0])

    dt_output_expected_1 = ad.einsum("cm,mabc->mab", C, intermediate_nodes[0])
    assert einsum_equal(dt_output_expected_1, intermediate_nodes[1])

    dt_output_expected_2 = ad.einsum("bm,mab->am", B, intermediate_nodes[1])
    assert einsum_equal(dt_output_expected_2, intermediate_nodes[2])

    dt_output_expected_3 = ad.einsum("am,mab->bm", A, intermediate_nodes[1])
    assert einsum_equal(dt_output_expected_3, intermediate_nodes[3])

    dt_output_expected_4 = ad.einsum("am,bm,mabc->cm", A, B,
                                     intermediate_nodes[0])
    assert einsum_equal(dt_output_expected_4, intermediate_nodes[4])
