import autodiff as ad
import backend as T
from utils import get_all_inputs
from graph_ops.graph_generator import generate_optimal_tree, split_einsum, get_common_ancestor
from tests.test_utils import tree_eq
from visualizer import print_computation_graph

BACKEND_TYPES = ['numpy', 'ctf', 'tensorflow']


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


def test_get_common_ancestor_simple():

    A = ad.Variable(name="A", shape=[3, 2])

    X1 = ad.Variable(name="X1", shape=[3, 4, 4])
    X2 = ad.Variable(name="X2", shape=[3, 2, 2])
    """
        The network and indices positions are as follows:

             g - A
                 |
        d        e
        |        |
        X1 - b - X2
        |        |
        i        j
    """
    einsum_node = ad.einsum('ge,bdi,bej->gdij', A, X1, X2)
    opt_einsum = generate_optimal_tree(einsum_node)
    sub_einsum = get_common_ancestor(opt_einsum, einsum_node.inputs, A)

    assert sorted(get_all_inputs(sub_einsum),
                  key=lambda node: node.name) == sorted(
                      [A, X2], key=lambda node: node.name)


def test_get_common_ancestor():

    A = ad.Variable(name="A", shape=[3, 2])

    X1 = ad.Variable(name="X1", shape=[3, 2, 2])
    X2 = ad.Variable(name="X2", shape=[3, 3, 2, 2])
    X3 = ad.Variable(name="X3", shape=[3, 2, 2])
    """
        The network and indices positions are as follows:

                      g - A
                          |
        c        d        e
        |        |        |
        X1 - a - X2 - b - X3
        |        |        |
        h        i        j
                          |
                      l - A

    """
    einsum_node = ad.einsum('lj,ge,bej,abdi,ach->cdhigl', A, A, X3, X2, X1)
    opt_einsum = generate_optimal_tree(einsum_node)
    sub_einsum = get_common_ancestor(opt_einsum, einsum_node.inputs, A)

    assert sorted(get_all_inputs(sub_einsum),
                  key=lambda node: node.name) == sorted(
                      [A, A, X3], key=lambda node: node.name)


def test_get_common_ancestor_w_inv():

    A = ad.Variable(name="A", shape=[3, 2])
    X = ad.Variable(name="X", shape=[3, 3, 3])
    inv = ad.tensorinv(ad.einsum("ab,ac->bc", A, A), ind=1)
    einsum_node = ad.einsum('abc,ad,de->bce', X, A, inv)
    opt_einsum = generate_optimal_tree(einsum_node)
    sub_einsum = get_common_ancestor(opt_einsum, einsum_node.inputs, A)

    # sub_einsum should be ad.einsum('ad,abc->dbc',A,X), and shouldn't include the inv node.
    assert sorted(get_all_inputs(sub_einsum),
                  key=lambda node: node.name) == sorted(
                      [A, X], key=lambda node: node.name)


def test_split_einsum_dup():
    for datatype in BACKEND_TYPES:

        A = ad.Variable(name="A", shape=[2, 2])
        B = ad.Variable(name="B", shape=[2, 2])

        einsum_node = ad.einsum("ab,bc,cd->ad", A, B, B)
        split_input_nodes = [A]
        new_einsum = split_einsum(einsum_node, split_input_nodes)

        assert len(new_einsum.inputs) == 2  # A, einsum(B, B)
        assert tree_eq(new_einsum, einsum_node, [A, B])
