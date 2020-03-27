import autodiff as ad
import backend as T
from graph_ops.graph_optimizer import fuse_einsums, find_sub_einsumtree, get_all_einsum_descendants
from graph_ops.graph_transformer import linearize
from utils import find_topo_sort
from utils import replace_node, OutputInjectedMode, PseudoNode
from tests.test_utils import tree_eq, gen_dict, float_eq

BACKEND_TYPES = ['numpy', 'ctf', 'tensorflow']


###############################################################################
# Helper functions for the tests
###############################################################################
def get_tree(prefix=""):
    """
        Returns input node list and outputnode
    """
    a1 = ad.Variable(name=prefix + "a1", shape=[3, 2])
    a2 = ad.Variable(name=prefix + "a2", shape=[2, 3])

    b1 = ad.Variable(name=prefix + "b1", shape=[3, 2])
    b2 = ad.Variable(name=prefix + "b2", shape=[2, 3])

    x = ad.einsum('ik,kj->ij', a1, a2)
    y = ad.einsum('jl,ls->js', b1, b2)

    z = ad.einsum('ij, js->is', x, y)
    return [a1, a2, b1, b2], z


###############################################################################


def test_find_all_einsum_descendants():
    """
        Find all einsum nodes.
    """
    a1 = ad.Variable(name="a1", shape=[3, 2])
    a2 = ad.Variable(name="a2", shape=[2, 3])

    x = ad.einsum('ik,kj->ij', a1, a2)

    assert [x] == get_all_einsum_descendants(x)

    inputs, output = get_tree()
    all_einsum_nodes = get_all_einsum_descendants(output)
    assert len(all_einsum_nodes) == 3


def test_einsum_simple_rewrite():
    """
        Rewrite the einsum expression.
    """

    for datatype in BACKEND_TYPES:
        T.set_backend(datatype)

        a1 = ad.Variable(name="a1", shape=[3, 2])
        a2 = ad.Variable(name="a2", shape=[2, 3])
        x = ad.einsum('ik,kj->ij', a1, a2)
        x_new = fuse_einsums(x, [a1, a2])
        assert tree_eq(x, x_new, [a1, a2])


def test_einsum():

    for datatype in BACKEND_TYPES:
        T.set_backend(datatype)

        input_nodes, z = get_tree()
        z_new = fuse_einsums(z, input_nodes)
        assert tree_eq(z, z_new, input_nodes)


def test_einsum_fuse_graph():
    """
        [Fuse einsum used twice]
        This case is rather subtle.
        We want to auto fuse
            A   B   C 
            |    \ /  
            |     es  
            |    /|   
            |  /  |   
            es    |   
              \   |   
                es
        Here es is einsum.
    """

    for datatype in BACKEND_TYPES:
        T.set_backend(datatype)
        a = ad.Variable(name="a", shape=[3, 3])
        b = ad.Variable(name="b", shape=[3, 2])
        c = ad.Variable(name="c", shape=[2, 3])

        BC = ad.einsum('ik, kj->ij', b, c)  # 3x3

        ABC = ad.einsum('ik, kj->ij', a, BC)  # 3x3

        out = ad.einsum('jk, ki->ji', ABC, BC)  # 3x3

        linearize(out)
        tree, = find_sub_einsumtree(PseudoNode(out))
        out, ins = tree
        new_z = fuse_einsums(out.node, ins)

        assert tree_eq(out.node, new_z, [a, b, c])


def test_einsum_fuse_w_identity():
    """
        [Fuse einsum with multiple identities]
        We want to fuse
            A   identity  identity
            |    \       /
            |     \     /
            |      \   /
            |       es
            |     /
            |    /
            |   /
            |  /
            es
        Here es is einsum.
    """

    for datatype in BACKEND_TYPES:
        T.set_backend(datatype)

        a = ad.Variable(name="a", shape=[3, 3])
        es_identity = ad.einsum('ik,kj->ij', ad.identity(3), ad.identity(3))
        out = ad.einsum('ai,ij->aj', a, es_identity)

        tree, = find_sub_einsumtree(PseudoNode(out))
        out, ins = tree
        new_out = fuse_einsums(out.node, ins)
        assert tree_eq(out.node, new_out, [a])


def test_einsum_fuse_only_identity():

    for datatype in BACKEND_TYPES:
        T.set_backend(datatype)

        es_identity = ad.einsum('ik,kj->ij', ad.identity(3), ad.identity(3))
        out = ad.einsum('ai,ij->aj', ad.identity(3), es_identity)

        tree, = find_sub_einsumtree(PseudoNode(out))
        out, ins = tree
        new_out = fuse_einsums(out.node, ins)
        assert tree_eq(out.node, new_out, [])


def test_einsum_multiuse():
    """
        Test manual fuse.
        A    B   inputs 
        |\   |
        | \  |
        |  \ |
        |   C
        |  / 
        | /
        output

        Note that here we assume A is split into 2 vars by some other operations.
    """

    for datatype in BACKEND_TYPES:
        T.set_backend(datatype)

        a = ad.Variable(name="a1", shape=[3, 2])
        a_copy = ad.Variable(name="a2", shape=[3, 2])
        b = ad.Variable(name="b", shape=[2, 3])

        c = ad.einsum('ik,kj->ij', a, b)
        output = ad.einsum('ik,ij->kj', a_copy, c)
        # New graph
        out_new = fuse_einsums(output, [a, a_copy, b])
        assert tree_eq(output, out_new, [a, a_copy, b])


def test_einsum_multiuse_auto_copy():
    """
        Test autolinearization and auto fuse.
        A    B   inputs 
        |\   |
        | \  |
        |  \ |
        |   C
        |  / 
        | /
        output

        Next: we would need to autoprune.
    """

    for datatype in BACKEND_TYPES:
        T.set_backend(datatype)

        a = ad.Variable(name="a1", shape=[3, 2])
        b = ad.Variable(name="b", shape=[2, 3])

        c = ad.einsum('ik,kj->ij', a, b)
        output = ad.einsum('ik,ij->kj', a, c)

        linearize(output)
        all_nodes = find_topo_sort([output])
        cloned_nodes = [
            tmp for tmp in all_nodes if isinstance(tmp, ad.CloneNode)
        ]

        out_new = fuse_einsums(output, [*cloned_nodes, b])
        # Test that every inputs is now fused.
        assert all([not isinstance(x, ad.EinsumNode) for x in out_new.inputs])

        assert tree_eq(output, out_new, [*cloned_nodes, b])


def test_einsum_multitier():

    for datatype in BACKEND_TYPES:
        T.set_backend(datatype)

        input_nodes1, zs1 = get_tree("set1")
        input_nodes2, zs2 = get_tree("set2")
        out1 = zs1 + zs2

        input_nodes3, zs3 = get_tree("set3")
        input_nodes4, zs4 = get_tree("set4")
        out2 = zs3 + zs4
        out = ad.einsum("ij, jk->ik", out1, out2)
        input_nodes = input_nodes1 + input_nodes2 + input_nodes3 + input_nodes4

        generated_feed_dict = gen_dict(input_nodes)

        executor = ad.Executor([out])
        z_val, = executor.run(feed_dict=generated_feed_dict)

        with OutputInjectedMode(find_topo_sort([out])):
            trees = find_sub_einsumtree(PseudoNode(out))
            for tree in trees:
                out_node, in_nodes = tree
                new_z = fuse_einsums(out_node.node, in_nodes)
                replace_node(out_node, new_z)

        executor = ad.Executor([out])
        z_new_val, = executor.run(feed_dict=generated_feed_dict)

        assert float_eq(z_val, z_new_val)


def test_einsum_subtree_clone():
    """
        [Subtree clone]
        This case is rather subtle.
        We want to auto fuse
            A   B   C   D
            |    \ /    |
            |     es    |
            |    /  \   |
            |  /      \ |
            es         es
              \       /
                  +

        Here es is einsum.
    """

    for datatype in BACKEND_TYPES:
        T.set_backend(datatype)
        a = ad.Variable(name="a", shape=[3, 3])
        b = ad.Variable(name="b", shape=[3, 2])
        c = ad.Variable(name="c", shape=[2, 3])
        d = ad.Variable(name="d", shape=[3, 3])

        BC = ad.einsum('ik, kj->ij', b, c)  # 3x3

        ABC = ad.einsum('ik, kj->ij', a, BC)  # 3x3

        BCD = ad.einsum('jk, ki->ji', BC, d)  # 3x3

        out = ABC + BCD

        input_nodes = [a, b, c, d]
        generated_feed_dict = gen_dict(input_nodes)

        executor = ad.Executor([out])
        out_val, = executor.run(feed_dict=generated_feed_dict)

        with OutputInjectedMode(find_topo_sort([out])):
            trees = find_sub_einsumtree(PseudoNode(out))
            assert len(trees) == 2
            for tree in trees:
                out_node, in_nodes = tree
                new_z = fuse_einsums(out_node.node, in_nodes)
                replace_node(out_node, new_z)

        new_out_val, = executor.run(feed_dict=generated_feed_dict)

        assert float_eq(out_val, new_out_val)
