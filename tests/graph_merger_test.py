import autodiff as ad
import numpy as np
import backend as T
from source import SourceToSource
from graph_ops.graph_optimizer import fuse_einsums, find_sub_einsumtree
from graph_ops.graph_transformer import linearize
from visualizer import print_computation_graph
from utils import find_topo_sort
from utils import replace_node, OutputInjectedMode

BACKEND_TYPES = ['numpy', 'ctf']
BACKEND_TYPES = ['numpy']


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


def test_einsum_simple_rewrite():
    """
        Rewrite the einsum expression.
    """

    for datatype in BACKEND_TYPES:
        T.set_backend(datatype)

        a1 = ad.Variable(name="a1", shape=[3, 2])
        a2 = ad.Variable(name="a2", shape=[2, 3])

        x = ad.einsum('ik,kj->ij', a1, a2)

        executor = ad.Executor([x])

        a1_val = T.tensor([[1, 2], [3, 4], [5, 6]])  # 3x2
        a2_val = T.tensor([[7, 8, 9], [10, 11, 12]])  # 2x3

        x_val, = executor.run(feed_dict={a1: a1_val, a2: a2_val})

        # New graph
        x_new, input_nodes = fuse_einsums(x, [a1, a2])
        a1_new, a2_new = input_nodes

        executor = ad.Executor([x_new])
        x_new_val, = executor.run(feed_dict={a1_new: a1_val, a2_new: a2_val})

        assert T.array_equal(x_val, x_new_val)


def test_einsum():

    for datatype in BACKEND_TYPES:
        T.set_backend(datatype)

        input_nodes, z = get_tree()
        executor = ad.Executor([z])

        input_values = [
            T.tensor([[1, 2], [3, 4], [5, 6]]),  # 3x2 a1
            T.tensor([[7, 8, 9], [10, 11, 12]]),  # 2x3 a2
            T.tensor([[1, 2], [3, 4], [5, 6]]),  # 3x2 b1
            T.tensor([[7, 8, 9], [10, 11, 12]])  # 2x3 b2
        ]
        z_val, = executor.run(feed_dict=dict(zip(input_nodes, input_values)))

        # New graph
        z_new, input_nodes = fuse_einsums(z, input_nodes)
        assert z_new.inputs == input_nodes

        executor = ad.Executor([z_new])
        z_new_val, = executor.run(
            feed_dict=dict(zip(input_nodes, input_values)))

        assert T.array_equal(z_val, z_new_val)


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

        executor = ad.Executor([output])

        a_val = T.tensor([[1, 2], [3, 4], [5, 6]])  # 3x2
        b_val = T.tensor([[7, 8, 9], [10, 11, 12]])  # 2x3

        out_val, = executor.run(feed_dict={a: a_val, a_copy: a_val, b: b_val})

        # New graph
        out_new, input_nodes = fuse_einsums(output, [a, a_copy, b])
        a_new, a_copy_new, b_new = input_nodes
        assert out_new.inputs == input_nodes

        executor = ad.Executor([out_new])
        out_new_val, = executor.run(feed_dict={
            a_new: a_val,
            a_copy_new: a_val,
            b_new: b_val
        })

        assert T.array_equal(out_val, out_new_val)


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

        executor = ad.Executor([output])

        a_val = T.tensor([[1, 2], [3, 4], [5, 6]])  # 3x2
        b_val = T.tensor([[7, 8, 9], [10, 11, 12]])  # 2x3

        out_val, = executor.run(feed_dict={a: a_val, b: b_val})

        # New graph
        out_new, input_nodes = linearize([output], [a, b])

        a_new, b_new = input_nodes  # Here we keep track of the original input.

        # Need to manually find the to be fused nodes.
        all_nodes = find_topo_sort(out_new)
        cloned_nodes = [
            tmp for tmp in all_nodes if isinstance(tmp, ad.CloneNode)
        ]

        out_new, input_nodes = fuse_einsums(*out_new, [*cloned_nodes, b])
        assert out_new.inputs == input_nodes

        executor = ad.Executor([out_new])
        # Should only run part of the graph.
        out_new_val, = executor.run(feed_dict={a_new: a_val, b_new: b_val})

        assert T.array_equal(out_val, out_new_val)


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

        executor = ad.Executor([out])

        input_values = [
            T.tensor([[1, 2], [3, 4], [5, 6]]),  # 3x2 a1
            T.tensor([[7, 8, 9], [10, 11, 12]]),  # 2x3 a2
            T.tensor([[1, 2], [3, 4], [5, 6]]),  # 3x2 b1
            T.tensor([[7, 8, 9], [10, 11, 12]])  # 2x3 b2
        ] * 4
        z_val, = executor.run(feed_dict=dict(zip(input_nodes, input_values)))

        with OutputInjectedMode(find_topo_sort([out])):
            trees = find_sub_einsumtree(out, input_nodes)
            new_zs = []
            for tree in trees:
                out_node, in_nodes = tree
                new_z, _ = fuse_einsums(out_node, in_nodes)
                replace_node(out_node, new_z)

        executor = ad.Executor([out])
        z_new_val, = executor.run(
            feed_dict=dict(zip(input_nodes, input_values)))

        assert T.array_equal(z_val, z_new_val)
