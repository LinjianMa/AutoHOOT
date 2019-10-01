import autodiff as ad
import numpy as np
import backend as T
from source import SourceToSource
from graph_optimizer import fuse_einsums
from graph_linearizer import linearize
from visualizer import print_computation_graph
from utils import find_topo_sort

BACKEND_TYPES = ['numpy', 'ctf']
BACKEND_TYPES = ['numpy']


def test_einsum():

    for datatype in BACKEND_TYPES:
        T.set_backend(datatype)

        a1 = ad.Variable(name="a1", shape=[3, 2])
        a2 = ad.Variable(name="a2", shape=[2, 3])

        b1 = ad.Variable(name="b1", shape=[3, 2])
        b2 = ad.Variable(name="b2", shape=[2, 3])

        x = ad.einsum('ik,kj->ij', a1, a2)
        y = ad.einsum('jl,ls->js', b1, b2)

        z = ad.einsum('ij, js->is', x, y)

        executor = ad.Executor([z, x, y])

        a1_val = T.tensor([[1, 2], [3, 4], [5, 6]])  # 3x2
        a2_val = T.tensor([[7, 8, 9], [10, 11, 12]])  # 2x3

        b1_val = T.tensor([[1, 2], [3, 4], [5, 6]])  # 3x2
        b2_val = T.tensor([[7, 8, 9], [10, 11, 12]])  # 2x3

        z_val, x_val, y_val = executor.run(feed_dict={
            a1: a1_val,
            a2: a2_val,
            b1: b1_val,
            b2: b2_val
        })

        # New graph
        z_new, input_nodes = fuse_einsums(z, [a1, a2, b1, b2])
        a1_new, a2_new, b1_new, b2_new = input_nodes

        executor = ad.Executor([z_new])
        z_new_val, = executor.run(feed_dict={
            a1_new: a1_val,
            a2_new: a2_val,
            b1_new: b1_val,
            b2_new: b2_val
        })

        expected_zval = T.einsum('ik, kj, js, sl->il', a1_val, a2_val, b1_val,
                                 b2_val)

        assert T.array_equal(z_val, expected_zval)
        assert T.array_equal(z_new_val, expected_zval)


def test_einsum_simple_rewrite():

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
        x_new, = executor.run(feed_dict={a1_new: a1_val, a2_new: a2_val})

        expected_xval = T.einsum('ik, kj->ij', a1_val, a2_val)

        assert T.array_equal(x_val, expected_xval)
        assert T.array_equal(x_new, expected_xval)


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

        executor = ad.Executor([out_new])
        out_new, = executor.run(feed_dict={
            a_new: a_val,
            a_copy_new: a_val,
            b_new: b_val
        })

        expected_outval = T.einsum('ac,ab,cd->bd', a_val, a_val, b_val)

        assert T.array_equal(out_val, expected_outval)
        assert T.array_equal(out_new, expected_outval)


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
        print_computation_graph(output)
        out_new, input_nodes = linearize(output, [a, b])

        print_computation_graph(out_new)

        a_new, b_new = input_nodes  # Here we keep track of the original input.

        # Need to manually find the to be fused nodes.
        all_nodes = find_topo_sort([out_new])
        cloned_nodes = [
            tmp for tmp in all_nodes if isinstance(tmp, ad.CloneNode)
        ]

        out_new, input_nodes = fuse_einsums(output, [*cloned_nodes, b])
        print_computation_graph(out_new)

        executor = ad.Executor([out_new])
        # Should only run part of the graph.
        out_new, = executor.run(feed_dict={a_new: a_val, b_new: b_val})

        expected_outval = T.einsum('ac,ab,cd->bd', a_val, a_val, b_val)

        assert T.array_equal(out_val, expected_outval)
        assert T.array_equal(out_new, expected_outval)
