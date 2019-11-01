import autodiff as ad
import backend as T
from graph_ops.graph_transformer import linearize, distribute
from graph_ops.graph_optimizer import find_sub_einsumtree

BACKEND_TYPES = ['numpy', 'ctf']
BACKEND_TYPES = ['numpy']

# TODO(yejiayu): Find a test engine that generate the test func name as prefix.


def test_einsum_multiuse():
    """
        An einsum graph like
        A    B   inputs 
        |\   |
        | \  |
        |  \ |
        |   C
        |  / 
        | /
        output

        will produce

        An einsum graph like
        A    B   inputs 
        |\   |
        | A1 |
        |  \ |
        A2  C
        |  / 
        | /
        output
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
        a_new, b_new = input_nodes

        executor = ad.Executor(out_new)
        out_new_val, = executor.run(feed_dict={a_new: a_val, b_new: b_val})

        assert T.array_equal(out_val, out_new_val)


def test_einsum_find_subtree_after_linearization():
    """
        An einsum graph like
        A    B   inputs 
        |\   |
        | \  |
        |  \ |
        |   C
        |  / 
        | /
        output

        will produce

        An einsum graph like
        A    B   inputs 
        |\   |
        | A1 |
        |  \ |
        A2  C
        |  / 
        | /
        output

        The subtree inputs must then be [A1, A2, B] rather than A, B.
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
        assert len(out_new) == 1
        tree, = find_sub_einsumtree(*out_new, input_nodes)
        assert (len(tree[1]) == 3)


def test_tree_distribution():
    """
        [Distributive] An einsum graph like
        A    B     C  inputs 
         \   |     |
          \  |     |
           \ |     |
           A + B   |
             \     |
              \    |
               \   |
               output

        will produce

        A    C    B inputs 
         \   |\   |
          \  | \  |
           \ |  \ |
            AC   BC
             \    |
              \   |
               \  |
               output (AC + BC)
        put in workd (A + B) * C = A * C + B * C where * is an einsum node.

    """

    for datatype in BACKEND_TYPES:
        T.set_backend(datatype)

        a = ad.Variable(name="a", shape=[3, 2])
        b = ad.Variable(name="b", shape=[3, 2])
        c = ad.Variable(name="c", shape=[2, 3])

        add_node = a + b
        output = ad.einsum('ik,kj->ij', add_node, c)

        new_output = distribute(add_node, output)

        executor = ad.Executor([output])

        a_val = T.tensor([[1, 2], [3, 4], [5, 6]])  # 3x2
        b_val = T.tensor([[1, 2], [3, 4], [5, 6]])  # 3x2
        c_val = T.tensor([[7, 8, 9], [10, 11, 12]])  # 2x3

        out_val, = executor.run(feed_dict={a: a_val, b: b_val, c: c_val})

        executor = ad.Executor([new_output])
        new_out_val, = executor.run(feed_dict={a: a_val, b: b_val, c: c_val})

        # New graph
        assert (out_val == new_out_val).all()


def test_tree_distribution_order():
    """
        [Distributive]
        Test C * (A + B) = C * A + C * B
    """

    for datatype in BACKEND_TYPES:
        T.set_backend(datatype)

        a = ad.Variable(name="a", shape=[3, 2])
        b = ad.Variable(name="b", shape=[3, 2])
        c = ad.Variable(name="c", shape=[2, 3])

        add_node = a + b
        output = ad.einsum('ik,kj->ij', c, add_node)

        new_output = distribute(add_node, output)

        executor = ad.Executor([output])

        a_val = T.tensor([[1, 2], [3, 4], [5, 6]])  # 3x2
        b_val = T.tensor([[1, 2], [3, 4], [5, 6]])  # 3x2
        c_val = T.tensor([[7, 8, 9], [10, 11, 12]])  # 2x3

        out_val, = executor.run(feed_dict={a: a_val, b: b_val, c: c_val})

        executor = ad.Executor([new_output])
        new_out_val, = executor.run(feed_dict={a: a_val, b: b_val, c: c_val})

        # New graph
        assert (out_val == new_out_val).all()
