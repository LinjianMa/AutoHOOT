import autodiff as ad
import backend as T
from graph_linearizer import linearize

BACKEND_TYPES = ['numpy', 'ctf']


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
        out_new, input_nodes = linearize(output, [a, b])
        a_new, b_new = input_nodes

        executor = ad.Executor([out_new])
        out_new, = executor.run(feed_dict={a_new: a_val, b_new: b_val})

        expected_outval = T.einsum('ac,ab,cd->bd', a_val, a_val, b_val)

        assert T.array_equal(out_val, expected_outval)
        assert T.array_equal(out_new, expected_outval)
