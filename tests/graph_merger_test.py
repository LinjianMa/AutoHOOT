import autodiff as ad
import numpy as np
import backend as T
from source import SourceToSource
from graph_optimizer import fuse_einsums

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
        z_new, = executor.run(feed_dict={
            a1: a1_val,
            a2: a2_val,
            b1: b1_val,
            b2: b2_val
        })

        expected_zval = T.einsum('ik, kj, js, sl->il', a1_val, a2_val, b1_val,
                                 b2_val)

        assert T.array_equal(z_val, expected_zval)
        assert T.array_equal(z_new, expected_zval)


test_einsum()
