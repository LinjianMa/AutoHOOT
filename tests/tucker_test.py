import autodiff as ad
import backend as T
from tensors.synthetic_tensors import init_rand_tucker
from examples.tucker import TuckerGraph

BACKEND_TYPES = ['numpy', 'ctf']


def test_tucker():
    for datatype in BACKEND_TYPES:
        T.set_backend(datatype)

        tg = TuckerGraph(4, 5, 3)
        executor = ad.Executor([tg.residual])

        A_val_list, core_val, X_val = init_rand_tucker(4, 5, 3)

        feed_dict = dict(zip(tg.A_list, A_val_list))
        feed_dict.update({tg.core: core_val, tg.X: X_val})

        residual_val, = executor.run(feed_dict=feed_dict)

        expect_residual_val = T.einsum('ae,bf,cg,dh,efgh->abcd', *A_val_list,
                                       core_val) - X_val

        assert T.norm(residual_val - expect_residual_val) < 1e-8
