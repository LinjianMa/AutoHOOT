import autodiff as ad
import backend as T
from source import SourceToSource

BACKEND_TYPES = ['numpy', 'ctf', 'tensorflow']


def test_s2s_hvp():
    for datatype in BACKEND_TYPES:
        T.set_backend(datatype)
        x = ad.Variable(name="x", shape=[3])
        H = ad.Variable(name="H", shape=[3, 3])
        v = ad.Variable(name="v", shape=[3])
        y = ad.einsum("a,ab,b->", x, H, x)

        grad_x, = ad.gradients(y, [x])
        Hv, = ad.hvp(output_node=y, node_list=[x], vector_list=[v])

        x_val = T.tensor([1., 2., 3.])  # 3
        v_val = T.tensor([1., 2., 3.])  # 3
        H_val = T.tensor([[2., 0., 0.], [0., 2., 0.], [0., 0., 2.]])  # 3x3

        expected_yval = T.einsum("a,ab,b->", x_val, H_val, x_val)
        expected_grad_x_val = 2 * T.einsum("ab,b->a", H_val, x_val)
        expected_hv_val = T.tensor([4., 8., 12.])

        StS = SourceToSource()
        StS.forward([y], file=open("example_forward.py", "w"))
        StS.gradients(y, [x], file=open("example_grad.py", "w"))
        StS.hvp(y, [x], [v], file=open("example_hvp.py", "w"))

        import example_forward, example_grad, example_hvp
        y_val_s2s, = example_forward.forward([x_val, H_val])
        grad_x_val_s2s, = example_grad.gradients([x_val, H_val])
        Hv_val_s2s, = example_hvp.hvp([x_val, H_val, v_val])

        assert isinstance(y, ad.Node)
        assert T.array_equal(y_val_s2s, expected_yval)
        assert T.array_equal(grad_x_val_s2s, expected_grad_x_val)
        assert T.array_equal(Hv_val_s2s, expected_hv_val)


def test_s2s_jtjvp():
    for datatype in BACKEND_TYPES:
        T.set_backend(datatype)
        x = ad.Variable(name="x", shape=[2])
        A = ad.Variable(name="A", shape=[3, 2])
        v = ad.Variable(name="v", shape=[2])
        y = ad.einsum("ab,b->a", A, x)

        jtjvp_x, = ad.jtjvps(y, [x], [v])

        x_val = T.tensor([1., 2.])
        A_val = T.tensor([[1., 2.], [3., 4.], [5, 6]])
        v_val = T.tensor([3, 4])

        expected_jtjvp_x_val = T.einsum("ba,bc,c->a", A_val, A_val, v_val)

        StS = SourceToSource()
        StS.forward([jtjvp_x],
                    file=open("example_jtjvp.py", "w"),
                    function_name='jtjvp')

        import example_jtjvp
        jtjvp_x_val_s2s, = example_jtjvp.jtjvp([A_val, v_val])

        assert isinstance(jtjvp_x, ad.Node)
        assert T.array_equal(jtjvp_x_val_s2s, expected_jtjvp_x_val)
