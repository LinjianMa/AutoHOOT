import autodiff as ad
import backend as T
from source import SourceToSource

BACKEND_TYPES = ['numpy', 'ctf']


def test_s2s_hvp():
    for datatype in BACKEND_TYPES:
        T.set_backend(datatype)
        x = ad.Variable(name="x")
        H = ad.Variable(name="H")
        v = ad.Variable(name="v")
        y = ad.transpose(x) @ H @ x

        grad_x, = ad.gradients(y, [x])
        Hv, = ad.hvp(output_node=y, node_list=[x], vector_list=[v])

        x_val = T.tensor([[1.], [2.], [3]])  # 2x1
        v_val = T.tensor([[1.], [2.], [3]])  # 2x1
        H_val = T.tensor([[2., 0., 0.], [0., 2., 0.], [0., 0., 2.]])  # 2x2

        expected_yval = T.transpose(x_val) @ H_val @ x_val
        expected_grad_x_val = 2 * H_val @ x_val
        expected_hv_val = T.tensor([[4.], [8.], [12.]])

        StS = SourceToSource()
        StS.forward(y, file=open("example_forward.py", "w"))
        StS.gradients(y, [x], file=open("example_grad.py", "w"))
        StS.hvp(y, [x], [v], file=open("example_hvp.py", "w"))

        import example_forward, example_grad, example_hvp
        y_val_s2s = example_forward.forward([x_val, H_val])
        grad_x_val_s2s, = example_grad.gradients([x_val, H_val])
        Hv_val_s2s, = example_hvp.hvp([x_val, H_val, v_val])

        assert isinstance(y, ad.Node)
        assert T.array_equal(y_val_s2s, expected_yval)
        assert T.array_equal(grad_x_val_s2s, expected_grad_x_val)
        assert T.array_equal(Hv_val_s2s, expected_hv_val)

