import autodiff as ad
import backend as T
from source import SourceToSource


def import_code(code, name="ephermal"):
    """
    Import dynamically generated code as a module. code is the
    object containing the code (a string, a file handle or an
    actual compiled code object, same types as accepted by an
    exec statement). The name is the name to give to the module.

    import foo

    is equivalent to

    foofile = open("/path/to/foo.py")
    foo = importCode(foofile,"foo",1)

    Returns a newly generated module.
    """

    import importlib.util
    spec = importlib.util.spec_from_loader(name, loader=None)
    m = importlib.util.module_from_spec(spec)
    exec(str(code), m.__dict__)
    return m


def test_s2s_hvp(backendopt):
    for datatype in backendopt:
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
        forward_str = StS.forward([y], backend=datatype)
        m = import_code(forward_str)
        y_val_s2s, = m.forward([x_val, H_val])
        grad_str = StS.gradients(y, [x], backend=datatype)
        m = import_code(grad_str)
        grad_x_val_s2s, = m.gradients([x_val, H_val])
        hvp_str = StS.hvp(y, [x], [v], backend=datatype)
        m = import_code(hvp_str)
        Hv_val_s2s, = m.hvp([x_val, H_val, v_val])

        assert isinstance(y, ad.Node)
        assert T.array_equal(y_val_s2s, expected_yval)
        assert T.array_equal(grad_x_val_s2s, expected_grad_x_val)
        assert T.array_equal(Hv_val_s2s, expected_hv_val)


def test_s2s_jtjvp(backendopt):
    for datatype in backendopt:
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
        forward_str = StS.forward([jtjvp_x],
                                  function_name='jtjvp',
                                  backend=datatype)
        m = import_code(forward_str)
        jtjvp_x_val_s2s, = m.jtjvp([A_val, v_val])

        assert isinstance(jtjvp_x, ad.Node)
        assert T.array_equal(jtjvp_x_val_s2s, expected_jtjvp_x_val)


def test_s2s_w_constants(backendopt):
    for datatype in backendopt:
        T.set_backend(datatype)
        A = ad.Variable(name="A", shape=[2, 2])
        I = ad.identity(2)
        B = A @ I

        A_val = T.tensor([[1., 2.], [3., 4.]])

        StS = SourceToSource()
        fwd_str = StS.forward([B], function_name='fwd', backend=datatype)
        m = import_code(fwd_str)
        out, = m.fwd([A_val])

        assert T.array_equal(A_val, out)
