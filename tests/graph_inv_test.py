import autodiff as ad
import backend as T
from graph_ops.graph_inv_optimizer import optimize_inverse, optimize_inverse2
from tests.test_utils import tree_eq

BACKEND_TYPES = ['numpy', 'ctf']
BACKEND_TYPES = ['numpy']


def test_kronecker_product_inv():

    for datatype in BACKEND_TYPES:
        T.set_backend(datatype)

        A = ad.Variable(name="A", shape=[2, 2])
        B = ad.Variable(name="B", shape=[3, 3])

        out = ad.einsum("ab,cd->acbd", A, B)
        inv = ad.tensorinv(out)
        newinv = optimize_inverse2(inv)

        assert isinstance(newinv, ad.EinsumNode)
        for node in newinv.inputs:
            assert isinstance(node, ad.TensorInverseNode)

        assert tree_eq(inv, newinv, [A, B], tol=1e-6)


def test_odd_tensor_inv():

    for datatype in BACKEND_TYPES:
        T.set_backend(datatype)

        A = ad.Variable(name="A", shape=[4, 4])
        B = ad.Variable(name="B", shape=[2, 2, 4])

        out = ad.einsum("ab,cde->acdbe", A, B)  # 4x2x2x4x4
        inv = ad.tensorinv(out, ind=3)  # 4x4x4x2x2
        newinv = optimize_inverse2(inv)

        assert isinstance(newinv, ad.EinsumNode)
        for node in newinv.inputs:
            assert isinstance(node, ad.TensorInverseNode)

        assert tree_eq(inv, newinv, [A, B], tol=1e-6)


def test_kronecker_product_repeated_inputs():

    for datatype in BACKEND_TYPES:
        T.set_backend(datatype)

        A = ad.Variable(name="A", shape=[2, 2])

        out = ad.einsum("ab,cd->acbd", A, A)
        inv = ad.tensorinv(out)
        newinv = optimize_inverse2(inv)

        assert isinstance(newinv, ad.EinsumNode)
        for node in newinv.inputs:
            assert isinstance(node, ad.TensorInverseNode)

        assert tree_eq(inv, newinv, [A], tol=1e-6)


def test_complex_product_inv():

    for datatype in BACKEND_TYPES:
        T.set_backend(datatype)

        A = ad.Variable(name="A", shape=[2, 2])
        B = ad.Variable(name="B", shape=[2, 2])
        C = ad.Variable(name="C", shape=[2, 2])
        D = ad.Variable(name="D", shape=[2, 2])

        # We would need to merge non output indices first.
        out = ad.einsum("ab,bc,de,ef->adcf", A, B, C, D)
        inv = ad.tensorinv(out)
        # T.einsum('ac,df->adcf',
        #     T.tensorinv(T.einsum('ab,bc->ac',A,B), ind=1),
        #     T.tensorinv(T.einsum('de,ef->df',C,D), ind=1))
        newinv = optimize_inverse(inv)

        assert isinstance(newinv, ad.EinsumNode)
        for node in newinv.inputs:
            assert isinstance(node, ad.TensorInverseNode)

        assert tree_eq(inv, newinv, [A, B, C, D], tol=1e-6)


def test_high_dim_inv():

    for datatype in BACKEND_TYPES:
        T.set_backend(datatype)

        A = ad.Variable(name="A", shape=[2, 2, 2, 2])
        B = ad.Variable(name="B", shape=[2, 2, 2, 2])

        out = ad.einsum("aceg,dbhf->abcdefgh", A, B)
        inv = ad.tensorinv(out)
        # T.einsum('aceg,bdfh->abcdefgh',
        #     T.tensorinv(T.einsum('aceg->aceg',A), ind=2),
        #     T.tensorinv(T.einsum('dbhf->bdfh',B), ind=2))
        newinv = optimize_inverse2(inv)

        assert isinstance(newinv, ad.EinsumNode)
        for node in newinv.inputs:
            assert isinstance(node, ad.TensorInverseNode)

        assert tree_eq(inv, newinv, [A, B], tol=1e-6)
