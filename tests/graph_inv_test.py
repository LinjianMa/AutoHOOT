import autodiff as ad
import backend as T
from graph_ops.graph_inv_optimizer import optimize_inverse, prune_inv_node
from tests.test_utils import tree_eq


def test_kronecker_product_inv(backendopt):

    for datatype in backendopt:
        T.set_backend(datatype)

        A = ad.Variable(name="A", shape=[2, 2])
        B = ad.Variable(name="B", shape=[3, 3])

        out = ad.einsum("ab,cd->acbd", A, B)
        inv = ad.tensorinv(out)
        newinv = optimize_inverse(inv)

        assert isinstance(newinv, ad.EinsumNode)
        for node in newinv.inputs:
            assert isinstance(node, ad.TensorInverseNode)

        assert tree_eq(inv, newinv, [A, B], tol=1e-6)


def test_kronecker_product_repeated_inputs(backendopt):

    for datatype in backendopt:
        T.set_backend(datatype)

        A = ad.Variable(name="A", shape=[2, 2])

        out = ad.einsum("ab,cd->acbd", A, A)
        inv = ad.tensorinv(out)
        newinv = optimize_inverse(inv)

        assert isinstance(newinv, ad.EinsumNode)
        for node in newinv.inputs:
            assert isinstance(node, ad.TensorInverseNode)

        assert tree_eq(inv, newinv, [A], tol=1e-5)


def test_complex_product_inv(backendopt):

    for datatype in backendopt:
        T.set_backend(datatype)

        A = ad.Variable(name="A", shape=[2, 2])
        B = ad.Variable(name="B", shape=[2, 2])
        C = ad.Variable(name="C", shape=[2, 2])
        D = ad.Variable(name="D", shape=[2, 2])

        out = ad.einsum("ab,bc,de,ef->adcf", A, B, C, D)
        inv = ad.tensorinv(out)
        # T.einsum('ac,df->adcf',
        #     T.tensorinv(T.einsum('ab,bc->ac',A,B), ind=1),
        #     T.tensorinv(T.einsum('de,ef->df',C,D), ind=1))
        newinv = optimize_inverse(inv)

        assert isinstance(newinv, ad.EinsumNode)
        for node in newinv.inputs:
            assert isinstance(node, ad.TensorInverseNode)

        assert tree_eq(inv, newinv, [A, B, C, D], tol=1e-5)


def test_high_dim_inv(backendopt):

    for datatype in backendopt:
        T.set_backend(datatype)

        A = ad.Variable(name="A", shape=[2, 2, 2, 2])
        B = ad.Variable(name="B", shape=[2, 2, 2, 2])

        out = ad.einsum("aceg,dbhf->abcdefgh", A, B)
        inv = ad.tensorinv(out)
        # T.einsum('aceg,bdfh->abcdefgh',
        #     T.tensorinv(T.einsum('aceg->aceg',A), ind=2),
        #     T.tensorinv(T.einsum('dbhf->bdfh',B), ind=2))
        newinv = optimize_inverse(inv)

        assert isinstance(newinv, ad.EinsumNode)
        for node in newinv.inputs:
            assert isinstance(node, ad.TensorInverseNode)

        assert tree_eq(inv, newinv, [A, B], tol=1e-6)


def test_inv_multiple_decomposation(backendopt):
    for datatype in backendopt:
        T.set_backend(datatype)

        A = ad.Variable(name="A", shape=[2, 2])
        B = ad.Variable(name="B", shape=[2, 2])
        C = ad.Variable(name="C", shape=[2, 2])

        out = ad.einsum("ab,cd,ef->acebdf", A, B, C)
        inv = ad.tensorinv(out)
        newinv = optimize_inverse(inv)

        assert isinstance(newinv, ad.EinsumNode)
        for node in newinv.inputs:
            assert isinstance(node, ad.TensorInverseNode)
        assert len(newinv.inputs) == 3

        assert tree_eq(inv, newinv, [A, B, C], tol=1e-5)


def test_kronecker_product_nondecomposable(backendopt):

    A = ad.Variable(name="A", shape=[2, 3])
    B = ad.Variable(name="B", shape=[3, 2])

    out = ad.einsum("ab,cd->acbd", A, B)
    inv = ad.tensorinv(out)
    newinv = optimize_inverse(inv)

    assert inv is newinv


def test_kronecker_product_non_even(backendopt):

    A = ad.Variable(name="A", shape=[4, 4, 2, 2])
    B = ad.Variable(name="B", shape=[2, 2])

    out = ad.einsum("abcd,ef->abcdef", A, B)
    inv = ad.tensorinv(out, ind=2)
    newinv = optimize_inverse(inv)

    assert inv is newinv


def test_prune_inv_nodes_simple(backendopt):
    for datatype in backendopt:
        A = ad.Variable(name="A", shape=[2, 2])
        B = ad.Variable(name="B", shape=[2, 2])

        inv_input = ad.einsum('ab,bc->ac', A, B)
        # inv(inv_input) @ inv_input
        output = ad.einsum('ac,cd,de->ae', ad.tensorinv(inv_input, ind=1), A,
                           B)
        new_output = prune_inv_node(output)

        assert isinstance(new_output, ad.IdentityNode)
        assert tree_eq(output, new_output, [A, B], tol=1e-6)


def test_prune_inv_nodes_transpose(backendopt):
    for datatype in backendopt:
        A = ad.Variable(name="A", shape=[2, 2])
        B = ad.Variable(name="B", shape=[2, 2])

        inv_input = ad.einsum('ab,bc->ca', A, B)
        # inv(inv_input.T) @ inv_input.T
        output = ad.einsum('ca,cd,de->ae', ad.tensorinv(inv_input, ind=1), A,
                           B)
        new_output = prune_inv_node(output)

        assert isinstance(new_output, ad.IdentityNode)
        assert tree_eq(output, new_output, [A, B], tol=1e-6)


def test_prune_inv_matmul_no_pruning(backendopt):
    A = ad.Variable(name="A", shape=[2, 2])
    B = ad.Variable(name="B", shape=[2, 2])

    inv_input = ad.einsum('ab,bc->ac', A, B)
    # inv(inv_input) @ inv_input.T, cannot be pruned
    output = ad.einsum('ac,ed,dc->ae', ad.tensorinv(inv_input, ind=1), A, B)

    new_output = prune_inv_node(output)

    assert new_output is output


def test_prune_inv_nonmatmul_no_pruning(backendopt):
    A = ad.Variable(name="A", shape=[2, 2])
    B = ad.Variable(name="B", shape=[2, 2])

    inv_input = ad.einsum('ab,bc->ac', A, B)
    # inv(inv_input) * inv_input.T, cannot be pruned
    output = ad.einsum('ac,ab,bc->ac', ad.tensorinv(inv_input, ind=1), A, B)

    new_output = prune_inv_node(output)

    assert new_output is output


def test_prune_inv_different_num_inputs_no_pruning(backendopt):
    A = ad.Variable(name="A", shape=[2, 2])

    inv_input = ad.einsum('ab,bc->ac', A, A)
    output = ad.einsum('ab,bc->ac', ad.tensorinv(inv_input, ind=1), A)
    new_output = prune_inv_node(output)

    assert new_output is output


def test_prune_inv_no_inv(backendopt):
    A = ad.Variable(name="A", shape=[2, 2])
    B = ad.Variable(name="B", shape=[2, 2])

    output = ad.einsum('ab,bc->ac', A, B)
    new_output = prune_inv_node(output)

    assert new_output is output


def test_prune_inv_set_not_match(backendopt):
    A = ad.Variable(name="A", shape=[2, 2])
    B = ad.Variable(name="B", shape=[2, 2])

    inv = ad.tensorinv(ad.einsum('ab,bc->ac', A, B), ind=1)
    output = ad.einsum('ab,bc->ac', inv, A)
    new_output = prune_inv_node(output)

    assert new_output is output


def test_prune_inv_multiple_inv(backendopt):
    for datatype in backendopt:
        A0 = ad.Variable(name="A0", shape=[2, 2])
        A1 = ad.Variable(name="A1", shape=[2, 2])
        A2 = ad.Variable(name="A2", shape=[2, 2])

        out = ad.einsum('ab,bc,cd,de,ef,fg,gh->ah', A0, A1, A1,
                        ad.tensorinv(ad.einsum('ab,bc->ac', A1, A1), ind=1),
                        A2, A2,
                        ad.tensorinv(ad.einsum('ab,bc->ac', A2, A2), ind=1))
        new_out = prune_inv_node(out)

        for node in new_out.inputs:
            assert not isinstance(node, ad.EinsumNode)

        assert tree_eq(out, new_out, [A0, A1, A2], tol=1e-6)


def test_prune_inv_nodes_cpd(backendopt):
    for datatype in backendopt:
        A = ad.Variable(name="A", shape=[2, 2])
        B = ad.Variable(name="B", shape=[2, 2])
        C = ad.Variable(name="C", shape=[2, 2])

        inv_input = ad.einsum('ab,dc,ac,db->bc', B, C, B, C)
        output = ad.einsum('ed,ea,cd,ba,ca,gd->bg', C, C, B, A, B,
                           ad.tensorinv(inv_input, ind=1))

        new_output = prune_inv_node(output)

        # T.einsum('ba,ag->bg',A,T.identity(2))
        assert len(new_output.inputs) == 2
        for node in new_output.inputs:
            if isinstance(node, ad.VariableNode):
                assert node == A

        assert tree_eq(output, new_output, [A, B, C], tol=1e-6)
