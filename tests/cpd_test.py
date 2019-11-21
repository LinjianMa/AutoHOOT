import autodiff as ad
import backend as T
from graph_ops.graph_transformer import optimize
from tensors.synthetic_tensors import init_rand_3d

BACKEND_TYPES = ['numpy']


def cpd_graph(size, rank):
    A = ad.Variable(name='A', shape=[size, rank])
    B = ad.Variable(name='B', shape=[size, rank])
    C = ad.Variable(name='C', shape=[size, rank])
    input_tensor = ad.Variable(name='input_tensor', shape=[size, size, size])
    output_tensor = ad.einsum("ia,ja,ka->ijk", A, B, C)
    residual = output_tensor - input_tensor
    norm_error = ad.norm(residual)
    loss = norm_error * norm_error
    return A, B, C, input_tensor, loss, residual


def expect_jtjvp_val(A, B, C, v_A, v_B, v_C):
    jtjvp_A = T.einsum(
        'ia,ja,ka,kb,jb->ib', v_A, B, C, C, B, optimize=True) + T.einsum(
            'ja,ia,ka,kb,jb->ib', v_B, A, C, C, B, optimize=True) + T.einsum(
                'ka,ia,ja,kb,jb->ib', v_C, A, B, C, B, optimize=True)
    jtjvp_B = T.einsum(
        'ia,ja,ka,kb,ib->jb', v_A, B, C, C, A, optimize=True) + T.einsum(
            'ja,ia,ka,kb,ib->jb', v_B, A, C, C, A, optimize=True) + T.einsum(
                'ka,ia,ja,kb,ib->jb', v_C, A, B, C, A, optimize=True)
    jtjvp_C = T.einsum(
        'ia,ja,ka,ib,jb->kb', v_A, B, C, A, B, optimize=True) + T.einsum(
            'ja,ia,ka,ib,jb->kb', v_B, A, C, A, B, optimize=True) + T.einsum(
                'ka,ia,ja,ib,jb->kb', v_C, A, B, A, B, optimize=True)
    return [jtjvp_A, jtjvp_B, jtjvp_C]


def test_cpd_grad():
    size, rank = 20, 5
    for datatype in BACKEND_TYPES:
        T.set_backend(datatype)

        A, B, C, input_tensor, loss, residual = cpd_graph(size, rank)
        grad_A, grad_B, grad_C = ad.gradients(loss, [A, B, C])
        executor = ad.Executor([loss, grad_A, grad_B, grad_C])

        A_val, B_val, C_val, input_tensor_val = init_rand_3d(size, rank)
        loss_val, grad_A_val, grad_B_val, grad_C_val = executor.run(
            feed_dict={
                input_tensor: input_tensor_val,
                A: A_val,
                B: B_val,
                C: C_val
            })

        expected_output_tensor = T.einsum("ia,ja,ka->ijk", A_val, B_val, C_val)
        expected_residual = expected_output_tensor - input_tensor_val
        expected_norm_error = T.norm(expected_residual)
        expected_loss = expected_norm_error * expected_norm_error

        expected_contract_residual_A = 2 * T.einsum("ijk,ia->ajk",
                                                    expected_residual, A_val)
        expected_contract_residual_B = 2 * T.einsum("ijk,ja->iak",
                                                    expected_residual, B_val)
        expected_contract_residual_C = 2 * T.einsum("ijk,ka->ija",
                                                    expected_residual, C_val)

        expected_grad_A = T.einsum("iak,ka->ia", expected_contract_residual_B,
                                   C_val)
        expected_grad_B = T.einsum("ajk,ka->ja", expected_contract_residual_A,
                                   C_val)
        expected_grad_C = T.einsum("ajk,ja->ka", expected_contract_residual_A,
                                   B_val)

        assert T.array_equal(loss_val, expected_loss)
        assert T.norm(grad_A_val - expected_grad_A) < 1e-8
        assert T.norm(grad_B_val - expected_grad_B) < 1e-8
        assert T.norm(grad_C_val - expected_grad_C) < 1e-8


def test_cpd_jtjvp():
    size, rank = 20, 5
    for datatype in BACKEND_TYPES:
        T.set_backend(datatype)

        A, B, C, input_tensor, loss, residual = cpd_graph(size, rank)
        v_A = ad.Variable(name="v_A", shape=[size, rank])
        v_B = ad.Variable(name="v_B", shape=[size, rank])
        v_C = ad.Variable(name="v_C", shape=[size, rank])

        A_val, B_val, C_val, input_tensor_val = init_rand_3d(size, rank)
        v_A_val, v_B_val, v_C_val, _ = init_rand_3d(size, rank)

        JtJvps = ad.jtjvps(output_node=residual,
                           node_list=[A, B, C],
                           vector_list=[v_A, v_B, v_C])
        executor_JtJvps = ad.Executor(JtJvps)

        jtjvp_val = executor_JtJvps.run(
            feed_dict={
                A: A_val,
                B: B_val,
                C: C_val,
                input_tensor: input_tensor_val,
                v_A: v_A_val,
                v_B: v_B_val,
                v_C: v_C_val
            })

        expected_hvp_val = expect_jtjvp_val(A_val, B_val, C_val, v_A_val,
                                            v_B_val, v_C_val)

        assert T.norm(jtjvp_val[0] - expected_hvp_val[0]) < 1e-8
        assert T.norm(jtjvp_val[1] - expected_hvp_val[1]) < 1e-8
        assert T.norm(jtjvp_val[2] - expected_hvp_val[2]) < 1e-8


def test_cpd_jtjvp_optimize():
    size, rank = 20, 5
    for datatype in BACKEND_TYPES:
        T.set_backend(datatype)

        A, B, C, input_tensor, loss, residual = cpd_graph(size, rank)
        v_A = ad.Variable(name="v_A", shape=[size, rank])
        v_B = ad.Variable(name="v_B", shape=[size, rank])
        v_C = ad.Variable(name="v_C", shape=[size, rank])

        A_val, B_val, C_val, input_tensor_val = init_rand_3d(size, rank)
        v_A_val, v_B_val, v_C_val, _ = init_rand_3d(size, rank)

        JtJvps = ad.jtjvps(output_node=residual,
                           node_list=[A, B, C],
                           vector_list=[v_A, v_B, v_C])

        JtJvps = [optimize(JtJvp) for JtJvp in JtJvps]
        for node in JtJvps:
            assert isinstance(node, ad.AddNode)
        executor_JtJvps = ad.Executor(JtJvps)

        jtjvp_val = executor_JtJvps.run(
            feed_dict={
                A: A_val,
                B: B_val,
                C: C_val,
                input_tensor: input_tensor_val,
                v_A: v_A_val,
                v_B: v_B_val,
                v_C: v_C_val
            })

        expected_hvp_val = expect_jtjvp_val(A_val, B_val, C_val, v_A_val,
                                            v_B_val, v_C_val)

        assert T.norm(jtjvp_val[0] - expected_hvp_val[0]) < 1e-8
        assert T.norm(jtjvp_val[1] - expected_hvp_val[1]) < 1e-8
        assert T.norm(jtjvp_val[2] - expected_hvp_val[2]) < 1e-8
