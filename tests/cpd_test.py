import autodiff as ad
import backend as T
from graph_ops.graph_transformer import optimize, linearize, simplify
from graph_ops.graph_dedup import dedup
from tensors.synthetic_tensors import init_rand_cp
from examples.cpd import cpd_graph, cpd_als, cpd_als_shared_exec
from utils import find_topo_sort
from graph_ops.graph_als_optimizer import generate_sequential_optiaml_tree

BACKEND_TYPES = ['numpy', 'ctf', 'tensorflow']
size, rank = 10, 5


def expect_jtjvp_val(A, B, C, v_A, v_B, v_C):
    jtjvp_A = T.einsum('ia,ja,ka,kb,jb->ib', v_A, B, C, C, B) + T.einsum(
        'ja,ia,ka,kb,jb->ib', v_B, A, C, C, B) + T.einsum(
            'ka,ia,ja,kb,jb->ib', v_C, A, B, C, B)
    jtjvp_B = T.einsum('ia,ja,ka,kb,ib->jb', v_A, B, C, C, A) + T.einsum(
        'ja,ia,ka,kb,ib->jb', v_B, A, C, C, A) + T.einsum(
            'ka,ia,ja,kb,ib->jb', v_C, A, B, C, A)
    jtjvp_C = T.einsum('ia,ja,ka,ib,jb->kb', v_A, B, C, A, B) + T.einsum(
        'ja,ia,ka,ib,jb->kb', v_B, A, C, A, B) + T.einsum(
            'ka,ia,ja,ib,jb->kb', v_C, A, B, A, B)
    return [jtjvp_A, jtjvp_B, jtjvp_C]


def test_cpd_grad():
    dim = 3
    for datatype in BACKEND_TYPES:
        T.set_backend(datatype)

        A_list, input_tensor, loss, residual = cpd_graph(dim, size, rank)
        A, B, C = A_list
        grad_A, grad_B, grad_C = ad.gradients(loss, [A, B, C])
        executor = ad.Executor([loss, grad_A, grad_B, grad_C])

        A_list, input_tensor_val = init_rand_cp(dim, size, rank)
        A_val, B_val, C_val = A_list
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

        assert abs(loss_val - expected_loss) < 1e-8
        assert T.norm(grad_A_val - expected_grad_A) < 1e-8
        assert T.norm(grad_B_val - expected_grad_B) < 1e-8
        assert T.norm(grad_C_val - expected_grad_C) < 1e-8


def test_cpd_jtjvp():
    dim = 3
    for datatype in BACKEND_TYPES:
        T.set_backend(datatype)

        A_list, input_tensor, loss, residual = cpd_graph(dim, size, rank)
        A, B, C = A_list
        v_A = ad.Variable(name="v_A", shape=[size, rank])
        v_B = ad.Variable(name="v_B", shape=[size, rank])
        v_C = ad.Variable(name="v_C", shape=[size, rank])

        A_list, input_tensor_val = init_rand_cp(dim, size, rank)
        A_val, B_val, C_val = A_list
        v_A_list, _ = init_rand_cp(dim, size, rank)
        v_A_val, v_B_val, v_C_val = v_A_list

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
    dim = 3
    for datatype in BACKEND_TYPES:
        T.set_backend(datatype)

        A_list, input_tensor, loss, residual = cpd_graph(dim, size, rank)
        A, B, C = A_list
        v_A = ad.Variable(name="v_A", shape=[size, rank])
        v_B = ad.Variable(name="v_B", shape=[size, rank])
        v_C = ad.Variable(name="v_C", shape=[size, rank])

        A_list, input_tensor_val = init_rand_cp(dim, size, rank)
        A_val, B_val, C_val = A_list
        v_A_list, _ = init_rand_cp(dim, size, rank)
        v_A_val, v_B_val, v_C_val = v_A_list

        JtJvps = ad.jtjvps(output_node=residual,
                           node_list=[A, B, C],
                           vector_list=[v_A, v_B, v_C])

        JtJvps = [optimize(JtJvp) for JtJvp in JtJvps]
        dedup(*JtJvps)
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


def test_cpd_hessian_simplify():
    dim = 3
    for datatype in BACKEND_TYPES:
        T.set_backend(datatype)

        A_list, input_tensor, loss, residual = cpd_graph(dim, size, rank)
        A, B, C = A_list
        A_list, input_tensor_val = init_rand_cp(dim, size, rank)
        A_val, B_val, C_val = A_list

        hessian = ad.hessian(loss, [A, B, C])
        # TODO (issue #101): test the off-diagonal elements
        hessian_diag = [hessian[0][0], hessian[1][1], hessian[2][2]]
        for node in hessian_diag:
            node = simplify(node)
            input_node = node.inputs[0]
            assert len(input_node.inputs) == 5

        executor = ad.Executor(hessian_diag)
        hes_diag_vals = executor.run(feed_dict={
            A: A_val,
            B: B_val,
            C: C_val,
            input_tensor: input_tensor_val,
        })

        expected_hes_diag_val = [
            2 * T.einsum('eb,ed,fb,fd,ac->abcd', B_val, B_val, C_val, C_val,
                         T.identity(size)),
            2 * T.einsum('eb,ed,fb,fd,ac->abcd', A_val, A_val, C_val, C_val,
                         T.identity(size)),
            2 * T.einsum('eb,ed,fb,fd,ac->abcd', A_val, A_val, B_val, B_val,
                         T.identity(size))
        ]
        assert T.norm(hes_diag_vals[0] - expected_hes_diag_val[0]) < 1e-8
        assert T.norm(hes_diag_vals[1] - expected_hes_diag_val[1]) < 1e-8
        assert T.norm(hes_diag_vals[2] - expected_hes_diag_val[2]) < 1e-8


def test_cpd_hessian_optimize_diag():
    dim = 3
    for datatype in BACKEND_TYPES:
        T.set_backend(datatype)

        A_list, input_tensor, loss, residual = cpd_graph(dim, size, rank)
        A, B, C = A_list
        A_list, input_tensor_val = init_rand_cp(dim, size, rank)
        A_val, B_val, C_val = A_list

        hessian = ad.hessian(loss, [A, B, C])
        hessian_diag = [hessian[0][0], hessian[1][1], hessian[2][2]]
        for node in hessian_diag:
            node = optimize(node)
            assert isinstance(node, ad.AddNode)
            num_operations = len(
                list(
                    filter(lambda x: isinstance(x, ad.OpNode),
                           find_topo_sort([node]))))
            """
            Use this assertion to test the optimize function.
            5 operations:
            1. T.einsum('ca,cb->ab',A,A),
            2. T.einsum('ca,cb->ab',B,B),
            3. T.einsum('ab,ab->ab',T.einsum('ca,cb->ab',A,A),T.einsum('ca,cb->ab',B,B)),
            4. T.einsum('bd,ac->abcd',T.einsum('ab,ab->ab',T.einsum('ca,cb->ab',A,A),T.einsum('ca,cb->ab',B,B)),T.identity(10)),
            5. (T.einsum('bd,ac->abcd',T.einsum('ab,ab->ab',T.einsum('ca,cb->ab',A,A),T.einsum('ca,cb->ab',B,B)),T.identity(10))+
            T.einsum('bd,ac->abcd',T.einsum('ab,ab->ab',T.einsum('ca,cb->ab',A,A),T.einsum('ca,cb->ab',B,B)),T.identity(10)))
            """
            assert num_operations == 5

        executor = ad.Executor(hessian_diag)
        hes_diag_vals = executor.run(feed_dict={
            A: A_val,
            B: B_val,
            C: C_val,
            input_tensor: input_tensor_val,
        })

        expected_hes_diag_val = [
            2 * T.einsum('eb,ed,fb,fd,ac->abcd', B_val, B_val, C_val, C_val,
                         T.identity(size)),
            2 * T.einsum('eb,ed,fb,fd,ac->abcd', A_val, A_val, C_val, C_val,
                         T.identity(size)),
            2 * T.einsum('eb,ed,fb,fd,ac->abcd', A_val, A_val, B_val, B_val,
                         T.identity(size))
        ]
        assert T.norm(hes_diag_vals[0] - expected_hes_diag_val[0]) < 1e-8
        assert T.norm(hes_diag_vals[1] - expected_hes_diag_val[1]) < 1e-8
        assert T.norm(hes_diag_vals[2] - expected_hes_diag_val[2]) < 1e-8


def test_cpd_hessian_optimize_offdiag():
    dim = 3
    for datatype in BACKEND_TYPES:
        T.set_backend(datatype)

        A_list, input_tensor, loss, residual = cpd_graph(dim, size, rank)
        A, B, C = A_list
        A_list, input_tensor_val = init_rand_cp(dim, size, rank)
        A_val, B_val, C_val = A_list

        hessian = ad.hessian(loss, [A, B, C])
        hessian_offdiag = [hessian[0][1], hessian[1][0]]
        for node in hessian_offdiag:
            optimize(node)
            assert isinstance(node, ad.AddNode)
            num_operations = len(
                list(
                    filter(lambda x: isinstance(x, ad.OpNode),
                           find_topo_sort([node]))))
            # This is currently non-deterministic.
            # assert num_operations == 14

        executor = ad.Executor(hessian_offdiag)
        hes_diag_vals = executor.run(feed_dict={
            A: A_val,
            B: B_val,
            C: C_val,
            input_tensor: input_tensor_val,
        })

        # TODO(linjianma): Fix the below tests.
        # expected_hes_diag_val = [
        #     2 * T.einsum('eb,ed,fb,fd,ac->abcd', B_val, B_val, C_val, C_val,
        #                  T.identity(size)),
        #     2 * T.einsum('eb,ed,fb,fd,ac->abcd', A_val, A_val, C_val, C_val,
        #                  T.identity(size)),
        #     2 * T.einsum('eb,ed,fb,fd,ac->abcd', A_val, A_val, B_val, B_val,
        #                  T.identity(size))
        # ]
        # assert T.norm(hes_diag_vals[0] - expected_hes_diag_val[0]) < 1e-8
        # assert T.norm(hes_diag_vals[1] - expected_hes_diag_val[1]) < 1e-8
        # assert T.norm(hes_diag_vals[2] - expected_hes_diag_val[2]) < 1e-8


def test_cpd_als():
    dim = 3

    for datatype in BACKEND_TYPES:
        T.set_backend(datatype)

        input_val = init_rand_cp(dim, size, rank)
        A_list, input_tensor_val = input_val
        A_val, B_val, C_val = A_list

        outputs = cpd_als(dim, size, rank, 1, input_val)

        # expected values
        A_val = T.einsum(
            "abc,bk,ck->ak", input_tensor_val, B_val, C_val) @ T.inv(
                (T.transpose(B_val) @ B_val) * (T.transpose(C_val) @ C_val))
        B_val = T.einsum(
            "abc,ak,ck->bk", input_tensor_val, A_val, C_val) @ T.inv(
                (T.transpose(A_val) @ A_val) * (T.transpose(C_val) @ C_val))
        C_val = T.einsum(
            "abc,ak,bk->ck", input_tensor_val, A_val, B_val) @ T.inv(
                (T.transpose(A_val) @ A_val) * (T.transpose(B_val) @ B_val))

        assert T.norm(outputs[0] - A_val) < 1e-8
        assert T.norm(outputs[1] - B_val) < 1e-8
        assert T.norm(outputs[2] - C_val) < 1e-8


def test_cpd_shared_exec_shape():
    dim = 3

    A_list, input_tensor, loss, residual = cpd_graph(dim, size, rank)
    A, B, C = A_list

    hes = ad.hessian(loss, [A, B, C])
    hes_A, hes_B, hes_C = hes[0][0], hes[1][1], hes[2][2]

    grad_A, grad_B, grad_C = ad.gradients(loss, [A, B, C])

    delta_A = ad.tensordot(ad.tensorinv(hes_A), grad_A, [[2, 3], [0, 1]])
    delta_B = ad.tensordot(ad.tensorinv(hes_B), grad_B, [[2, 3], [0, 1]])
    delta_C = ad.tensordot(ad.tensorinv(hes_C), grad_C, [[2, 3], [0, 1]])

    new_A = A - delta_A
    new_B = B - delta_B
    new_C = C - delta_C

    new_A = simplify(new_A)
    new_B = simplify(new_B)
    new_C = simplify(new_C)
    loss = simplify(loss)

    new_A, new_B, new_C = generate_sequential_optiaml_tree(
        [new_A, new_B, new_C])

    for node in find_topo_sort([new_A, new_B, new_C]):
        assert len(node.shape) <= 3


def test_cpd_shared_exec():
    dim = 3

    for datatype in BACKEND_TYPES:
        T.set_backend(datatype)

        input_val = init_rand_cp(dim, size, rank)
        A_list, input_tensor_val = input_val
        A_val, B_val, C_val = A_list

        outputs = cpd_als_shared_exec(dim, size, rank, 1, input_val)

        # expected values
        A_val = T.einsum(
            "abc,bk,ck->ak", input_tensor_val, B_val, C_val) @ T.inv(
                (T.transpose(B_val) @ B_val) * (T.transpose(C_val) @ C_val))
        B_val = T.einsum(
            "abc,ak,ck->bk", input_tensor_val, A_val, C_val) @ T.inv(
                (T.transpose(A_val) @ A_val) * (T.transpose(C_val) @ C_val))
        C_val = T.einsum(
            "abc,ak,bk->ck", input_tensor_val, A_val, B_val) @ T.inv(
                (T.transpose(A_val) @ A_val) * (T.transpose(B_val) @ B_val))

        assert T.norm(outputs[0] - A_val) < 1e-8
        assert T.norm(outputs[1] - B_val) < 1e-8
        assert T.norm(outputs[2] - C_val) < 1e-8
