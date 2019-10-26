import math
import autodiff as ad
import numpy as np
import backend as T
from tensors.synthetic_tensors import init_rand_3d
from utils import conjugate_gradient, cp_nls_optimizer
from utils import OutputInjectedMode, find_topo_sort
from utils import replace_node
from graph_optimizer import find_sub_einsumtree, fuse_einsums
from graph_linearizer import linearize
from visualizer import print_computation_graph
import time

BACKEND_TYPES = ['numpy']


def cpd_gradient_descent(size, rank, learning_rate):

    for datatype in BACKEND_TYPES:
        T.set_backend(datatype)

        A_val, B_val, C_val, input_tensor_val = init_rand_3d(size, rank)

        A = ad.Variable(name='A', shape=[size, rank])
        B = ad.Variable(name='B', shape=[size, rank])
        C = ad.Variable(name='C', shape=[size, rank])

        contract_A_B = ad.einsum("ia,ja->ija", A, B)
        output_tensor = ad.einsum("ija,ka->ijk", contract_A_B, C)
        norm_error = ad.norm(output_tensor - input_tensor_val)
        loss = norm_error * norm_error

        grad_A, grad_B, grad_C = ad.gradients(loss, [A, B, C])
        executor = ad.Executor([loss, grad_A, grad_B, grad_C])

        for i in range(100):
            loss_val, grad_A_val, grad_B_val, grad_C_val = executor.run(
                feed_dict={
                    A: A_val,
                    B: B_val,
                    C: C_val
                })
            A_val -= learning_rate * grad_A_val
            B_val -= learning_rate * grad_B_val
            C_val -= learning_rate * grad_C_val
            print(f'At iteration {i} the loss is: {loss_val}')


def cpd_nls(size, rank, regularization=1e-7, mode='ad'):
    """
    mode: ad / optimized / jax
    """
    assert mode in {'ad', 'jax', 'optimized'}

    for datatype in BACKEND_TYPES:
        T.set_backend(datatype)
        T.seed(1)

        A_val, B_val, C_val, input_tensor_val = init_rand_3d(size, rank)

        A = ad.Variable(name='A', shape=[size, rank])
        B = ad.Variable(name='B', shape=[size, rank])
        C = ad.Variable(name='C', shape=[size, rank])
        input_tensor = ad.Variable(name='input_tensor',
                                   shape=[size, size, size])
        v_A = ad.Variable(name="v_A", shape=[size, rank])
        v_B = ad.Variable(name="v_B", shape=[size, rank])
        v_C = ad.Variable(name="v_C", shape=[size, rank])

        contract_A_B = ad.einsum("ia,ja->ija", A, B)
        output_tensor = ad.einsum("ija,ka->ijk", contract_A_B, C)
        residual = output_tensor - input_tensor
        norm_error = ad.norm(residual)
        loss = norm_error * norm_error

        grads = ad.gradients(loss, [A, B, C])
        JtJvps = ad.jtjvps(output_node=residual,
                           node_list=[A, B, C],
                           vector_list=[v_A, v_B, v_C])

        if mode == 'jax':
            from source import SourceToSource
            StS = SourceToSource()
            StS.forward(JtJvps,
                        file=open("examples/jax_jtjvp.py", "w"),
                        function_name='jtjvp',
                        backend='jax')

        executor_grads = ad.Executor([loss] + grads)
        executor_JtJvps = ad.Executor(JtJvps)
        optimizer = cp_nls_optimizer(input_tensor_val, [A_val, B_val, C_val])

        regu_increase = False
        normT = T.norm(input_tensor_val)
        time_all, fitness = 0., 0.

        for i in range(10):

            t0 = time.time()

            def hess_fn(v):
                if mode == 'optimized':
                    from examples.cpd_jtjvp_optimized import jtjvp
                    return jtjvp([v[0], B_val, C_val, v[1], A_val, v[2]])
                elif mode == 'ad':
                    linearize(JtJvps, [A, B, C, input_tensor, v_A, v_B, v_C])
                    with OutputInjectedMode(find_topo_sort(JtJvps)):
                        for JtJvp in JtJvps:
                            trees = find_sub_einsumtree(
                                JtJvp, [A, B, C, input_tensor, v_A, v_B, v_C])
                            for tree in trees:
                                out_node, in_nodes = tree
                                print(in_nodes)
                                # print_computation_graph([out_node], in_nodes)
                                # assert False
                                new_z, _ = fuse_einsums(out_node, in_nodes)
                                # print_computation_graph([new_z])
                                replace_node(out_node, new_z)

                    return executor_JtJvps.run(
                        feed_dict={
                            A: A_val,
                            B: B_val,
                            C: C_val,
                            input_tensor: input_tensor_val,
                            v_A: v[0],
                            v_B: v[1],
                            v_C: v[2]
                        })
                elif mode == 'jax':
                    from examples.jax_jtjvp import jtjvp
                    return jtjvp([v[0], B_val, C_val, v[1], A_val, v[2]])

            loss_val, grad_A_val, grad_B_val, grad_C_val = executor_grads.run(
                feed_dict={
                    A: A_val,
                    B: B_val,
                    C: C_val,
                    input_tensor: input_tensor_val
                })

            res = math.sqrt(loss_val)
            fitness = 1 - res / normT
            print(f"[ {i} ] Residual is {res} fitness is: {fitness}")
            print(f"Regularization is: {regularization}")

            [A_val, B_val, C_val], total_cg_time = optimizer.step(
                hess_fn=hess_fn,
                grads=[grad_A_val / 2, grad_B_val / 2, grad_C_val / 2],
                regularization=regularization)

            t1 = time.time()
            print(f"[ {i} ] Sweep took {t1 - t0} seconds")
            time_all += t1 - t0

            if regularization < 1e-07:
                regu_increase = True
            elif regularization > 1:
                regu_increase = False
            if regu_increase:
                regularization = regularization * 2
            else:
                regularization = regularization / 2

        return total_cg_time, fitness


def cpd_nls_benchmark():
    cg_time_ad, fitness_ad = cpd_nls(size=64, rank=10, mode='ad')
    cg_time_optimized, fitness_optimized = cpd_nls(size=64,
                                                   rank=10,
                                                   mode='optimized')
    cg_time_jax, fitness_jax = cpd_nls(size=64, rank=10, mode='jax')

    assert (abs(fitness_ad - fitness_optimized) < 1e-3)
    assert (abs(fitness_jax - fitness_optimized) < 1e-3)

    print(
        f"time with AD is [{cg_time_ad/cg_time_optimized}] times of the optimized implementation."
    )
    print(
        f"time with jax is [{cg_time_jax/cg_time_optimized}] times of the optimized implementation."
    )


def cpd_newton(size, rank):

    for datatype in BACKEND_TYPES:
        T.set_backend(datatype)

        A_val, B_val, C_val, input_tensor_val = init_rand_3d(size, rank)

        A = ad.Variable(name='A', shape=[size, rank])
        B = ad.Variable(name='B', shape=[size, rank])
        C = ad.Variable(name='C', shape=[size, rank])
        input_tensor = ad.Variable(name='input_tensor',
                                   shape=[size, size, size])
        v_A = ad.Variable(name="v_A", shape=[size, rank])
        v_B = ad.Variable(name="v_B", shape=[size, rank])
        v_C = ad.Variable(name="v_C", shape=[size, rank])

        contract_A_B = ad.einsum("ia,ja->ija", A, B)
        output_tensor = ad.einsum("ija,ka->ijk", contract_A_B, C)
        norm_error = ad.norm(output_tensor - input_tensor)
        loss = norm_error * norm_error

        grads = ad.gradients(loss, [A, B, C])
        Hvps = ad.hvp(output_node=loss,
                      node_list=[A, B, C],
                      vector_list=[v_A, v_B, v_C])

        executor_grads = ad.Executor([loss] + grads)
        executor_Hvps = ad.Executor(Hvps)

        for i in range(100):

            def hess_fn(v):
                return executor_Hvps.run(
                    feed_dict={
                        A: A_val,
                        B: B_val,
                        C: C_val,
                        input_tensor: input_tensor_val,
                        v_A: v[0],
                        v_B: v[1],
                        v_C: v[2]
                    })

            loss_val, grad_A_val, grad_B_val, grad_C_val = executor_grads.run(
                feed_dict={
                    A: A_val,
                    B: B_val,
                    C: C_val,
                    input_tensor: input_tensor_val
                })

            delta = conjugate_gradient(
                hess_fn=hess_fn,
                grads=[grad_A_val, grad_B_val, grad_C_val],
                error_tol=1e-9,
                max_iters=250)

            A_val -= delta[0]
            B_val -= delta[1]
            C_val -= delta[2]
            print(f'At iteration {i} the loss is: {loss_val}')


if __name__ == "__main__":
    # cpd_gradient_descent(size=20, rank=5, learning_rate=1e-3)
    # cpd_NLS(size=64, rank=10, optimized=False)
    # cpd_newton(size=20, rank=5)
    cpd_nls_benchmark()
