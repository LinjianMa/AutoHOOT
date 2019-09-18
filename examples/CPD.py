import math
import autodiff as ad
import numpy as np
import backend as T
from tensors.synthetic_tensors import init_rand_3d
from utils import conjugate_gradient, cp_nls_optimizer
import time

BACKEND_TYPES = ['numpy']


def cpd_gradient_descent(size, rank, learning_rate):

    for datatype in BACKEND_TYPES:
        T.set_backend(datatype)

        A_val, B_val, C_val, input_tensor_val = init_rand_3d(size, rank)

        A = ad.Variable(name='A')
        B = ad.Variable(name='B')
        C = ad.Variable(name='C')

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


def cpd_nls(size, rank, regularization=1e-7, optimized=True):

    for datatype in BACKEND_TYPES:
        T.set_backend(datatype)
        T.seed(1)

        A_val, B_val, C_val, input_tensor_val = init_rand_3d(size, rank)

        A = ad.Variable(name='A')
        B = ad.Variable(name='B')
        C = ad.Variable(name='C')
        input_tensor = ad.Variable(name='input_tensor')
        v_A = ad.Variable(name="v_A")
        v_B = ad.Variable(name="v_B")
        v_C = ad.Variable(name="v_C")

        contract_A_B = ad.einsum("ia,ja->ija", A, B)
        output_tensor = ad.einsum("ija,ka->ijk", contract_A_B, C)
        residual = output_tensor - input_tensor
        norm_error = ad.norm(residual)
        loss = norm_error * norm_error

        grads = ad.gradients(loss, [A, B, C])
        JtJvps = ad.jtjvps(output_node=residual,
                           node_list=[A, B, C],
                           vector_list=[v_A, v_B, v_C])

        executor_grads = ad.Executor([loss] + grads)
        executor_JtJvps = ad.Executor(JtJvps)
        optimizer = cp_nls_optimizer(input_tensor_val, [A_val, B_val, C_val])

        regu_increase = False
        normT = T.norm(input_tensor_val)
        time_all, fitness = 0., 0.

        for i in range(10):

            t0 = time.time()

            from examples.cpd_jtjvp_optimized import jtjvp

            def hess_fn(v):
                if optimized:
                    return jtjvp([v[0], B_val, C_val, v[1], A_val, v[2]])
                else:
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
    cg_time_ad, fitness_ad = cpd_nls(size=64, rank=10, optimized=False)
    cg_time_ad_optimized, fitness_optimized = cpd_nls(size=64,
                                                      rank=10,
                                                      optimized=True)
    assert (abs(fitness_ad - fitness_optimized) < 1e-5)
    print(
        f"time with AD is {cg_time_ad/cg_time_ad_optimized} times of the optimized implementation."
    )


def cpd_newton(size, rank):

    for datatype in BACKEND_TYPES:
        T.set_backend(datatype)

        A_val, B_val, C_val, input_tensor_val = init_rand_3d(size, rank)

        A = ad.Variable(name='A')
        B = ad.Variable(name='B')
        C = ad.Variable(name='C')
        input_tensor = ad.Variable(name='input_tensor')
        v_A = ad.Variable(name="v_A")
        v_B = ad.Variable(name="v_B")
        v_C = ad.Variable(name="v_C")

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
