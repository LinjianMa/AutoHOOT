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


def np_einsum_path(inputs):
    v_A = inputs[0]
    B = inputs[1]
    C = inputs[2]
    v_B = inputs[3]
    A = inputs[4]
    v_C = inputs[5]
    inter_C = T.einsum('ka, kb->ab', C, C)
    inter_A = T.einsum('ia, ib->ab', A, A)
    inter_B = T.einsum('ja, jb->ab', B, B)

    paths = []
    paths.append(
        np.einsum_path('ia,ab,ab->ib', v_A, inter_B, inter_C,
                       optimize=True)[0])
    paths.append(
        np.einsum_path('ja,ia,ab,jb->ib', v_B, A, inter_C, B,
                       optimize=True)[0])
    paths.append(
        np.einsum_path('ka,ia,ab,kb->ib', v_C, A, inter_B, C,
                       optimize=True)[0])
    paths.append(
        np.einsum_path('ia,ja,ab,ib->jb', v_A, B, inter_C, A,
                       optimize=True)[0])
    paths.append(
        np.einsum_path('ja,ab,ab->jb', v_B, inter_A, inter_C,
                       optimize=True)[0])
    paths.append(
        np.einsum_path('ka,ja,kb,ab->jb', v_C, B, C, inter_A,
                       optimize=True)[0])

    paths.append(
        np.einsum_path('ia,ab,ka,ib->kb', v_A, inter_B, C, A,
                       optimize=True)[0])
    paths.append(
        np.einsum_path('ja,ab,ka,jb->kb', v_B, inter_A, C, B,
                       optimize=True)[0])
    paths.append(
        np.einsum_path('ka,ab,ab->kb', v_C, inter_A, inter_B,
                       optimize=True)[0])
    return paths


def np_jtjvp(inputs, paths):

    # forward pass starts
    v_A = inputs[0]
    B = inputs[1]
    C = inputs[2]
    # _b = T.einsum('ia,ja,ka->ijk', v_A, B, C)  # v_A B C
    v_B = inputs[3]
    A = inputs[4]
    # _d = T.einsum('ja,ia,ka->ijk', v_B, A, C)  # v_B A C
    v_C = inputs[5]
    # _g = T.einsum('ka,ia,ja->ijk', v_C, A, B)  # v_C A B
    # _i = T.einsum('ia,ja,ka,kb->ijb', v_A, B, C, C) + T.einsum('ijk,kb->ijb', _d, C) + T.einsum('ijk,kb->ijb', _g, C) # [v_A B C + v_B A C + v_C A B ] C
    inter_C = T.einsum('ka, kb->ab', C, C)
    inter_A = T.einsum('ia, ib->ab', A, A)
    inter_B = T.einsum('ja, jb->ab', B, B)

    inter_AB = T.einsum('ab, ab->ab', inter_A, inter_B)
    inter_AC = T.einsum('ab, ab->ab', inter_A, inter_C)
    inter_BC = T.einsum('ab, ab->ab', inter_B, inter_C)

    _j = T.einsum('ia,ab->ib', v_A, inter_BC) + T.einsum(
        'ja,ia,ab,jb->ib', v_B, A, inter_C, B, optimize=paths[1]) + T.einsum(
            'ka,ia,ab,kb->ib', v_C, A, inter_B, C, optimize=paths[2])

    _k = T.einsum(
        'ia,ja,ab,ib->jb', v_A, B, inter_C, A, optimize=paths[3]) + T.einsum(
            'ja,ab->jb', v_B, inter_AC) + T.einsum(
                'ka,ja,kb,ab->jb', v_C, B, C, inter_A, optimize=paths[5])

    _l = T.einsum('ia,ab,ka,ib->kb', v_A, inter_B, C, A,
                  optimize=paths[6]) + T.einsum(
                      'ja,ab,ka,jb->kb', v_B, inter_A, C, B,
                      optimize=paths[7]) + T.einsum('ka,ab->kb', v_C, inter_AB)

    # _j = T.einsum(
    #     'ia,ab,ab->ib', v_A, inter_B, inter_C, optimize=paths[0]) + T.einsum(
    #         'ja,ia,ab,jb->ib', v_B, A, inter_C, B,
    #         optimize=paths[1]) + T.einsum(
    #             'ka,ia,ab,kb->ib', v_C, A, inter_B, C,
    #             optimize=paths[2])  # [v_A B C + v_B A C + v_C A B ] C B
    # _k = T.einsum(
    #     'ia,ja,ab,ib->jb', v_A, B, inter_C, A, optimize=paths[3]) + T.einsum(
    #         'ja,ab,ab->jb', v_B, inter_A, inter_C,
    #         optimize=paths[4]) + T.einsum(
    #             'ka,ja,kb,ab->jb', v_C, B, C, inter_A,
    #             optimize=paths[5])  # [v_A B C + v_B A C + v_C A B ] C A
    # _l = T.einsum(
    #     'ia,ab,ka,ib->kb', v_A, inter_B, C, A, optimize=paths[6]) + T.einsum(
    #         'ja,ab,ka,jb->kb', v_B, inter_A, C, B,
    #         optimize=paths[7]) + T.einsum(
    #             'ka,ab,ab->kb', v_C, inter_A, inter_B,
    #             optimize=paths[8])  # [v_A B C + v_B A C + v_C A B ] A B
    return [_j, _k, _l]


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

        paths = []
        path_cal = False
        for i in range(10):

            t0 = time.time()

            def hess_fn(v):
                if mode == 'optimized':
                    from examples.cpd_jtjvp_optimized import jtjvp
                    return jtjvp([v[0], B_val, C_val, v[1], A_val, v[2]])
                elif mode == 'ad':
                    nonlocal path_cal
                    if not path_cal:
                        nonlocal paths
                        paths = np_einsum_path(
                            [v[0], B_val, C_val, v[1], A_val, v[2]])
                        path_cal = True
                    return np_jtjvp([v[0], B_val, C_val, v[1], A_val, v[2]],
                                    paths)
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
    cg_time_ad, fitness_ad = cpd_nls(size=100, rank=100, mode='ad')
    cg_time_optimized, fitness_optimized = cpd_nls(size=100,
                                                   rank=100,
                                                   mode='optimized')
    cg_time_jax, fitness_jax = cpd_nls(size=100, rank=100, mode='jax')

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
