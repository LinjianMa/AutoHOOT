import math
import autodiff as ad
import backend as T
from tensors.synthetic_tensors import init_rand_cp
from utils import conjugate_gradient, cp_nls_optimizer, CharacterGetter
from graph_ops.graph_transformer import optimize, simplify
from graph_ops.graph_dedup import dedup
from graph_ops.graph_als_optimizer import generate_sequential_optiaml_tree
import time

BACKEND_TYPES = ['numpy']


def cpd_graph(dim, size, rank):
    cg = CharacterGetter()

    input_tensor = ad.Variable(name='input_tensor',
                               shape=[size for _ in range(dim)])
    input_tensor_subs = "".join([cg.getchar() for _ in range(dim)])

    rank_char = cg.getchar()

    A_list = []
    A_list_subs = []
    for i in range(dim):
        node = ad.Variable(name=f'A{i}', shape=[size, rank])
        A_list.append(node)
        A_list_subs.append(f"{input_tensor_subs[i]}{rank_char}")

    input_subs = ','.join(A_list_subs)
    einsum_subscripts = input_subs + '->' + input_tensor_subs
    output = ad.einsum(einsum_subscripts, *A_list)

    residual = output - input_tensor
    residual_shape = list(range(len(residual.shape)))

    loss = ad.tensordot(residual,
                        residual,
                        axes=[residual_shape, residual_shape])

    return A_list, input_tensor, loss, residual


def cpd_gradient_descent(size, rank, learning_rate):
    dim = 3

    for datatype in BACKEND_TYPES:
        T.set_backend(datatype)

        A_list, input_tensor, loss, residual = cpd_graph(dim, size, rank)
        A, B, C = A_list
        grad_A, grad_B, grad_C = ad.gradients(loss, [A, B, C])
        executor = ad.Executor([loss, grad_A, grad_B, grad_C])

        A_list, input_tensor_val = init_rand_cp(dim, size, rank)
        A_val, B_val, C_val = A_list

        for i in range(100):
            loss_val, grad_A_val, grad_B_val, grad_C_val = executor.run(
                feed_dict={
                    input_tensor: input_tensor_val,
                    A: A_val,
                    B: B_val,
                    C: C_val
                })
            A_val -= learning_rate * grad_A_val
            B_val -= learning_rate * grad_B_val
            C_val -= learning_rate * grad_C_val
            print(f'At iteration {i} the loss is: {loss_val}')


def cpd_als(dim, size, rank, num_iter, input_val=[], calculate_loss=True):

    A_list, input_tensor, loss, residual = cpd_graph(dim, size, rank)

    full_hessian = ad.hessian(loss, A_list)
    hessians = [full_hessian[i][i] for i in range(len(full_hessian))]
    grads = ad.gradients(loss, A_list)

    updates = [
        ad.tensordot(ad.tensorinv(hes), grad, [[2, 3], [0, 1]])
        for (hes, grad) in zip(hessians, grads)
    ]

    new_A_list = [simplify(A - update) for (A, update) in zip(A_list, updates)]

    executor = ad.Executor(new_A_list)
    executor_loss = ad.Executor([simplify(loss)])

    if input_val == []:
        A_val_list, input_tensor_val = init_rand_cp(dim, size, rank)
    else:
        A_val_list, input_tensor_val = input_val

    for iter in range(num_iter):
        # als iterations
        for i in range(len(A_list)):

            feed_dict = dict(zip(A_list, A_val_list))
            feed_dict.update({input_tensor: input_tensor_val})
            A_val_list[i], = executor.run(feed_dict=feed_dict,
                                          out_nodes=[new_A_list[i]])

        if calculate_loss:
            feed_dict = dict(zip(A_list, A_val_list))
            feed_dict.update({input_tensor: input_tensor_val})
            loss_val, = executor_loss.run(feed_dict=feed_dict)
            print(f'At iteration {iter} the loss is: {loss_val}')

    return A_val_list


def cpd_als_shared_exec(dim,
                        size,
                        rank,
                        num_iter,
                        input_val=[],
                        calculate_loss=True):

    A_list, input_tensor, loss, residual = cpd_graph(dim, size, rank)

    full_hessian = ad.hessian(loss, A_list)
    hessians = [full_hessian[i][i] for i in range(len(full_hessian))]
    grads = ad.gradients(loss, A_list)

    updates = [
        ad.tensordot(ad.tensorinv(hes), grad, [[2, 3], [0, 1]])
        for (hes, grad) in zip(hessians, grads)
    ]

    new_A_list = [simplify(A - update) for (A, update) in zip(A_list, updates)]
    new_A_list = generate_sequential_optiaml_tree(new_A_list)

    executor = ad.Executor(new_A_list)
    executor_loss = ad.Executor([simplify(loss)])

    if input_val == []:
        A_val_list, input_tensor_val = init_rand_cp(dim, size, rank)
    else:
        A_val_list, input_tensor_val = input_val

    for iter in range(num_iter):
        # als iterations
        for i in range(len(A_list)):

            feed_dict = dict(zip(A_list, A_val_list))
            feed_dict.update({input_tensor: input_tensor_val})

            if i == 0:
                A_val_list[0], = executor.run(feed_dict=feed_dict,
                                              out_nodes=[new_A_list[0]])
            else:
                A_val_list[i], = executor.run(
                    feed_dict=feed_dict,
                    reset_graph=False,
                    evicted_inputs=[A_list[i - 1]],
                    out_nodes=[new_A_list[i]])

        if calculate_loss:
            feed_dict = dict(zip(A_list, A_val_list))
            feed_dict.update({input_tensor: input_tensor_val})
            loss_val, = executor_loss.run(feed_dict=feed_dict)
            print(f'At iteration {iter} the loss is: {loss_val}')

    return A_val_list


def cpd_nls(size, rank, regularization=1e-7, mode='ad'):
    """
    mode: ad / optimized / jax
    """
    assert mode in {'ad', 'jax', 'optimized'}

    dim = 3

    for datatype in BACKEND_TYPES:
        T.set_backend(datatype)
        T.seed(1)

        A_list, input_tensor, loss, residual = cpd_graph(dim, size, rank)
        A, B, C = A_list

        v_A = ad.Variable(name="v_A", shape=[size, rank])
        v_B = ad.Variable(name="v_B", shape=[size, rank])
        v_C = ad.Variable(name="v_C", shape=[size, rank])
        grads = ad.gradients(loss, [A, B, C])
        JtJvps = ad.jtjvps(output_node=residual,
                           node_list=[A, B, C],
                           vector_list=[v_A, v_B, v_C])

        A_list, input_tensor_val = init_rand_cp(dim, size, rank)
        A_val, B_val, C_val = A_list

        if mode == 'jax':
            from source import SourceToSource
            StS = SourceToSource()
            StS.forward(JtJvps,
                        file=open("examples/jax_jtjvp.py", "w"),
                        function_name='jtjvp',
                        backend='jax')

        executor_grads = ad.Executor([loss] + grads)
        JtJvps = [optimize(JtJvp) for JtJvp in JtJvps]
        dedup(*JtJvps)
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
                    return jtjvp([B_val, C_val, v[0], A_val, v[1], v[2]])

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


def cpd_nls_benchmark(size=100, rank=100):
    cg_time_ad, fitness_ad = cpd_nls(size, rank, mode='ad')
    cg_time_optimized, fitness_optimized = cpd_nls(size,
                                                   rank,
                                                   mode='optimized')
    cg_time_jax, fitness_jax = cpd_nls(size, rank, mode='jax')

    assert (abs(fitness_ad - fitness_optimized) < 1e-3)
    assert (abs(fitness_jax - fitness_optimized) < 1e-3)

    print(
        f"time with AD is [{cg_time_ad/cg_time_optimized}] times of the optimized implementation."
    )
    print(
        f"time with jax is [{cg_time_jax/cg_time_optimized}] times of the optimized implementation."
    )


def cpd_newton(size, rank):
    dim = 3

    for datatype in BACKEND_TYPES:
        T.set_backend(datatype)

        A_list, input_tensor, loss, residual = cpd_graph(dim, size, rank)
        A, B, C = A_list
        v_A = ad.Variable(name="v_A", shape=[size, rank])
        v_B = ad.Variable(name="v_B", shape=[size, rank])
        v_C = ad.Variable(name="v_C", shape=[size, rank])
        grads = ad.gradients(loss, [A, B, C])
        Hvps = ad.hvp(output_node=loss,
                      node_list=[A, B, C],
                      vector_list=[v_A, v_B, v_C])

        executor_grads = ad.Executor([loss] + grads)
        executor_Hvps = ad.Executor(Hvps)

        A_list, input_tensor_val = init_rand_cp(dim, size, rank)
        A_val, B_val, C_val = A_list

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
    # cpd_newton(size=20, rank=5)
    # cpd_nls_benchmark()
    cpd_als_shared_exec(400, 100, 1)
