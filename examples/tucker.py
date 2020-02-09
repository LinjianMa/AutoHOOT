import autodiff as ad
import backend as T
from utils import CharacterGetter
from tensors.synthetic_tensors import init_rand_tucker

BACKEND_TYPES = ['numpy']


def tucker_graph(dim, size, rank):
    """
    Produce a graph representing the Tucker decomposition.

    Note: current graph produces the decomposition with equidimensional core tensor.

    Parameters
    ----------
    dim: dimensionality of the input tensor
    size: the size of input tensor's each dim
    rank: the rank of the decomposition

    Returns
    -------
    1. A list of decomposed matrices nodes
    2. Decomposed core tensor node
    3. Input tensor
    4. Tucker decomposition loss
    5. Residual of the decomposition

    """
    cg = CharacterGetter()

    X = ad.Variable(name='X', shape=[size for _ in range(dim)])
    X_subscripts = "".join([cg.getchar() for _ in range(dim)])

    core = ad.Variable(name='core', shape=[rank for _ in range(dim)])
    core_subscripts = "".join([cg.getchar() for _ in range(dim)])

    A_list = []
    A_list_subscripts = []
    for i in range(dim):
        node = ad.Variable(name=f'A{i}', shape=[size, rank])
        A_list.append(node)
        A_list_subscripts.append(f"{X_subscripts[i]}{core_subscripts[i]}")

    input_subs = ','.join(
        [subscripts for subscripts in A_list_subscripts + [core_subscripts]])
    einsum_subscripts = input_subs + '->' + X_subscripts

    output = ad.einsum(einsum_subscripts, *(A_list + [core]))
    residual = output - X
    loss = ad.einsum(f"{X_subscripts},{X_subscripts}->", residual, residual)
    return A_list, core, X, loss, residual


def tucker_als(dim, size, rank, num_iter, input_val=[]):
    A_list, core, X, loss, residual = tucker_graph(dim, size, rank)

    executors = []

    for i in range(dim):

        core_A = ad.intermediate(loss, {core, A_list[i]})
        hes = ad.hessian(loss, core_A)
        grad = ad.gradients(loss, core_A)

        new_core_A = core_A - ad.tensordot(
            ad.tensorinv(hes), grad,
            [[i + dim for i in range(dim)], [i for i in range(dim)]])

        executor = ad.Executor([loss, new_core_A])
        executors.append(executor)

    if input_val == []:
        A_list_val, core_val, X_val = init_rand_tucker(dim, size, rank)
    else:
        A_list_val, core_val, X_val = input_val

    for iter in range(num_iter):
        # als iterations
        for i in range(dim):

            loss_val, new_core_A_val = executors[i].run(feed_dict={  #TODO
            })

            # update core and A_i

        print(f'At iteration {iter} the loss is: {loss_val}')

    return A_val, B_val, C_val


if __name__ == "__main__":
    A_list, core, X, loss, residual = tucker_graph(4, 5, 3)
