import autodiff as ad
import backend as T
from utils import CharacterGetter

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


if __name__ == "__main__":
    A_list, core, X, loss, residual = tucker_graph(4, 5, 3)
