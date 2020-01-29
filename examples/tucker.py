import autodiff as ad
import backend as T
from utils import CharacterGetter

BACKEND_TYPES = ['numpy']


def tucker_graph(dim, size, rank):
    """
    Produce a graph representing the Tucker decomposition:

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
    X.subscripts = "".join([cg.getchar() for _ in range(dim)])

    core = ad.Variable(name='core', shape=[rank for _ in range(dim)])
    core.subscripts = "".join([cg.getchar() for _ in range(dim)])

    A_list = []
    for i in range(dim):
        node = ad.Variable(name=f'A{i}', shape=[size, rank])
        node.subscripts = f"{X.subscripts[i]}{core.subscripts[i]}"
        A_list.append(node)

    input_subs = ','.join([node.subscripts for node in A_list + [core]])
    einsum_subscripts = input_subs + '->' + X.subscripts

    output = ad.einsum(einsum_subscripts, *(A_list + [core]))
    residual = output - X
    loss = ad.einsum(f"{X.subscripts},{X.subscripts}->", residual, residual)
    return A_list, core, X, loss, residual


if __name__ == "__main__":
    A_list, core, X, loss, residual = tucker_graph(4, 5, 3)
    print(residual)
