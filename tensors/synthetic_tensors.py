import backend as T


def init_rand_cp(dim, size, rank):

    X = T.random([size for _ in range(dim)])

    A_list = []
    for i in range(dim):
        A_list.append(T.random((size, rank)))

    return A_list, X


def init_rand_tucker(dim, size, rank):
    assert size > rank

    X = T.random([size for _ in range(dim)])
    core = T.random([rank for _ in range(dim)])

    A_list = []
    for i in range(dim):
        # for Tucker, factor matrices are orthogonal
        mat, _, _ = T.svd(T.random((size, rank)))
        A_list.append(mat[:, :rank])

    return A_list, core, X
