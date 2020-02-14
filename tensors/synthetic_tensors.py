import backend as T


def init_rand_3d(s, R):
    A = T.random((s, R))
    B = T.random((s, R))
    C = T.random((s, R))
    input_tensor = T.einsum("ia,ja->ija", A, B)
    input_tensor = T.einsum("ija,ka->ijk", input_tensor, C)
    A = T.random((s, R))
    B = T.random((s, R))
    C = T.random((s, R))
    return [A, B, C, input_tensor]


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
