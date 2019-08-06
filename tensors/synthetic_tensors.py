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
