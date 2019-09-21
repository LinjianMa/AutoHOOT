import backend as T
from utils import conjugate_gradient

BACKEND_TYPES = ['numpy', 'ctf']


def test_HinverseG():
    for datatype in BACKEND_TYPES:
        T.set_backend(datatype)

        N = 10
        T.seed(1224)

        A = T.random([N, N])
        A = T.transpose(A) @ A
        A = A + T.eye(N)
        b = T.random(N)

        def hess_fn(x):
            return [A @ x[0]]

        error_tol = 1e-9
        x, = conjugate_gradient(hess_fn, [b], error_tol)
        assert (T.norm(T.abs(A @ x - b)) <= 1e-4)
