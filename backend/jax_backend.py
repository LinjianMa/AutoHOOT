from jax.config import config
config.update('jax_enable_x64', True)

import jax.numpy as np
import jax.scipy.linalg as sla
from jax import random
from .core import Backend


class JaxBackend(Backend):
    backend_name = 'jax'
    PRNG_key = random.PRNGKey(0)

    @staticmethod
    def context(tensor):
        return {'dtype': tensor.dtype}

    @staticmethod
    def tensor(data, dtype=np.float64):
        return np.array(data, dtype=dtype)

    @staticmethod
    def is_tensor(tensor):
        return isinstance(tensor, np.ndarray)

    @staticmethod
    def to_numpy(tensor):
        return tensor

    @staticmethod
    def shape(tensor):
        return tensor.shape

    @staticmethod
    def ndim(tensor):
        return tensor.ndim

    @staticmethod
    def clip(tensor, a_min=None, a_max=None, inplace=False):
        return np.clip(tensor, a_min, a_max)

    @staticmethod
    def dot(a, b):
        return a.dot(b)

    @staticmethod
    def einsum(subscripts, *operands, optimize=True):
        return np.einsum(subscripts, *operands, optimize=optimize)

    @staticmethod
    def solve_tri(A, B, lower=True, from_left=True, transp_L=False):
        if not from_left:
            return sla.solve_triangular(A.T,
                                        B.T,
                                        trans=transp_L,
                                        lower=not lower).T
        else:
            return sla.solve_triangular(A, B, trans=transp_L, lower=lower)

    @staticmethod
    def norm(tensor, order=2, axis=None):
        # handle difference in default axis notation
        if axis == ():
            axis = None

        if order == 'inf':
            return np.max(np.abs(tensor), axis=axis)
        elif order == 1:
            return np.sum(np.abs(tensor), axis=axis)
        elif order == 2:
            return np.sqrt(np.sum(tensor**2, axis=axis))
        else:
            return np.sum(np.abs(tensor)**order, axis=axis)**(1 / order)

    def kr(self, matrices, weights=None, mask=None):
        if mask is None:
            mask = 1
        n_columns = matrices[0].shape[1]
        n_factors = len(matrices)

        start = ord('a')
        common_dim = 'z'
        target = ''.join(chr(start + i) for i in range(n_factors))
        source = ','.join(i + common_dim for i in target)
        operation = source + '->' + target + common_dim

        if weights is not None:
            matrices = [
                m if i else m * self.reshape(weights, (1, -1))
                for i, m in enumerate(matrices)
            ]

        return np.einsum(operation, *matrices).reshape((-1, n_columns)) * mask

    def random(self, shape):
        return random.uniform(self.PRNG_key, shape=shape)

    def seed(self, seed):
        self.PRNG_key = random.PRNGKey(seed)

    @staticmethod
    def tensorinv(tensor, ind=2):
        oldshape = tensor.shape
        if ind > 0:
            invshape = oldshape[ind:] + oldshape[:ind]
            prod = np.prod(oldshape[:ind])
            assert prod == np.prod(oldshape[ind:])
        else:
            raise ValueError("Invalid ind argument.")
        tensor = np.reshape(tensor, [prod, -1])
        invtensor = np.linalg.inv(tensor)
        return np.reshape(invtensor, oldshape)


for name in [
        'reshape', 'moveaxis', 'where', 'copy', 'transpose', 'arange', 'ones',
        'ones_like', 'zeros', 'zeros_like', 'eye', 'kron', 'concatenate',
        'max', 'min', 'all', 'mean', 'sum', 'prod', 'sign', 'abs', 'sqrt',
        'argmin', 'argmax', 'stack', 'conj', 'array_equal', 'power',
        'identity', 'diag', 'tensordot'
]:
    JaxBackend.register_method(name, getattr(np, name))

for name in ['solve', 'qr', 'inv', 'cholesky', 'svd', 'eigh']:
    JaxBackend.register_method(name, getattr(np.linalg, name))
