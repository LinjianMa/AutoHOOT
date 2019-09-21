import numpy as np
from .core import Backend


class NumpyBackend(Backend):
    backend_name = 'numpy'

    @staticmethod
    def context(tensor):
        return {'dtype': tensor.dtype}

    @staticmethod
    def tensor(data, dtype=None):
        return np.array(data, dtype=dtype)

    @staticmethod
    def is_tensor(tensor):
        return isinstance(tensor, np.ndarray)

    @staticmethod
    def to_numpy(tensor):
        return np.copy(tensor)

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


for name in [
        'reshape', 'moveaxis', 'where', 'copy', 'transpose', 'arange', 'ones',
        'ones_like', 'zeros', 'zeros_like', 'eye', 'kron', 'concatenate',
        'max', 'min', 'all', 'mean', 'sum', 'prod', 'sign', 'abs', 'sqrt',
        'argmin', 'argmax', 'stack', 'conj', 'array_equal', 'power', 'einsum'
]:
    NumpyBackend.register_method(name, getattr(np, name))

for name in ['solve', 'qr']:
    NumpyBackend.register_method(name, getattr(np.linalg, name))

for name in ['random', 'seed']:
    NumpyBackend.register_method(name, getattr(np.random, name))
