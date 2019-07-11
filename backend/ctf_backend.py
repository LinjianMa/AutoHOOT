import numpy as np
import ctf
from .core import Backend


class CTFBackend(Backend):
    backend_name = 'ctf'

    @staticmethod
    def context(tensor):
        return {'dtype': tensor.dtype}

    @staticmethod
    def tensor(data, dtype=None):
        return ctf.astensor(data, dtype=dtype)

    @staticmethod
    def is_tensor(tensor):
        return isinstance(tensor, ctf.core.tensor)

    @staticmethod
    def to_numpy(tensor):
        return ctf.to_nparray(tensor)

    @staticmethod
    def shape(tensor):
        return tensor.shape

    @staticmethod
    def ndim(tensor):
        return tensor.ndim

    # @staticmethod
    # def clip(tensor, a_min=None, a_max=None, inplace=False):
    #     return np.clip(tensor, a_min, a_max)

    @staticmethod
    def norm(tensor, order=2, axis=None):
        # handle difference in default axis notation
        # TODO: this is not complete
        return ctf.vecnorm(T)
        # if axis == ():
        #     axis = None

        # if order == 'inf':
        #     return np.max(np.abs(tensor), axis=axis)
        # if order == 1:
        #     return np.sum(np.abs(tensor), axis=axis)
        # elif order == 2:
        #     return np.sqrt(np.sum(tensor**2, axis=axis))
        # else:
        #     return np.sum(np.abs(tensor)**order, axis=axis)**(1 / order)

    @staticmethod
    def array_equal(a, b):
        return np.array_equal(ctf.to_nparray(a), ctf.to_nparray(b))

    @staticmethod
    def ones_like(tensor):
        return ctf.ones(tensor.shape)


for name in ['reshape', 'transpose', 'copy', 'qr', 'ones', 'zeros',
             'zeros_like', 'eye', 'abs', 'dot', 'power', 'einsum']:
    CTFBackend.register_method(name, getattr(ctf, name))
