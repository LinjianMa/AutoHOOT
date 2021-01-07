# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from jax.config import config
config.update('jax_enable_x64', True)

import jax.numpy as np
import jax.scipy.linalg as sla
from jax import random
from .core import Backend
from formats import DenseFormat


class JaxBackend(Backend):
    backend_name = 'jax'
    support_sparse_format = False
    random_index = 0

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
        import numpy as onp
        return onp.asarray(tensor)

    @staticmethod
    def get_format(tensor):
        return DenseFormat()

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

    def random(self, shape):
        rand_val = random.uniform(random.PRNGKey(self.random_index),
                                  shape=shape)
        self.random_index += 1
        return rand_val

    def seed(self, seed):
        self.random_index = seed

    @staticmethod
    def tensorinv(tensor, ind=2):
        import numpy as onp

        oldshape = tensor.shape
        if ind > 0:
            invshape = oldshape[ind:] + oldshape[:ind]
            prod = onp.prod(oldshape[:ind])
            assert prod == onp.prod(oldshape[ind:])
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
