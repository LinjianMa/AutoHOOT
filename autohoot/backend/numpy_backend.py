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

import numpy as np
import scipy.linalg as sla
# import sparse
from autohoot import formats
from .core import Backend


class NumpyBackend(Backend):
    backend_name = 'numpy'
    support_sparse_format = True

    @staticmethod
    def context(tensor):
        return {'dtype': tensor.dtype}

    @staticmethod
    def tensor(data, dtype=None, format="dense"):
        """
        Parameters
        ----------
        data: the input data representing the tensor.
        dtype: datatype
        format: a string denoting the tensor datatype.
            if "dense", then return a dense tensor.
            if "coo", then return a sparse tensor in the COO format.
        """
        # if the data is not a np array (e.g. in the sparse format),
        # first transfer it to the standard np array.
        data = NumpyBackend.to_numpy(data).astype(dtype)
        if format == "dense":
            return data
        # elif format == "coo":
        #     return sparse.COO.from_numpy(data)
        else:
            raise NotImplementedError

    @staticmethod
    def is_tensor(tensor):
        typelist = (np.ndarray)#, sparse._coo.core.COO)
        return isinstance(tensor, typelist)

    @staticmethod
    def random(shape, format='dense', density=1.):
        if format == "dense":
            return np.random.random(shape)
        # elif format == "coo":
        #     return sparse.random(shape, density=density, format='coo')
        else:
            raise NotImplementedError

    @staticmethod
    def to_numpy(tensor):
        # if isinstance(tensor, sparse._coo.core.COO):
        #     # transfer the sparse tensor to numpy array
        #     return tensor.todense()
        # else:
        return np.copy(tensor)

    @staticmethod
    def get_format(tensor):
        # if isinstance(tensor, sparse._coo.core.COO):
        #     return formats.SparseFormat(
        #         [formats.compressed for _ in range(tensor.ndim)])
        # else:
        return formats.DenseFormat()

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
    def einsum(subscripts,
               *operands,
               optimize=True,
               out_format=formats.DenseFormat()):
        # if isinstance(out_format, formats.SparseFormat):
        #     raise NotImplementedError
        # NumPy einsum cannot correctly optimize some einsums, use opt_einsum instead.
        from opt_einsum import contract
        return contract(subscripts, *operands, optimize=optimize)

    @staticmethod
    def solve_tri(A, B, lower=True, from_left=True, transp_L=False):
        if not isinstance(A, np.ndarray):
            A = A.todense()
        if not isinstance(B, np.ndarray):
            B = B.todense()

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


for name in [
        'reshape', 'moveaxis', 'where', 'copy', 'transpose', 'arange', 'ones',
        'ones_like', 'zeros', 'zeros_like', 'eye', 'kron', 'concatenate',
        'max', 'min', 'all', 'mean', 'sum', 'prod', 'sign', 'abs', 'sqrt',
        'argmin', 'argmax', 'stack', 'conj', 'array_equal', 'power',
        'identity', 'diag', 'tensordot'
]:
    NumpyBackend.register_method(name, getattr(np, name))

for name in ['solve', 'qr', 'inv', 'tensorinv', 'cholesky', 'svd', 'eigh']:
    NumpyBackend.register_method(name, getattr(np.linalg, name))

for name in ['seed']:
    NumpyBackend.register_method(name, getattr(np.random, name))
