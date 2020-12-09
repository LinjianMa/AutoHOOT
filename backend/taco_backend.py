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
import pytaco as pt
import scipy.linalg as sla
import sparse
from .core import Backend

# No neg for taco, FR pending (https://github.com/tensor-compiler/taco/issues/342).
setattr(pt.tensor, '__neg__', lambda x: 0 - x)


class TacoBackend(Backend):
    backend_name = 'taco'

    @staticmethod
    def context(tensor):
        return {'dtype': tensor.dtype}

    @staticmethod
    def tensor(data, dtype=None, format="dense"):
        """
        Parameters
        ----------
        data: the input multidimentional array.
        dtype: datatype
        format: a string denoting the tensor datatype.
            if "dense", then return a dense tensor.
            if "coo", then return a sparse tensor in the COO format.
        """
        if format == "dense":
            return pt.from_array(np.array(data))
        elif format == "coo":
            return sparse.COO.from_numpy(np.array(data, dtype=dtype))
        else:
            raise NotImplementedError

    @staticmethod
    def is_tensor(tensor):
        typelist = (np.ndarray, sparse._coo.core.COO, pt.tensor)
        return isinstance(tensor, typelist)

    @staticmethod
    def random(shape, format='dense', density=1.):
        if format == "dense":
            return pt.from_array(np.random.random(shape))
        elif format == "coo":
            return sparse.random(shape, density=density, format='coo')
        else:
            raise NotImplementedError

    @staticmethod
    def to_numpy(tensor):
        if isinstance(tensor, sparse._coo.core.COO):
            # transfer the sparse tensor to numpy array
            return tensor.todense()
        else:
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
    def einsum(subscripts, *operands, optimize=True):
        # Use taco einsum evaluation.
        return pt.einsum(subscripts, *operands)

    @staticmethod
    def ones(shape, dtype=None):
        return pt.from_array(np.ones(shape, dtype))

    @staticmethod
    def ones_like(tensor):
        return TacoBackend.ones(tensor.shape)

    @staticmethod
    def power(A, B):
        return pt.tensor_pow(A, B, out_format=pt.dense)

    @staticmethod
    def transpose(tensor):
        assert len(tensor.shape) == 2
        return pt.einsum('ij->ji', tensor)

    @staticmethod
    def dot(A, B):
        # This is just matrix multiplication in test code.
        # Taco dot has SEGV issues, avoid.
        return pt.einsum('ij,jk->ik', A, B)

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
    def array_equal(A, B):
        return np.array_equal(A.to_array(), B.to_array())

    @staticmethod
    def tensorinv(a, ind=2):
        return pt.from_array(np.linalg.tensorinv(a.to_array(), ind))

    @staticmethod
    def inv(a):
        return pt.from_array(np.linalg.inv(a.to_array()))

    @staticmethod
    def norm(tensor, order=2, axis=None):
        # handle difference in default axis notation
        if axis == ():
            axis = None

        if order == 'inf':
            return pt.tensor_max(pt.tensor_abs(tensor), axis=axis)
        elif order == 1:
            return pt.tensor_sum(pt.tensor_abs(tensor), axis=axis)
        elif order == 2:
            return pt.tensor_sqrt(pt.tensor_sum(tensor**2, axis=axis),
                                  out_format=pt.dense)
        else:
            return pt.tensor_sum(pt.tesnor_abs(tensor)**order,
                                 axis=axis)**(1 / order)


for name in ['tensordot']:
    TacoBackend.register_method(name, getattr(pt, name))
for name in ['sum']:
    TacoBackend.register_method(name, getattr(pt, 'tensor_' + name))
