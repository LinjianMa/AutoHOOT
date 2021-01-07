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
from utils import DenseTensor, SparseTensor

# Hack for @ operator for simplicity. No dimension checks.
setattr(pt.tensor, '__matmul__', lambda x, y: TacoBackend.dot(x, y))


def _scalarfunc(t):
    if len(t.shape) == 0:
        # TODO(yejiayu): Extend this to support any type casts.
        return float(t[0])
    else:
        raise TypeError("Only rank 0 can be converted to scalars.")


setattr(pt.tensor, '__float__', _scalarfunc)

###############################################################################
# Impl to get around pytaco slicing.
###############################################################################
__pt_impl_get_item = getattr(pt.tensor, '__getitem__')


def _getitem(self, index):
    if index is None or isinstance(index, int) or len(index) == 0:
        return __pt_impl_get_item(self, index)

    # Fallback to numpy slicing if an input has slice().
    for i in index:
        if isinstance(i, slice):
            return pt.from_array(self.to_array()[index])
    return __pt_impl_get_item(self, index)


setattr(pt.tensor, '__getitem__', lambda self, index: _getitem(self, index))
###############################################################################

setattr(pt.tensor, '__deepcopy__', lambda t, _: pt.as_tensor(t, copy=True))


class TacoBackend(Backend):
    backend_name = 'taco'
    support_sparse_format = True
    pt_format_dict = {"dense": pt.dense, "compressed": pt.compressed}

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
            sp_tensor = sparse.COO.from_numpy(np.array(data, dtype=dtype))
            pt_tensor = pt.tensor(sp_tensor.shape, pt.compressed)
            for i in range(len(sp_tensor.data)):
                pt_tensor.insert(sp_tensor.coords[:, i], sp_tensor.data[i])
            return pt_tensor
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
            sp_tensor = sparse.random(shape, density=density, format='coo')
            pt_tensor = pt.tensor(shape, pt.compressed)
            for i in range(len(sp_tensor.data)):
                pt_tensor.insert(sp_tensor.coords[:, i], sp_tensor.data[i])
            return pt_tensor
        else:
            raise NotImplementedError

    @staticmethod
    def to_numpy(tensor):
        return tensor.to_array()

    @staticmethod
    def get_format(tensor):
        if not isinstance(tensor, pt.pytensor.taco_tensor.tensor):
            return DenseTensor()
        mode_formats = [mode.name for mode in tensor.format.mode_formats]
        if all(name == "dense" for name in mode_formats):
            return DenseTensor()
        return SparseTensor(mode_formats=mode_formats,
                            mode_order=tensor.format.mode_ordering)

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
    def einsum(subscripts, *operands, optimize=True, out_format=DenseTensor()):
        if isinstance(out_format, DenseTensor):
            return pt.einsum(subscripts, *operands)
        pt_mode_formats = [
            TacoBackend.pt_format_dict[fmt] for fmt in out_format.mode_formats
        ]
        pt_format = pt.format(pt_mode_formats, out_format.mode_order)
        return pt.einsum(subscripts, *operands, out_format=pt_format)

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
    def transpose(tensor, axes=None):
        if axes == None:
            assert len(tensor.shape) == 2
            return pt.einsum('ij->ji', tensor)
        return tensor.transpose(axes)

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
        return pt.tensor_sum(A != B) == 0

    @staticmethod
    def identity(n):
        return pt.from_array(np.identity(n))

    @staticmethod
    def svd(matrix):
        u, s, vh = np.linalg.svd(matrix.to_array())
        return pt.from_array(u), pt.from_array(s), pt.from_array(vh)

    @staticmethod
    def seed(seed):
        np.random.seed(seed)

    @staticmethod
    def tensorinv(a, ind=2):
        return pt.from_array(np.linalg.tensorinv(a.to_array(), ind))

    @staticmethod
    def inv(a):
        return pt.from_array(np.linalg.inv(a.to_array()))

    @staticmethod
    def abs(tensor):
        return pt.tensor_abs(tensor, out_format=pt.dense)

    @staticmethod
    def norm(tensor, order=2, axis=None):
        # handle difference in default axis notation
        if axis == ():
            axis = None

        if order == 'inf':
            return pt.tensor_max(pt.tensor_abs(tensor, out_format=pt.dense),
                                 axis=axis)
        elif order == 1:
            return pt.tensor_sum(pt.tensor_abs(tensor, out_format=pt.dense),
                                 axis=axis)
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
