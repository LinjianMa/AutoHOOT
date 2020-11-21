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

import warnings

import numpy as np
import scipy.linalg
import scipy.sparse.linalg


class Backend(object):
    @classmethod
    def register_method(cls, name, func):
        """Register a method with the backend.

        Parameters
        ----------
        name : str
            The method name.
        func : callable
            The method
        """
        setattr(cls, name, staticmethod(func))

    @staticmethod
    def context(tensor):
        """Returns the context of a tensor

        Creates a dictionary of the parameters characterising the tensor.

        Examples
        --------
        >>> import backend as T
        >>> T.set_backend('numpy')
        >>> tensor = T.tensor([0, 1, 2])

        The context, here, will simply be the dtype:

        >>> T.context(tensor)
        {'dtype': dtype('float32')}

        If you want to create a new tensor in the same context, use this context:

        >>> new_tensor = T.tensor([1, 2, 3], **T.context(tensor))
        """
        raise NotImplementedError

    @staticmethod
    def tensor(data, **context):
        """Tensor class
        Returns a tensor on the specified context, depending on the backend.
        """
        raise NotImplementedError

    @staticmethod
    def is_tensor(obj):
        """Returns if `obj` is a tensor for the current backend"""
        raise NotImplementedError

    @staticmethod
    def shape(tensor):
        """Return the shape of a tensor"""
        raise NotImplementedError

    @staticmethod
    def ndim(tensor):
        """Return the number of dimensions of a tensor"""
        raise NotImplementedError

    @staticmethod
    def to_numpy(tensor):
        """Returns a copy of the tensor as a NumPy array.

        """
        raise NotImplementedError

    @staticmethod
    def copy(tensor):
        """Return a copy of the given tensor"""
        raise NotImplementedError

    @staticmethod
    def concatenate(tensors, axis=0):
        """Concatenate tensors along an axis.

        Parameters
        ----------
        tensors : list of tensor
            The tensors to concatenate. Non-empty tensors provided must have the
            same shape, except along the specified axis.
        axis : int, optional
            The axis to concatenate on. Default is 0.

        Returns
        -------
        tensor
        """
        raise NotImplementedError

    @staticmethod
    def reshape(tensor, newshape):
        """Gives a new shape to a tensor without changing its data.

        Parameters
        ----------
        tensor : T.tensor
        newshape : int or tuple of ints
            The new shape should be compatible with the original shape. If an
            integer, then the result will be a 1-D tensor of that length.

        Returns
        -------
        tensor
        """
        raise NotImplementedError

    @staticmethod
    def transpose(tensor):
        """Permute the dimensions of a tensor.

        Parameters
        ----------
        tensor : tensor
        """
        raise NotImplementedError

    @staticmethod
    def moveaxis(tensor, source, destination):
        """Move axes of a tensor to new positions.

        Parameters
        ----------
        tensor : T.tensor
        source : int or sequence of int
            Original positions of the axes to move. These must be unique.
        destination : int or sequence of int
            Destination positions for each of the original axes. These must also be
            unique.

        Returns
        -------
        tensor
        """
        raise NotImplementedError

    @staticmethod
    def arange(start=0, stop=None, step=None):
        """Return evenly spaced values within a given interval.

        Parameters
        ----------
        start : number, optional
            Start of the interval, inclusive. Default is 0.
        stop : number
            End of the interval, exclusive.
        step : number, optional
            Spacing between values. Default is 1.

        Returns
        -------
        tensor
        """
        raise NotImplementedError

    @staticmethod
    def ones(shape, dtype=None):
        """Return a new tensor of given shape and type, filled with ones.

        Parameters
        ----------
        shape : int or sequence of ints
            Shape of the new tensor.
        dtype : data-type, optional
            The desired data-type for the tensor.
        """
        raise NotImplementedError

    @staticmethod
    def zeros(shape, dtype=None):
        """Return a new tensor of given shape and type, filled with zeros.

        Parameters
        ----------
        shape : int or sequence of ints
            Shape of the new tensor.
        dtype : data-type, optional
            The desired data-type for the tensor.
        """
        raise NotImplementedError

    @staticmethod
    def identity(shape, dtype=None):
        raise NotImplementedError

    @staticmethod
    def zeros_like(tensor):
        """Return at tensor of zeros with the same shape and type as a given tensor.

        Parameters
        ----------
        tensor : tensor
        """
        raise NotImplementedError

    @staticmethod
    def ones_like(tensor):
        """Return at tensor of ones with the same shape and type as a given tensor.

        Parameters
        ----------
        tensor : tensor
        """
        raise NotImplementedError

    @staticmethod
    def eye(N):
        """Return a 2-D tensor with ones on the diagonal and zeros elsewhere.

        Parameters
        ----------
        N : int
            Number of rows in the output.
        """
        raise NotImplementedError

    @staticmethod
    def where(condition, x, y):
        """Return elements, either from `x` or `y`, depending on `condition`.

        Parameters
        ----------
        condition : tensor
            When True, yield element from `x`, otherwise from `y`.
        x, y : tensor
            Values from which to choose.

        Returns
        -------
        tensor
        """
        raise NotImplementedError

    @staticmethod
    def clip(tensor, a_min=None, a_max=None):
        """Clip the values of a tensor to within an interval.

        Given an interval, values outside the interval are clipped to the interval
        edges.  For example, if an interval of ``[0, 1]`` is specified, values
        smaller than 0 become 0, and values larger than 1 become 1.

        Not more than one of `a_min` and `a_max` may be `None`.

        Parameters
        ----------
        tensor : tl.tensor
            The tensor.
        a_min : scalar, optional
            Minimum value. If `None`, clipping is not performed on lower bound.
        a_max : scalar, optional
            Maximum value. If `None`, clipping is not performed on upper bound.

        Returns
        -------
        tensor
        """
        raise NotImplementedError

    @staticmethod
    def max(tensor):
        """The max value in a tensor.

        Parameters
        ----------
        tensor : tensor

        Returns
        -------
        scalar
        """
        raise NotImplementedError

    @staticmethod
    def min(tensor):
        """The min value in a tensor.

        Parameters
        ----------
        tensor : tensor

        Returns
        -------
        scalar
        """
        raise NotImplementedError

    @staticmethod
    def argmax(tensor):
        """The argument of the max value in a tensor.

        Parameters
        ----------
        tensor : tensor

        Returns
        -------
        scalar
        """
        raise NotImplementedError

    @staticmethod
    def argmin(tensor):
        """The argument of the min value in a tensor.

        Parameters
        ----------
        tensor : tensor

        Returns
        -------
        scalar
        """
        raise NotImplementedError

    @staticmethod
    def all(tensor):
        """Returns if all array elements in a tensor are True.

        Parameters
        ----------
        tensor : tensor

        Returns
        -------
        bool
        """
        raise NotImplementedError

    @staticmethod
    def mean(tensor, axis=None):
        """Compute the mean of a tensor, optionally along an axis.

        Parameters
        ----------
        tensor : tensor
        axis : int, optional
            If provided, the mean is computed along this axis.

        Returns
        -------
        out : scalar or tensor
        """
        raise NotImplementedError

    @staticmethod
    def sum(tensor, axis=None):
        """Compute the sum of a tensor, optionally along an axis.

        Parameters
        ----------
        tensor : tensor
        axis : int, optional
            If provided, the sum is computed along this axis.

        Returns
        -------
        out : scalar or tensor
        """
        raise NotImplementedError

    @staticmethod
    def prod(tensor, axis=None):
        """Compute the product of a tensor, optionally along an axis.

        Parameters
        ----------
        tensor : tensor
        axis : int, optional
            If provided, the product is computed along this axis.

        Returns
        -------
        out : scalar or tensor
        """
        raise NotImplementedError

    @staticmethod
    def sign(tensor):
        """Computes the element-wise sign of the given input tensor.

        Parameters
        ----------
        tensor : tensor

        Returns
        -------
        out : tensor
        """
        raise NotImplementedError

    @staticmethod
    def abs(tensor):
        """Computes the element-wise absolute value of the given input tensor.

        Parameters
        ----------
        tensor : tensor

        Returns
        -------
        out : tensor
        """
        raise NotImplementedError

    @staticmethod
    def sqrt(tensor):
        """Computes the element-wise sqrt of the given input tensor.

        Parameters
        ----------
        tensor : tensor

        Returns
        -------
        out : tensor
        """
        raise NotImplementedError

    @staticmethod
    def norm(tensor, order=2, axis=None):
        """Computes the l-`order` norm of a tensor.

        Parameters
        ----------
        tensor : tl.tensor
        order : int
        axis : int or tuple

        Returns
        -------
        float or tensor
            If `axis` is provided returns a tensor.
        """
        raise NotImplementedError

    @staticmethod
    def dot(a, b):
        """Dot product of two tensors.

        Parameters
        ----------
        a, b : tensor
            The tensors to compute the dot product of.

        Returns
        -------
        tensor
        """
        raise NotImplementedError

    @staticmethod
    def power(a, b):
        """power of two tensors.
        """
        raise NotImplementedError

    # It's worth noticing that type enforcement won't work here.
    # This is b/c the __signature__ object for a function is not enforced
    # when passed into a function. __signature__ is purely for introspection.
    # So the argspecs here are useless.

    @staticmethod
    def einsum():
        """einstein summation
        """
        raise NotImplementedError

    @staticmethod
    def solve(a, b):
        """Solve a linear matrix equation, or system of linear scalar equations.

        Computes the "exact" solution, `x`, of the well-determined, i.e., full
        rank, linear matrix equation `ax = b`.

        Parameters
        ----------
        a : tensor, shape (M, M)
            The coefficient matrix.
        b : tensor, shape (M,) or (M, K)
            The ordinate values.

        Returns
        -------
        x : tensor, shape (M,) or (M, K)
            Solution to the system a x = b. Returned shape is identical to `b`.
        """
        raise NotImplementedError

    @staticmethod
    def qr(a):
        """Compute the qr factorization of a matrix.

        Factor the matrix `a` as *qr*, where `q` is orthonormal and `r` is
        upper-triangular.

        Parameters
        ----------
        a : tensor, shape (M, N)
            Matrix to be factored.

        Returns
        -------
        Q, R : tensor
        """
        raise NotImplementedError

    @staticmethod
    def tensorinv(a, ind=2):
        raise NotImplementedError

    @staticmethod
    def solve_tri(L, B, lower=True, from_left=True, transp_L=False):
        """ Triangular solver

        Parameters
        ----------
        L: matrix
           Triangular matrix encoding equations

        B: matrix
           Right or left hand sides

        lower: bool
           if true L is lower triangular, if false upper

        from_left: bool
           if true solve LX = B, if false, solve XL=B

        transp_L: bool
           if true solve L^TX = B or XL^T=B

        Returns
        -------
        X: matrix
        """
        raise NotImplementedError

    @staticmethod
    def stack(arrays, axis=0):
        """
        Join a sequence of arrays along a new axis.
        """
        raise NotImplementedError

    @staticmethod
    def random(shape):
        raise NotImplementedError

    @staticmethod
    def seed(seed):
        raise NotImplementedError

    @staticmethod
    def conj(x, *args, **kwargs):
        """Return the complex conjugate, element-wise.

            The complex conjugate of a complex number is obtained by changing the sign of its imaginary part.
        """
        raise NotImplementedError

    @staticmethod
    def array_equal(a, b):
        """ Return whether a and b are equal.
        """
        raise NotImplementedError
