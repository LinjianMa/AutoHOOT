import tensorflow as tf
import numpy as np
from .core import Backend


class TensorflowBackend(Backend):
    backend_name = 'tensorflow'

    @staticmethod
    def context(tensor):
        return {'dtype': tensor.dtype}

    @staticmethod
    def tensor(data, dtype=tf.float64):
        return tf.constant(data, dtype=dtype)

    @staticmethod
    def random(shape, dtype=tf.float64):
        return tf.random.uniform(shape, dtype=dtype)

    @staticmethod
    def identity(length, dtype=tf.float64):
        return tf.eye(length, dtype=dtype)

    @staticmethod
    def is_tensor(tensor):
        return isinstance(tensor, tf.Tensor)

    @staticmethod
    def ones(shape, dtype=tf.float64):
        return tf.ones(shape, dtype)

    @staticmethod
    def svd(matrix):
        s, u, v = tf.linalg.svd(matrix)
        return u, s, v

    @staticmethod
    def to_numpy(tensor):
        if isinstance(tensor, np.ndarray):
            return tensor
        elif isinstance(tensor, tf.Tensor):
            return tensor.numpy()
        else:
            return tensor

    @staticmethod
    def shape(tensor):
        return tensor.shape

    @staticmethod
    def ndim(tensor):
        return tensor.ndim

    @staticmethod
    def clip(tensor, a_min=None, a_max=None, inplace=False):
        raise NotImplementedError

    @staticmethod
    def dot(a, b):
        return a @ b

    @staticmethod
    def einsum(subscripts, *operands, optimize=True):
        if optimize == True:
            return tf.einsum(subscripts, *operands, optimize='greedy')
        elif optimize == False:
            return tf.einsum(subscripts, *operands)

        else:
            return tf.einsum(subscripts, *operands, optimize=optimize)

    @staticmethod
    def norm(tensor, order=2, axis=None):
        if order == 'inf':
            order = np.inf
        return tf.norm(tensor=tensor, ord=order, axis=axis)

    @staticmethod
    def array_equal(a, b):
        return tf.reduce_all(tf.math.equal(a, b))

    @staticmethod
    def tensorinv(tensor, ind=2):
        oldshape = tensor.shape
        if ind > 0:
            invshape = oldshape[ind:] + oldshape[:ind]
            prod = tf.reduce_prod(oldshape[:ind])
            assert prod == tf.reduce_prod(oldshape[ind:])
        else:
            raise ValueError("Invalid ind argument.")
        tensor = tf.reshape(tensor, [prod, -1])
        invtensor = tf.linalg.inv(tensor)
        return tf.reshape(invtensor, oldshape)

    def kr(self, matrices, weights=None, mask=None):
        raise NotImplementedError


for name in [
        'reshape', 'where', 'transpose', 'ones_like', 'zeros',
        'zeros_like', 'eye', 'sign', 'abs', 'sqrt', 'argmin', 'argmax', 'stack'
]:
    TensorflowBackend.register_method(name, getattr(tf, name))
_FUN_NAMES = [
    (tf.concat, 'concatenate'),
    (tf.reduce_min, 'min'),
    (tf.reduce_max, 'max'),
    (tf.reduce_mean, 'mean'),
    (tf.reduce_sum, 'sum'),
    (tf.reduce_prod, 'prod'),
    (tf.math.pow, 'power'),
    (tf.identity, 'copy'),
    (tf.random.set_seed, 'seed'),
]
for source_fun, target_fun_name in _FUN_NAMES:
    TensorflowBackend.register_method(target_fun_name, source_fun)

for name in ['solve', 'qr', 'inv']:
    TensorflowBackend.register_method(name, getattr(tf.linalg, name))
