import autodiff as ad
import backend as T
import tensorly as tl
import time
import numpy as np

from tensors.synthetic_tensors import init_rand_tucker
from examples.tucker import tucker_als, tucker_als_shared_exec
from tensorly.decomposition import tucker as tucker_tensorly
from sktensor import dtensor
from sktensor.tucker import hooi as sk_tucker


def tucker_als_benchmark_numpy(dim, size, rank, num_iter):
    T.set_backend('numpy')

    input_val = init_rand_tucker(dim, size, rank)

    # als
    _, _, _, sweep_times_als = tucker_als(dim,
                                          size,
                                          rank,
                                          num_iter,
                                          input_val,
                                          calculate_loss=False,
                                          return_time=True)
    print(f'als time is: {np.mean(sweep_times_als)}')

    # dt
    _, _, _, sweep_times_dt = tucker_als_shared_exec(dim,
                                                     size,
                                                     rank,
                                                     num_iter,
                                                     input_val,
                                                     calculate_loss=False,
                                                     return_time=True)
    print(f'dt time is: {np.mean(sweep_times_dt)}')

    # tensorly
    _, _, input_tensor_val = init_rand_tucker(dim, size, rank)
    input_tensor = tl.tensor(input_tensor_val, dtype='float64')

    _, _, sweep_time_tensorly = tucker_tensorly(input_tensor,
                                                rank=rank,
                                                init='random',
                                                tol=0,
                                                n_iter_max=num_iter,
                                                verbose=0,
                                                return_time=True)
    print(f'Tensorly time is: {np.mean(sweep_time_tensorly)}')

    # sktensor
    _, _, input_tensor_val = init_rand_tucker(dim, size, rank)

    _, _, sweep_time_sktensor = sk_tucker(dtensor(input_tensor_val),
                                          rank=[rank, rank, rank],
                                          maxIter=num_iter,
                                          init='random',
                                          return_time=True)
    print(f'sktensor time is: {np.mean(sweep_time_sktensor)}')

    print('full summary')
    print(f'als time is: {sweep_times_als}')
    print(f'dt time is: {sweep_times_dt}')
    print(f'Tensorly time is: {sweep_time_tensorly}')
    print(f'sktensor time is: {sweep_time_sktensor}')

    print('summary')
    print(f'als time is: {np.mean(sweep_times_als)}')
    print(f'dt time is: {np.mean(sweep_times_dt)}')
    print(f'Tensorly time is: {np.mean(sweep_time_tensorly)}')
    print(f'sktensor time is: {np.mean(sweep_time_sktensor)}')


def tucker_als_benchmark_tf(dim, size, rank, num_iter):
    T.set_backend('tensorflow')
    input_val = init_rand_tucker(dim, size, rank)

    # als
    _, _, _, sweep_times_als = tucker_als(dim,
                                          size,
                                          rank,
                                          num_iter,
                                          input_val,
                                          calculate_loss=False,
                                          return_time=True)
    print(f'als time is: {np.mean(sweep_times_als)}')

    # dt
    _, _, _, sweep_times_dt = tucker_als_shared_exec(dim,
                                                     size,
                                                     rank,
                                                     num_iter,
                                                     input_val,
                                                     calculate_loss=False,
                                                     return_time=True)
    print(f'dt time is: {np.mean(sweep_times_dt)}')

    # tensorly
    tl.set_backend('tensorflow')
    _, _, input_tensor_val = input_val
    input_tensor = tl.tensor(input_tensor_val, dtype='float')

    _, _, sweep_time_tensorly = tucker_tensorly(input_tensor,
                                                rank=rank,
                                                init='random',
                                                tol=0,
                                                n_iter_max=num_iter,
                                                verbose=0,
                                                return_time=True)
    print(f'Tensorly time is: {np.mean(sweep_time_tensorly)}')

    print('full summary')
    print(f'als time is: {sweep_times_als}')
    print(f'dt time is: {sweep_times_dt}')
    print(f'Tensorly time is: {sweep_time_tensorly}')

    print('summary')
    print(f'als time is: {np.mean(sweep_times_als)}')
    print(f'dt time is: {np.mean(sweep_times_dt)}')
    print(f'Tensorly time is: {np.mean(sweep_time_tensorly)}')


import tensorflow as tf
print("Num GPUs Available: ",
      len(tf.config.experimental.list_physical_devices('GPU')))
tucker_als_benchmark_tf(dim=3, size=200, rank=40, num_iter=3)
