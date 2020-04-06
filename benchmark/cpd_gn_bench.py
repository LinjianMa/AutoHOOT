import autodiff as ad
import backend as T
import numpy as np

from tensors.synthetic_tensors import init_rand_cp
from examples.cpd import cpd_nls

num_iter = 10
dim = 3


def cpd_gn_benchmark_numpy(size, rank, num_iter):
    T.set_backend('numpy')
    input_val = init_rand_cp(dim, size, rank)

    # als
    _, _, cg_times_ad = cpd_nls(size,
                                rank,
                                mode='ad',
                                input_val=input_val,
                                num_iter=num_iter,
                                return_cg_detail=True)
    print(f'als time is: {np.mean(cg_times_ad)}')

    # optimized
    _, _, cg_times_opt = cpd_nls(size,
                                 rank,
                                 mode='optimized',
                                 input_val=input_val,
                                 num_iter=num_iter,
                                 return_cg_detail=True)
    print(f'opt time is: {np.mean(cg_times_opt)}')

    # jax
    _, _, cg_times_jax = cpd_nls(size,
                                 rank,
                                 mode='jax',
                                 input_val=input_val,
                                 num_iter=num_iter,
                                 return_cg_detail=True)
    print(f'jax time is: {np.mean(cg_times_jax)}')

    print('summary')
    print(f'als time is: {np.mean(cg_times_ad)}')
    print(f'opt time is: {np.mean(cg_times_opt)}')
    print(f'jax time is: {np.mean(cg_times_jax)}')

    print('summary_full')
    print(f'als time list is: {cg_times_ad}')
    print(f'opt time list is: {cg_times_opt}')
    print(f'jax time list is: {cg_times_jax}')


def cpd_gn_benchmark_tf(size, rank, num_iter):
    T.set_backend('tensorflow')
    input_val = init_rand_cp(dim, size, rank)

    # als
    _, _, cg_times_ad = cpd_nls(size,
                                rank,
                                mode='ad',
                                input_val=input_val,
                                num_iter=num_iter,
                                return_cg_detail=True)
    print(f'als time is: {np.mean(cg_times_ad)}')

    # optimized
    _, _, cg_times_opt = cpd_nls(size,
                                 rank,
                                 mode='optimized',
                                 input_val=input_val,
                                 num_iter=num_iter,
                                 return_cg_detail=True)
    print(f'opt time is: {np.mean(cg_times_opt)}')

    print('summary')
    print(f'als time is: {np.mean(cg_times_ad)}')
    print(f'opt time is: {np.mean(cg_times_opt)}')

    print('summary_full')
    print(f'als time list is: {cg_times_ad}')
    print(f'opt time list is: {cg_times_opt}')


def cpd_gn_benchmark_tf_jax(size, rank, num_iter):
    T.set_backend('numpy')
    input_val = init_rand_cp(dim, size, rank)

    # chagne to float
    A_list, input_tensor_val = input_val
    A_val, B_val, C_val = A_list
    A_val, B_val, C_val = A_val.astype(np.float32), B_val.astype(
        np.float32), C_val.astype(np.float32)
    input_tensor_val = input_tensor_val.astype(np.float32)
    input_val = [[A_val, B_val, C_val], input_tensor_val]

    # jax
    _, _, cg_times_jax = cpd_nls(size,
                                 rank,
                                 mode='jax',
                                 input_val=input_val,
                                 num_iter=num_iter,
                                 return_cg_detail=True)
    print(f'jax time is: {np.mean(cg_times_jax)}')

    print('summary')
    print(f'jax time is: {np.mean(cg_times_jax)}')

    print('summary_full')
    print(f'jax time list is: {cg_times_jax}')


import tensorflow as tf
print("Num GPUs Available: ",
      len(tf.config.experimental.list_physical_devices('GPU')))
cpd_gn_benchmark_tf(size=40, rank=40, num_iter=10)
cpd_gn_benchmark_tf_jax(size=40, rank=40, num_iter=10)
