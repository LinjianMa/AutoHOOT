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


cpd_gn_benchmark_numpy(size=100, rank=100, num_iter=10)
