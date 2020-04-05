import autodiff as ad
import backend as T
import tensorly as tl
import time
import numpy as np

from tensors.synthetic_tensors import init_rand_cp
from examples.cpd import cpd_als, cpd_als_shared_exec
from tensorly.decomposition import parafac
from sktensor import dtensor
from sktensor import cp_als as sk_cp_als


def cpd_als_benchmark_numpy(dim, size, rank, num_iter):
    T.set_backend('numpy')
    input_tensor = init_rand_cp(dim, size, rank)

    # als
    _, sweep_times_als = cpd_als(dim,
                                 size,
                                 rank,
                                 num_iter,
                                 input_tensor,
                                 calculate_loss=False,
                                 return_time=True)
    print(f'als time is: {np.mean(sweep_times_als)}')

    # dt
    _, sweep_times_dt = cpd_als_shared_exec(dim,
                                            size,
                                            rank,
                                            num_iter,
                                            input_tensor,
                                            calculate_loss=False,
                                            return_time=True)
    print(f'dt time is: {np.mean(sweep_times_dt)}')

    # tensorly
    _, input_tensor_val = init_rand_cp(dim, size, rank)
    input_tensor = tl.tensor(input_tensor_val, dtype='float64')

    _, tensorly_sweep_times = parafac(input_tensor,
                                      rank=rank,
                                      init='random',
                                      tol=None,
                                      n_iter_max=num_iter,
                                      normalize_factors=False,
                                      verbose=1,
                                      ret_time=True)
    print(f'Tensorly time is: {np.mean(tensorly_sweep_times)}')

    # sktensor
    _, input_tensor_val = init_rand_cp(dim, size, rank)

    _, _, _, sk_sweep_times = sk_cp_als(dtensor(input_tensor_val),
                                        rank=rank,
                                        max_iter=num_iter,
                                        init='random',
                                        fit_method=None)
    print(f'sktensor time is: {np.mean(sk_sweep_times)}')

    print('summary')
    print(f'als time is: {np.mean(sweep_times_als)}')
    print(f'dt time is: {np.mean(sweep_times_dt)}')
    print(f'Tensorly time is: {np.mean(tensorly_sweep_times)}')
    print(f'sktensor time is: {np.mean(sk_sweep_times)}')


cpd_als_benchmark_numpy(dim=3, size=200, rank=200, num_iter=3)
