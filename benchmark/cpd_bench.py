import autodiff as ad
import backend as T
import tensorly as tl
import time

from tensors.synthetic_tensors import init_rand_cp
from examples.cpd import cpd_als, cpd_als_shared_exec
from tensorly.decomposition import parafac
from sktensor import dtensor
from sktensor import cp_als as sk_cp_als

BACKEND_TYPES = ['numpy']
size, rank = 200, 200
dim = 3
num_iter = 20
num_iter_slow = 3


def cpd_als_benchmark_numpy():
    input_tensor = init_rand_cp(dim, size, rank)

    # als
    t0 = time.time()
    cpd_als(size, rank, num_iter, input_tensor, calculate_loss=False)
    time_als = time.time() - t0
    print(f'als time is: {time_als/num_iter}')

    # dt
    t0 = time.time()
    cpd_als_shared_exec(size,
                        rank,
                        num_iter,
                        input_tensor,
                        calculate_loss=False)
    time_dt = time.time() - t0
    print(f'dt time is: {time_dt/num_iter}')

    # tensorly
    _, input_tensor_val = init_rand_cp(dim, size, rank)
    input_tensor = tl.tensor(input_tensor_val, dtype='float64')

    t0 = time.time()
    parafac(input_tensor,
            rank=rank,
            init='random',
            tol=0,
            n_iter_max=num_iter_slow,
            verbose=1)
    time_tensorly = time.time() - t0
    print(f'Tensorly time is: {time_tensorly/num_iter_slow}')

    # sktensor
    _, input_tensor_val = init_rand_cp(dim, size, rank)

    t0 = time.time()
    sk_cp_als(dtensor(input_tensor_val),
              rank=rank,
              max_iter=num_iter_slow,
              init='random')
    time_sktensor = time.time() - t0
    print(f'sktensor time is: {time_sktensor/num_iter_slow}')

    print('summary')
    print(f'als time is: {time_als/num_iter}')
    print(f'dt time is: {time_dt/num_iter}')
    print(f'Tensorly time is: {time_tensorly/num_iter_slow}')
    print(f'sktensor time is: {time_sktensor/num_iter_slow}')


cpd_als_benchmark_numpy()
