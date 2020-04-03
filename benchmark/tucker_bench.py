import autodiff as ad
import backend as T
import tensorly as tl
import time

from tensors.synthetic_tensors import init_rand_tucker
from examples.tucker import tucker_als, tucker_als_shared_exec
from tensorly.decomposition import tucker as tucker_tensorly
from sktensor import dtensor
from sktensor.tucker import hooi as sk_tucker

BACKEND_TYPES = ['numpy']
size, rank = 400, 80
dim = 3
num_iter = 8
num_iter_slow = 2


def tucker_als_benchmark_numpy():
    input_val = init_rand_tucker(dim, size, rank)

    # als
    t0 = time.time()
    tucker_als(dim, size, rank, num_iter, input_val, calculate_loss=False)
    time_als = time.time() - t0
    print(f'als time is: {time_als/num_iter}')

    # dt
    t0 = time.time()
    tucker_als_shared_exec(dim,
                           size,
                           rank,
                           num_iter,
                           input_val,
                           calculate_loss=False)
    time_dt = time.time() - t0
    print(f'dt time is: {time_dt/num_iter}')

    # tensorly
    _, _, input_tensor_val = init_rand_tucker(dim, size, rank)
    print(input_tensor_val.shape)
    input_tensor = tl.tensor(input_tensor_val, dtype='float64')

    t0 = time.time()
    tucker_tensorly(input_tensor,
                    rank=rank,
                    init='random',
                    tol=0,
                    n_iter_max=num_iter_slow,
                    verbose=1)
    time_tensorly = time.time() - t0
    print(f'Tensorly time is: {time_tensorly/num_iter_slow}')

    # sktensor
    _, _, input_tensor_val = init_rand_tucker(dim, size, rank)
    print(input_tensor_val.shape)
    t0 = time.time()
    sk_tucker(dtensor(input_tensor_val),
              rank=[rank, rank, rank],
              maxIter=num_iter_slow,
              init='random')
    time_sktensor = time.time() - t0
    print(f'sktensor time is: {time_sktensor/num_iter_slow}')

    print('summary')
    print(f'als time is: {time_als/num_iter}')
    print(f'dt time is: {time_dt/num_iter}')
    print(f'Tensorly time is: {time_tensorly/num_iter_slow}')
    print(f'sktensor time is: {time_sktensor/num_iter_slow}')


tucker_als_benchmark_numpy()
