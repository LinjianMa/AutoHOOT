import autodiff as ad
import backend as T
import pytest
import tensorly as tl
import time

from tensors.synthetic_tensors import init_rand_tucker
from examples.tucker import tucker_als, tucker_als_shared_exec
from tensorly.decomposition import tucker as tucker_tensorly
from sktensor import dtensor
from sktensor.tucker import hooi as sk_tucker

BACKEND_TYPES = ['numpy']
size, rank = 800, 160
dim = 3


# @pytest.mark.benchmark(group="als")
# def test_cpd_als_tensorly(benchmark):
#     for datatype in BACKEND_TYPES:
#         tl.set_backend(datatype)
#         assert tl.get_backend() == datatype

#         _, _, _, input_tensor_val = init_rand_3d(size, rank)
#         input_tensor = tl.tensor(input_tensor_val, dtype='float64')
#         factors = benchmark(parafac,
#                             input_tensor,
#                             rank=rank,
#                             init='random',
#                             tol=0,
#                             n_iter_max=1,
#                             verbose=0)


# @pytest.mark.benchmark(group="als")
# def test_cpd_als_sktensor(benchmark):
#     for datatype in BACKEND_TYPES:

#         _, _, _, input_tensor_val = init_rand_3d(size, rank)
#         benchmark(sk_cp_als,
#                   dtensor(input_tensor_val),
#                   rank=rank,
#                   max_iter=1,
#                   init='random')


@pytest.mark.benchmark(group="tucker_als")
def test_tucker_als(benchmark):
    for datatype in BACKEND_TYPES:
        input_val = init_rand_tucker(dim, size, rank)
        outputs = benchmark(tucker_als, dim, size, rank, 10, input_val)


@pytest.mark.benchmark(group="tucker_als")
def test_tucker_als_shared_exec(benchmark):
    for datatype in BACKEND_TYPES:
        input_val = init_rand_tucker(dim, size, rank)
        outputs = benchmark(tucker_als_shared_exec, dim, size, rank, 10, input_val)


def tucker_als_benchmark():
    input_val = init_rand_tucker(dim, size, rank)
    num_iter = 5
    num_iter_slow = 5

    # als
    t0 = time.time()
    tucker_als(dim, size, rank, num_iter, input_val)
    time_als = time.time() - t0
    print(f'als time is: {time_als/num_iter}')

    # dt
    t0 = time.time()
    tucker_als_shared_exec(dim, size, rank, num_iter, input_val)
    time_dt = time.time() - t0
    print(f'dt time is: {time_dt/num_iter}')

    # # # # jax
    # # # t0 = time.time()
    # # # cpd_als_jax(size, rank, 1, input_tensor)
    # # # time_jax = time.time() - t0
    # # # print(f'jax time is: {time_jax}')

    # tensorly
    _, _, input_tensor_val = init_rand_tucker(dim, size, rank)
    print(input_tensor_val.shape)
    input_tensor = tl.tensor(input_tensor_val, dtype='float64')

    t0 = time.time()
    tucker_tensorly(input_tensor, rank=rank, init='random', tol=0, n_iter_max=num_iter_slow, verbose=1)
    time_tensorly = time.time() - t0
    print(f'Tensorly time is: {time_tensorly/num_iter_slow}')

    # sktensor
    _, _, input_tensor_val = init_rand_tucker(dim, size, rank)
    print(input_tensor_val.shape)
    t0 = time.time()
    sk_tucker(dtensor(input_tensor_val), rank=[rank, rank, rank], maxIter=num_iter_slow, init='random')
    time_sktensor = time.time() - t0
    print(f'sktensor time is: {time_sktensor/num_iter_slow}')

tucker_als_benchmark()
