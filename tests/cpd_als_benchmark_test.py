import autodiff as ad
import backend as T
import pytest
import tensorly as tl
import time

from tensors.synthetic_tensors import init_rand_3d
from examples.cpd import cpd_als, cpd_als_shared_exec, cpd_als_jax
from tensorly.decomposition import parafac
from sktensor import dtensor
from sktensor import cp_als as sk_cp_als

BACKEND_TYPES = ['numpy']
size, rank = 25, 25


@pytest.mark.benchmark(group="als")
def test_cpd_als_tensorly(benchmark):
    for datatype in BACKEND_TYPES:
        tl.set_backend(datatype)
        assert tl.get_backend() == datatype

        _, _, _, input_tensor_val = init_rand_3d(size, rank)
        input_tensor = tl.tensor(input_tensor_val, dtype='float64')
        factors = benchmark(parafac,
                            input_tensor,
                            rank=rank,
                            init='random',
                            tol=0,
                            n_iter_max=1,
                            verbose=0)


@pytest.mark.benchmark(group="als")
def test_cpd_als_sktensor(benchmark):
    for datatype in BACKEND_TYPES:

        _, _, _, input_tensor_val = init_rand_3d(size, rank)
        benchmark(sk_cp_als,
                  dtensor(input_tensor_val),
                  rank=rank,
                  max_iter=1,
                  init='random')


@pytest.mark.benchmark(group="als")
def test_cpd_als(benchmark):
    for datatype in BACKEND_TYPES:
        input_tensor = init_rand_3d(size, rank)
        outputs = benchmark(cpd_als, size, rank, 1, input_tensor)


@pytest.mark.benchmark(group="als")
def test_cpd_als_jax(benchmark):
    for datatype in BACKEND_TYPES:
        input_tensor = init_rand_3d(size, rank)
        outputs = benchmark(cpd_als_jax, size, rank, 1, input_tensor)


@pytest.mark.benchmark(group="als")
def test_cpd_als_shared_exec(benchmark):
    for datatype in BACKEND_TYPES:
        input_tensor = init_rand_3d(size, rank)
        outputs = benchmark(cpd_als_shared_exec, size, rank, 1, input_tensor)

def cpd_als_benchmark():
    input_tensor = init_rand_3d(size, rank)
    num_iter = 200
    num_iter_slow = 100

    # als
    t0 = time.time()
    cpd_als(size, rank, num_iter, input_tensor)
    time_als = time.time() - t0
    print(f'als time is: {time_als/num_iter}')

    # dt
    t0 = time.time()
    cpd_als_shared_exec(size, rank, num_iter, input_tensor)
    time_dt = time.time() - t0
    print(f'dt time is: {time_dt/num_iter}')

    # # # jax
    # # t0 = time.time()
    # # cpd_als_jax(size, rank, 1, input_tensor)
    # # time_jax = time.time() - t0
    # # print(f'jax time is: {time_jax}')

    # tensorly
    _, _, _, input_tensor_val = init_rand_3d(size, rank)
    input_tensor = tl.tensor(input_tensor_val, dtype='float64')

    t0 = time.time()
    parafac(input_tensor, rank=rank, init='random', tol=0, n_iter_max=num_iter_slow, verbose=1)
    time_tensorly = time.time() - t0
    print(f'Tensorly time is: {time_tensorly/num_iter_slow}')

    # sktensor
    _, _, _, input_tensor_val = init_rand_3d(size, rank)

    t0 = time.time()
    sk_cp_als(dtensor(input_tensor_val), rank=rank, max_iter=num_iter_slow, init='random')
    time_sktensor = time.time() - t0
    print(f'sktensor time is: {time_sktensor/num_iter_slow}')

cpd_als_benchmark()
