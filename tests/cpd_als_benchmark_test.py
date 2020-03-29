import autodiff as ad
import backend as T
import pytest
import tensorly as tl

from tensors.synthetic_tensors import init_rand_cp
from examples.cpd import cpd_als, cpd_als_shared_exec
from tensorly.decomposition import parafac
from sktensor import dtensor
from sktensor import cp_als as sk_cp_als

BACKEND_TYPES = ['numpy']
size, rank = 150, 150
dim = 3


@pytest.mark.benchmark(group="als")
def test_cpd_als_tensorly(benchmark):
    for datatype in BACKEND_TYPES:
        tl.set_backend(datatype)
        assert tl.get_backend() == datatype

        _, input_tensor_val = init_rand_cp(dim, size, rank)
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

        _, input_tensor_val = init_rand_cp(dim, size, rank)
        benchmark(sk_cp_als,
                  dtensor(input_tensor_val),
                  rank=rank,
                  max_iter=1,
                  init='random')


@pytest.mark.benchmark(group="als")
def test_cpd_als(benchmark):
    for datatype in BACKEND_TYPES:
        input_tensor = init_rand_cp(dim, size, rank)
        outputs = benchmark(cpd_als, size, rank, 1, input_tensor)


@pytest.mark.benchmark(group="als")
def test_cpd_als_shared_exec(benchmark):
    for datatype in BACKEND_TYPES:
        input_tensor = init_rand_cp(dim, size, rank)
        outputs = benchmark(cpd_als_shared_exec, size, rank, 1, input_tensor)
