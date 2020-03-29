import pytest
from tensors.synthetic_tensors import init_rand_tucker
from examples.tucker import tucker_als, tucker_als_shared_exec

BACKEND_TYPES = ['numpy']
size, rank = 100, 20
dim = 3


@pytest.mark.benchmark(group="tucker_als")
def test_tucker_als(benchmark):
    for datatype in BACKEND_TYPES:
        input_val = init_rand_tucker(dim, size, rank)
        outputs = benchmark(tucker_als, dim, size, rank, 10, input_val)


@pytest.mark.benchmark(group="tucker_als")
def test_tucker_als_shared_exec(benchmark):
    for datatype in BACKEND_TYPES:
        input_val = init_rand_tucker(dim, size, rank)
        outputs = benchmark(tucker_als_shared_exec, dim, size, rank, 10,
                            input_val)
