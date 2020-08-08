# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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


@pytest.mark.benchmark(group="cp_als")
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


@pytest.mark.benchmark(group="cp_als")
def test_cpd_als_sktensor(benchmark):
    for datatype in BACKEND_TYPES:

        _, input_tensor_val = init_rand_cp(dim, size, rank)
        benchmark(sk_cp_als,
                  dtensor(input_tensor_val),
                  rank=rank,
                  max_iter=1,
                  init='random')


@pytest.mark.benchmark(group="cp_als")
def test_cpd_als(benchmark):
    for datatype in BACKEND_TYPES:
        input_tensor = init_rand_cp(dim, size, rank)
        outputs = benchmark(cpd_als, dim, size, rank, 1, input_tensor)


@pytest.mark.benchmark(group="cp_als")
def test_cpd_als_shared_exec(benchmark):
    for datatype in BACKEND_TYPES:
        input_tensor = init_rand_cp(dim, size, rank)
        outputs = benchmark(cpd_als_shared_exec, dim, size, rank, 1, input_tensor)
