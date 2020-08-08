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
