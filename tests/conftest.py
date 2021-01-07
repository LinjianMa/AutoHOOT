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
"""
Configuration file. Must resides in tests root directory.
"""
import pytest


def pytest_addoption(parser):
    parser.addoption("--backendopt",
                     action="store",
                     nargs='+',
                     default=["numpy", "jax", "ctf", "tensorflow", "taco"],
                     help="A list of backends.")


@pytest.fixture
def backendopt(request):
    return request.config.getoption("--backendopt")
