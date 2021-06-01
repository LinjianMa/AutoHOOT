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

from setuptools import setup, find_packages

setup(
    name='autohoot',
    version='0.0.1',
    description='Automatic high-order optimization for tensors.',
    author='Linjian Ma and Jiayu Ye',
    author_email='lma16@illinois.edu',
    packages=find_packages(exclude=["examples", "tests", "tensors"]),
    python_requires='>=3.6',
    install_requires=[
        'attrs',
        'jax==0.2.8',
        'jaxlib>=0.1.59',
        'numpy',
        'tensorflow>=2.1.2',
        'ply',
        'opt_einsum',
        'sparse',
        'numba==0.51.2',
        'sympy',
    ],
    url='https://github.com/LinjianMa/AutoHOOT',
    license='Apache-2.0',
)
