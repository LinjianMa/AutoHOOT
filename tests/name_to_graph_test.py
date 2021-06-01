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

from autohoot import autodiff as ad
from tests.test_utils import tree_eq
from autohoot.name_parser import AutodiffParser


def test_add():

    A = ad.Variable(name="A", shape=[3])
    B = ad.Variable(name="B", shape=[3])
    y = A + B

    assert AutodiffParser.parse(y.name, [A, B]).name == y.name


def test_minus():

    A = ad.Variable(name="A", shape=[3])
    B = ad.Variable(name="B", shape=[3])
    y = A - B

    assert AutodiffParser.parse(y.name, [A, B]).name == y.name


def test_scalar_mul():

    A = ad.Variable(name="A", shape=[3])
    y = 3.0 * A

    assert AutodiffParser.parse(y.name, [A]).name == y.name


def test_add_3():

    A = ad.Variable(name="A", shape=[3])
    B = ad.Variable(name="B", shape=[3])
    y = A + B + B

    assert AutodiffParser.parse(y.name, [A, B]).name == y.name


def test_einsum():

    A = ad.Variable(name="A", shape=[3, 2])
    B = ad.Variable(name="B", shape=[2, 3])
    y = ad.einsum('ik,kj->ij', A, B)

    assert AutodiffParser.parse(y.name, [A, B]).name == y.name


def test_einsum_mul():

    A = ad.Variable(name="A", shape=[3, 2])
    B = ad.Variable(name="B", shape=[2, 3])
    y = ad.einsum('ik,kj->ij', A, B)
    z = 2 * y

    assert AutodiffParser.parse(z.name, [A, B]).name == z.name


def test_tensorinv():

    A = ad.Variable(name="A", shape=[3, 3])
    y = ad.tensorinv(A)

    assert AutodiffParser.parse(y.name, [A]).name == y.name


def test_identity():

    A = ad.identity(3)

    assert AutodiffParser.parse(A.name, []).name == A.name
