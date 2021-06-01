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
from autohoot.utils import sympy_simplify


def test_add():

    A = ad.Variable(name="A", shape=[3])
    y = A + A

    inputs = [A]
    assert isinstance(sympy_simplify(y, inputs), ad.MulByConstNode)


def test_minus():

    A = ad.Variable(name="A", shape=[3])
    B = ad.Variable(name="B", shape=[3])
    y = A - (A - B)

    inputs = [A, B]
    assert sympy_simplify(y, inputs).name == 'B'


def test_einsum():

    A = ad.Variable(name="A", shape=[3, 3])
    B = ad.Variable(name="B", shape=[3, 3])
    C = ad.einsum('ij,jk->ik', A, B)

    y = C + C

    assert isinstance(sympy_simplify(y, [C]), ad.MulByConstNode)


def test_einsum():

    A = ad.Variable(name="A", shape=[3, 3])
    B = ad.Variable(name="B", shape=[3, 3])
    C = ad.einsum('ij,jk->ik', A, B)

    y = B + C + C

    sim_y = sympy_simplify(y, [B, C])
    assert isinstance(sim_y, ad.AddNode)
    assert (2 * C).name in [i.name for i in sim_y.inputs]
