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
from autohoot import backend as T
from autohoot.utils import update_variables
from tests.test_utils import tree_eq

def test_update_variables(backendopt):
    for datatype in backendopt:
        T.set_backend(datatype)

        a = ad.Variable(name="a", shape=[2, 2])
        b = ad.Variable(name="b", shape=[2, 2])
        c = ad.Variable(name="c", shape=[2, 2])

        d = ad.einsum("ab,bc->ac", a, b) + c

        a = ad.Variable(name="a", shape=[3, 3])
        b = ad.Variable(name="b", shape=[3, 3])
        c = ad.Variable(name="c", shape=[3, 3])

        update_variables([d], [a, b, c])
        assert (tree_eq(d, ad.einsum("ab,bc->ac", a, b) + c, [a, b, c]))
