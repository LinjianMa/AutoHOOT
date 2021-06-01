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
from autohoot.graph_ops.graph_transformer import simplify, optimize, rewrite_einsum_expr


def test_rewrite_expr():
    """
        Test rewrite the einsum expression.
    """

    a1 = ad.Variable(name="a1", shape=[3, 2])
    a2 = ad.Variable(name="a2", shape=[2, 3])

    x = ad.einsum('ik,kj->ij', a1, a2)

    y = ad.einsum('sm,ml->sl', a1, a2)

    rewrite_einsum_expr(x)
    rewrite_einsum_expr(y)
    assert x.einsum_subscripts == y.einsum_subscripts


def test_einsum_equal():

    a1 = ad.Variable(name="a1", shape=[3, 2])
    a2 = ad.Variable(name="a2", shape=[2, 3])

    x = ad.einsum('ik,kj->ij', a1, a2)
    y = ad.einsum('ml,sm->sl', a2, a1)

    rewrite_einsum_expr(x)
    rewrite_einsum_expr(y)

    assert x.einsum_subscripts == y.einsum_subscripts
    assert x.inputs == y.inputs


def test_einsum_equal_repeated_transpose():

    A = ad.Variable(name="A", shape=[3, 5])

    x = ad.einsum('or,ob->br', A, A)
    y = ad.einsum('eb,ed->bd', A, A)

    uf1 = rewrite_einsum_expr(x)
    uf2 = rewrite_einsum_expr(y)

    assert x.einsum_subscripts == y.einsum_subscripts
    assert x.inputs == y.inputs


def test_einsum_equal_repeated_transpose():

    A = ad.Variable(name="A", shape=[3, 3])
    B = ad.Variable(name="B", shape=[3, 3])

    x = ad.einsum("ac,ba,bc->", A, A, B)
    y = ad.einsum("ba,ac,bc->", A, A, B)

    uf1 = rewrite_einsum_expr(x)
    uf2 = rewrite_einsum_expr(y)

    assert x.einsum_subscripts == y.einsum_subscripts
    assert x.inputs == y.inputs


def test_einsum_equal_uf_assign_order():

    A = ad.Variable(name="A", shape=[3, 3])
    B = ad.Variable(name="B", shape=[3, 3])
    I = ad.identity(10)

    x = ad.einsum('pb,or,ob,pr,st->srtb', B, A, A, B, I)
    y = ad.einsum('eb,ed,fb,fd,ac->abcd', A, A, B, B, I)

    uf1 = rewrite_einsum_expr(x)
    uf2 = rewrite_einsum_expr(y)

    assert x.einsum_subscripts == y.einsum_subscripts
    assert x.inputs == y.inputs


def test_einsum_rewrite_duplicate_input():

    a = ad.Variable(name="a", shape=[3, 2])

    x = ad.einsum('ca,cb->ab', a, a)
    y = ad.einsum('cb,ca->ab', a, a)

    rewrite_einsum_expr(x)
    rewrite_einsum_expr(y)

    assert x.einsum_subscripts == y.einsum_subscripts


def test_simplify_optimize_w_tail_einsum():

    A = ad.Variable(name="A", shape=[2, 2])

    out = ad.einsum("ab,bc->ac", A,
                    ad.einsum("ab,bc->ac", ad.identity(2), ad.identity(2)))
    newout_optimize = optimize(out)
    newout_simplify = simplify(out)

    assert newout_optimize == A
    assert newout_simplify == A


def test_simplify_symmetric_einsum_expr():

    H = ad.Variable(name="H", shape=[2, 2], symmetry=[[0, 1]])
    x1 = ad.Variable(name="x1", shape=[2])
    x2 = ad.Variable(name="x2", shape=[2])

    inner1 = ad.einsum("ab,a,b->", H, x1, x2)
    inner2 = ad.einsum("ab,b,a->", H, x1, x2)
    out = 0.5 * inner1 + 0.5 * inner2
    newout_simplify = simplify(out)

    # ad.einsum("ab,a,b->", H, x1, x2) or ad.einsum("ab,b,a->", H, x1, x2)
    assert isinstance(newout_simplify, ad.EinsumNode)
