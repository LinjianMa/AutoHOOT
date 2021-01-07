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
import autodiff as ad
import backend as T
from graph_ops.graph_transformer import linearize, simplify, optimize, distribute_tree, copy_tree, rewrite_einsum_expr, prune_identity_nodes, prune_scalar_nodes, prune_orthonormal_matmuls
from graph_ops.graph_optimizer import find_sub_einsumtree
from tests.test_utils import tree_eq, gen_dict
from utils import PseudoNode


def test_einsum_multiuse(backendopt):
    """
        An einsum graph like
        A    B   inputs 
        |\   |
        | \  |
        |  \ |
        |   C
        |  / 
        | /
        output

        will produce

        An einsum graph like
        A    B   inputs 
        |\   |
        | A1 |
        |  \ |
        A2  C
        |  / 
        | /
        output
    """

    for datatype in backendopt:
        T.set_backend(datatype)

        a = ad.Variable(name="a1", shape=[3, 2])
        b = ad.Variable(name="b", shape=[2, 3])

        c = ad.einsum('ik,kj->ij', a, b)
        output = ad.einsum('ik,ij->kj', a, c)

        feed_dict = gen_dict([a, b])

        executor = ad.Executor([output])
        out_val, = executor.run(feed_dict=feed_dict)

        linearize(output)
        executor = ad.Executor([output])
        out_new_val, = executor.run(feed_dict=feed_dict)

        assert T.array_equal(out_val, out_new_val)


def test_einsum_find_subtree_after_linearization(backendopt):
    """
        An einsum graph like
        A    B   inputs 
        |\   |
        | \  |
        |  \ |
        |   C
        |  / 
        | /
        output

        will produce

        An einsum graph like
        A    B   inputs 
        |\   |
        | A1 |
        |  \ |
        A2  C
        |  / 
        | /
        output

        The subtree inputs must then be [A1, A2, B] rather than A, B.
    """

    for datatype in backendopt:
        T.set_backend(datatype)

        a = ad.Variable(name="a1", shape=[3, 2])
        b = ad.Variable(name="b", shape=[2, 3])

        c = ad.einsum('ik,kj->ij', a, b)
        output = ad.einsum('ik,ij->kj', a, c)

        feed_dict = gen_dict([a, b])
        executor = ad.Executor([output])
        out_val, = executor.run(feed_dict=feed_dict)

        # New graph
        linearize(output)
        tree, = find_sub_einsumtree(PseudoNode(output))
        assert (len(tree[1]) == 3)


def test_linearization_multiple_same_output(backendopt):
    """
        An einsum graph like
        A      inputs
        |\
        | \
        |  \
        |   |
        |  /
        | /
        output

        will produce

        An einsum graph like
        A      inputs
        |\
        | \
        |  \
        A1  A2
        |  /
        | /
        output

        The subtree inputs must then be [A1, A2] rather than A.
    """
    x = ad.Variable(name="x", shape=[3])
    y = ad.einsum("i,i->", x, x)
    linearize(y)
    assert len(y.inputs) == 2


@pytest.mark.parametrize("dist_op", [ad.AddNode, ad.SubNode])
def test_tree_distribution(dist_op, backendopt):
    """
        [Distributive] An einsum graph like
        A    B     C  inputs 
         \   |     |
          \  |     |
           \ |     |
           A + B   |
             \     |
              \    |
               \   |
               output

        will produce

        A    C    B inputs 
         \   |\   |
          \  | \  |
           \ |  \ |
            AC   BC
             \    |
              \   |
               \  |
               output (AC + BC)
        put in workd (A + B) * C = A * C + B * C where * is an einsum node.

    """

    for datatype in backendopt:
        T.set_backend(datatype)

        a = ad.Variable(name="a", shape=[3, 2])
        b = ad.Variable(name="b", shape=[3, 2])
        c = ad.Variable(name="c", shape=[2, 3])

        add_node = dist_op(a, b)
        output = ad.einsum('ik,kj->ij', add_node, c)

        new_output = distribute_tree(output)
        assert isinstance(new_output, dist_op)

        assert tree_eq(output, new_output, [a, b, c])


@pytest.mark.parametrize("dist_op", [ad.AddNode, ad.SubNode])
def test_tree_distribution_order(dist_op, backendopt):
    """
        [Distributive]
        Test C * (A + B) = C * A + C * B
    """

    for datatype in backendopt:
        T.set_backend(datatype)

        a = ad.Variable(name="a", shape=[3, 2])
        b = ad.Variable(name="b", shape=[3, 2])
        c = ad.Variable(name="c", shape=[2, 3])

        output = ad.einsum('ik,kj->ij', c, dist_op(a, b))
        new_output = distribute_tree(output)
        assert isinstance(new_output, dist_op)

        assert tree_eq(output, new_output, [a, b, c])


@pytest.mark.parametrize("dist_op", [ad.AddNode, ad.SubNode])
def test_tree_distribution_w_add_output(dist_op, backendopt):
    """
        Test C * (A + B) + F * (D + E)
            = (C * A + C * B) + (F * D + F * E)
    """

    for datatype in backendopt:
        T.set_backend(datatype)

        a = ad.Variable(name="a", shape=[3, 3])
        b = ad.Variable(name="b", shape=[3, 3])
        c = ad.Variable(name="c", shape=[3, 3])

        d = ad.Variable(name="d", shape=[3, 3])
        e = ad.Variable(name="e", shape=[3, 3])
        f = ad.Variable(name="f", shape=[3, 3])

        out1 = ad.einsum('ik,kj->ij', c, dist_op(a, b))
        out2 = ad.einsum('ik,kj->ij', d, dist_op(e, f))
        output = dist_op(out1, out2)
        new_output = distribute_tree(output)
        assert isinstance(new_output, dist_op)
        for input_node in new_output.inputs:
            assert isinstance(input_node, dist_op)
        assert tree_eq(output, new_output, [a, b, c, d, e, f])


@pytest.mark.parametrize("dist_op", [ad.AddNode, ad.SubNode])
def test_tree_distribution_four_terms(dist_op, backendopt):
    """
        [Distributive] (A + B) * (C + D) 
        A    B     C     D   inputs 
         \   |     |    /
          \  |     |   /
           \ |     |  /
           A + B   C+D
             \     |
              \    |
               \   |
               output

        will produce
        
        AC + BD + BC + DB

    """

    for datatype in backendopt:
        T.set_backend(datatype)

        a = ad.Variable(name="a", shape=[3, 2])
        b = ad.Variable(name="b", shape=[3, 2])
        c = ad.Variable(name="c", shape=[2, 3])
        d = ad.Variable(name="d", shape=[2, 3])

        dist_nodeab = dist_op(a, b)
        dist_nodecd = dist_op(c, d)
        output = ad.einsum('ik,kj->ij', dist_nodeab, dist_nodecd)

        # Idea:
        # (A + B) * (C + D) = A * (C + D) + B * (C + D)
        # Then do A * (C + D) and B * (C + D)
        new_output = distribute_tree(output)

        assert isinstance(new_output, dist_op)
        add1, add2 = new_output.inputs
        assert isinstance(add1, dist_op)
        assert isinstance(add2, dist_op)

        assert tree_eq(output, new_output, [a, b, c, d])


@pytest.mark.parametrize("dist_op", [ad.AddNode, ad.SubNode])
def test_tree_distribution_mim(dist_op, backendopt):
    """
        [Distributive] (A + B) * G * (C + D) 

        will produce
        
        AGC + BGD + BGC + DGB

        Note that G must be in the middle.

        We do the following, 
        (A+B)*G*(C+D)
        = A*G*(C+D) + B*G*(C+D)
        = AGC + AGD + BGC + BGD

        Mim: man in the middle

    """

    for datatype in backendopt:
        if datatype == "taco":
            # '..,kk,..->..' is not supported in taco
            continue
        T.set_backend(datatype)

        a = ad.Variable(name="a", shape=[3, 2])
        b = ad.Variable(name="b", shape=[3, 2])
        g = ad.Variable(name="g", shape=[2, 2])
        c = ad.Variable(name="c", shape=[2, 3])
        d = ad.Variable(name="d", shape=[2, 3])

        add_nodeab = dist_op(a, b)
        add_nodecd = dist_op(c, d)
        output = ad.einsum('ik,kk,kj->ij', add_nodeab, g, add_nodecd)

        new_output = distribute_tree(output)

        assert isinstance(new_output, dist_op)
        for node in new_output.inputs:
            assert isinstance(node, dist_op)

        assert tree_eq(output, new_output, [a, b, c, d, g])


@pytest.mark.parametrize("dist_op", [ad.AddNode, ad.SubNode])
def test_tree_distribution_two_layers(dist_op, backendopt):
    """
        [Distributive] ((A + B) * G) * C

        will produce
        
        AGC + BGC

        Note that (A+B) * G is contracted first.
    """

    for datatype in backendopt:
        if datatype == "taco":
            # '..,kk,..->..' is not supported in taco
            continue
        T.set_backend(datatype)

        a = ad.Variable(name="a", shape=[3, 2])
        b = ad.Variable(name="b", shape=[3, 2])
        g = ad.Variable(name="g", shape=[2, 2])
        c = ad.Variable(name="c", shape=[2, 3])

        interm = ad.einsum('ik, kk->ik', dist_op(a, b), g)
        output = ad.einsum('ik,kj->ij', interm, c)

        new_output = distribute_tree(output)
        assert isinstance(new_output, dist_op)

        assert tree_eq(output, new_output, [a, b, c, g])


@pytest.mark.parametrize("dist_op", [ad.AddNode, ad.SubNode])
def test_tree_distribution_ppE(dist_op, backendopt):
    """
        [Distributive] ((A + B) + C) * G

        will produce
        
        AG + BG + CG

        Note that (A+B) has parent (A + B) + C.
    """

    for datatype in backendopt:
        if datatype == "taco":
            # '..,kk,..->..' is not supported in taco
            continue
        T.set_backend(datatype)

        a = ad.Variable(name="a", shape=[3, 2])
        b = ad.Variable(name="b", shape=[3, 2])
        c = ad.Variable(name="c", shape=[3, 2])
        g = ad.Variable(name="g", shape=[2, 2])

        output = ad.einsum('ik,kk->ik', dist_op(dist_op(a, b), c), g)

        new_output = distribute_tree(output)
        assert isinstance(new_output, dist_op)

        assert tree_eq(output, new_output, [a, b, c, g])


@pytest.mark.parametrize("dist_op", [ad.AddNode, ad.SubNode])
def test_distribute_dup(dist_op, backendopt):
    from graph_ops.graph_transformer import distribute_graph_w_linearize

    for datatype in backendopt:
        T.set_backend(datatype)

        a = ad.Variable(name="a", shape=[3, 3])
        b = ad.Variable(name="b", shape=[3, 3])
        c = ad.Variable(name="c", shape=[3, 3])

        output = ad.einsum("ab,ab->", dist_op(a, c), dist_op(a, c))
        new_output = distribute_graph_w_linearize(output)
        assert tree_eq(output, new_output, [a, c])


def test_copy_tree(backendopt):
    """
        [Copy] Test copying a tree.
    """
    for datatype in backendopt:
        T.set_backend(datatype)

        a = ad.Variable(name="a", shape=[3, 2])
        b = ad.Variable(name="b", shape=[2, 3])

        c = ad.einsum('ik,kj->ij', a, b)
        output = ad.einsum('ik,ij->kj', a, c)

        new_node = copy_tree(output)
        # The cloned variable names must be different since the clone.
        assert new_node.name != output.name


def test_rewrite_expr(backendopt):
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


def test_einsum_equal(backendopt):

    a1 = ad.Variable(name="a1", shape=[3, 2])
    a2 = ad.Variable(name="a2", shape=[2, 3])

    x = ad.einsum('ik,kj->ij', a1, a2)
    y = ad.einsum('ml,sm->sl', a2, a1)

    rewrite_einsum_expr(x)
    rewrite_einsum_expr(y)

    assert x.einsum_subscripts == y.einsum_subscripts
    assert x.inputs == y.inputs


def test_einsum_equal_repeated_transpose(backendopt):

    A = ad.Variable(name="A", shape=[3, 5])

    x = ad.einsum('or,ob->br', A, A)
    y = ad.einsum('eb,ed->bd', A, A)

    uf1 = rewrite_einsum_expr(x)
    uf2 = rewrite_einsum_expr(y)

    assert x.einsum_subscripts == y.einsum_subscripts
    assert x.inputs == y.inputs


def test_einsum_equal_repeated_transpose(backendopt):

    A = ad.Variable(name="A", shape=[3, 3])
    B = ad.Variable(name="B", shape=[3, 3])

    x = ad.einsum("ac,ba,bc->", A, A, B)
    y = ad.einsum("ba,ac,bc->", A, A, B)

    uf1 = rewrite_einsum_expr(x)
    uf2 = rewrite_einsum_expr(y)

    assert x.einsum_subscripts == y.einsum_subscripts
    assert x.inputs == y.inputs


def test_einsum_equal_uf_assign_order(backendopt):

    A = ad.Variable(name="A", shape=[3, 3])
    B = ad.Variable(name="B", shape=[3, 3])
    I = ad.identity(10)

    x = ad.einsum('pb,or,ob,pr,st->srtb', B, A, A, B, I)
    y = ad.einsum('eb,ed,fb,fd,ac->abcd', A, A, B, B, I)

    uf1 = rewrite_einsum_expr(x)
    uf2 = rewrite_einsum_expr(y)

    assert x.einsum_subscripts == y.einsum_subscripts
    assert x.inputs == y.inputs


def test_einsum_rewrite_duplicate_input(backendopt):

    a = ad.Variable(name="a", shape=[3, 2])

    x = ad.einsum('ca,cb->ab', a, a)
    y = ad.einsum('cb,ca->ab', a, a)

    rewrite_einsum_expr(x)
    rewrite_einsum_expr(y)

    assert x.einsum_subscripts == y.einsum_subscripts


def test_prune_identity(backendopt):
    for datatype in backendopt:
        T.set_backend(datatype)

        a1 = ad.Variable(name="a1", shape=[3, 3])
        a2 = ad.Variable(name="a2", shape=[3, 3])
        i1 = ad.identity(3)
        i2 = ad.identity(3)
        i3 = ad.identity(3)

        out = ad.einsum("ab,cd,ac,be,ef->abdf", a1, a2, i1, i2, i3)
        prune_identity_nodes(out)
        """
        Explanation to the einsum above:
        The identity node i1 means that a and c should be the same dim.
        we can get rid of i1 and rewrite the expr as 
        ad.einsum("ab,ad,be,ef->abdf", a1, a2, i2, i3).
        we can also combine i2 and i3 cuz e is useless. Therefore,
        we can rewrite the expr as
        ad.einsum("ab,ad,bf->abdf", a1, a2, i2).
        """
        out_expect = ad.einsum("ab,ad,bf->abdf", a1, a2, i2)
        assert len(out.inputs) == 3

        assert tree_eq(out, out_expect, [a1, a2])


def test_prune_identity_w_dup(backendopt):
    for datatype in backendopt:
        T.set_backend(datatype)

        a1 = ad.Variable(name="a1", shape=[3, 3])
        i1 = ad.identity(3)
        i2 = ad.identity(3)
        i3 = ad.identity(3)

        out = ad.einsum("ab,bc,cd,de,ef->af", a1, a1, i1, i2, i3)
        prune_identity_nodes(out)
        out_expect = ad.einsum("ab,bc->ac", a1, a1)
        assert len(out.inputs) == 2

        assert tree_eq(out, out_expect, [a1])


def test_simplify_inv_w_identity(backendopt):

    for datatype in backendopt:
        T.set_backend(datatype)

        A = ad.Variable(name="A", shape=[2, 2])

        out = ad.einsum("ab,cd->acbd", A, ad.tensorinv(ad.identity(3)))
        newout = simplify(out)

        assert isinstance(newout, ad.EinsumNode)
        assert isinstance(newout.inputs[1], ad.IdentityNode)

        assert tree_eq(out, newout, [A], tol=1e-6)


def test_simplify_inv_w_redundent_einsum(backendopt):

    for datatype in backendopt:
        T.set_backend(datatype)

        A = ad.Variable(name="A", shape=[2, 2])

        out = ad.einsum("ab,cd->abcd", A, ad.tensorinv(ad.einsum("ab->ab", A)))
        newout = simplify(out)

        inv_node = newout.inputs[1]

        assert isinstance(inv_node.inputs[0], ad.VariableNode)

        assert tree_eq(out, newout, [A], tol=1e-6)


def test_simplify_optimize_w_tail_einsum(backendopt):

    for datatype in backendopt:
        T.set_backend(datatype)

        A = ad.Variable(name="A", shape=[2, 2])

        out = ad.einsum("ab,bc->ac", A,
                        ad.einsum("ab,bc->ac", ad.identity(2), ad.identity(2)))
        newout_optimize = optimize(out)
        newout_simplify = simplify(out)

        assert newout_optimize == A
        assert newout_simplify == A


def test_simplify_symmetric_einsum_expr(backendopt):

    H = ad.Variable(name="H", shape=[2, 2], symmetry=[[0, 1]])
    x1 = ad.Variable(name="x1", shape=[2])
    x2 = ad.Variable(name="x2", shape=[2])

    inner1 = ad.einsum("ab,a,b->", H, x1, x2)
    inner2 = ad.einsum("ab,b,a->", H, x1, x2)
    out = 0.5 * inner1 + 0.5 * inner2
    newout_simplify = simplify(out)

    # ad.einsum("ab,a,b->", H, x1, x2) or ad.einsum("ab,b,a->", H, x1, x2)
    assert isinstance(newout_simplify, ad.EinsumNode)


def test_prune_scalar_nodes(backendopt):
    for datatype in backendopt:
        T.set_backend(datatype)

        a1 = ad.Variable(name="a1", shape=[3, 3])
        a2 = ad.Variable(name="a2", shape=[3, 3])
        s = ad.scalar(3.)

        out = ad.einsum("ab,,ab->ab", a1, s, a2)
        out_prune = prune_scalar_nodes(out)

        assert isinstance(out_prune, ad.MulByConstNode)
        assert tree_eq(out, out_prune, [a1, a2])


def test_prune_orthonormal_matmuls():

    a1 = ad.Matrix(name="a1", shape=[3, 3], orthonormal='column')
    a2 = ad.Matrix(name="a2", shape=[3, 3], orthonormal='column')
    out = ad.einsum("ab,cb,de,fe->acdf", a1, a1, a2, a2)
    out_prune = prune_orthonormal_matmuls(out)
    # out: einsum('dc,ba->abcd',identity(3),identity(3))
    assert isinstance(out_prune, ad.EinsumNode)
    assert len(out_prune.inputs) == 2
    for innode in out_prune.inputs:
        assert isinstance(innode, ad.IdentityNode)


def test_prune_orthonormal_chain_matmuls():

    a1 = ad.Matrix(name="a1", shape=[3, 3], orthonormal='column')
    a2 = ad.Matrix(name="a2", shape=[3, 3])
    out = ad.einsum("ab,bc,dc,de,ef,gf->ag", a2, a1, a1, a2, a1, a1)
    out_prune = prune_orthonormal_matmuls(out)
    # out: T.einsum('ad,dc,ce,eb->ab',a2,T.identity(3),a2,T.identity(3))
    assert isinstance(out_prune, ad.EinsumNode)
    assert len(out_prune.inputs) == 4
