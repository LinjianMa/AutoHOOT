import autodiff as ad
import backend as T
from graph_ops.graph_transformer import linearize, distribute_tree, copy_tree, rewrite_einsum_expr, prune_identity_nodes
from graph_ops.graph_optimizer import find_sub_einsumtree
from tests.test_utils import tree_eq, gen_dict
from graph_ops.utils import einsum_equal

BACKEND_TYPES = ['numpy', 'ctf']
BACKEND_TYPES = ['numpy']


def test_einsum_multiuse():
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

    for datatype in BACKEND_TYPES:
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


def test_einsum_find_subtree_after_linearization():
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

    for datatype in BACKEND_TYPES:
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
        tree, = find_sub_einsumtree(output)
        assert (len(tree[1]) == 3)


def test_linearization_multiple_same_output():
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


def test_tree_distribution():
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

    for datatype in BACKEND_TYPES:
        T.set_backend(datatype)

        a = ad.Variable(name="a", shape=[3, 2])
        b = ad.Variable(name="b", shape=[3, 2])
        c = ad.Variable(name="c", shape=[2, 3])

        add_node = a + b
        output = ad.einsum('ik,kj->ij', add_node, c)

        new_output = distribute_tree(output)
        assert isinstance(new_output, ad.AddNode)

        assert tree_eq(output, new_output, [a, b, c])


def test_tree_distribution_order():
    """
        [Distributive]
        Test C * (A + B) = C * A + C * B
    """

    for datatype in BACKEND_TYPES:
        T.set_backend(datatype)

        a = ad.Variable(name="a", shape=[3, 2])
        b = ad.Variable(name="b", shape=[3, 2])
        c = ad.Variable(name="c", shape=[2, 3])

        output = ad.einsum('ik,kj->ij', c, a + b)
        new_output = distribute_tree(output)
        assert isinstance(new_output, ad.AddNode)

        assert tree_eq(output, new_output, [a, b, c])


def test_tree_distribution_four_terms():
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

    for datatype in BACKEND_TYPES:
        T.set_backend(datatype)

        a = ad.Variable(name="a", shape=[3, 2])
        b = ad.Variable(name="b", shape=[3, 2])
        c = ad.Variable(name="c", shape=[2, 3])
        d = ad.Variable(name="d", shape=[2, 3])

        add_nodeab = a + b
        add_nodecd = c + d
        output = ad.einsum('ik,kj->ij', add_nodeab, add_nodecd)

        # Idea:
        # (A + B) * (C + D) = A * (C + D) + B * (C + D)
        # Then do A * (C + D) and B * (C + D)
        new_output = distribute_tree(output)

        assert isinstance(new_output, ad.AddNode)
        add1, add2 = new_output.inputs
        assert isinstance(add1, ad.AddNode)
        assert isinstance(add2, ad.AddNode)

        assert tree_eq(output, new_output, [a, b, c, d])


def test_tree_distribution_mim():
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

    for datatype in BACKEND_TYPES:
        T.set_backend(datatype)

        a = ad.Variable(name="a", shape=[3, 2])
        b = ad.Variable(name="b", shape=[3, 2])
        g = ad.Variable(name="g", shape=[2, 2])
        c = ad.Variable(name="c", shape=[2, 3])
        d = ad.Variable(name="d", shape=[2, 3])

        add_nodeab = a + b
        add_nodecd = c + d
        output = ad.einsum('ik,kk,kj->ij', add_nodeab, g, add_nodecd)

        new_output = distribute_tree(output)

        assert isinstance(new_output, ad.AddNode)
        for node in new_output.inputs:
            assert isinstance(node, ad.AddNode)

        assert tree_eq(output, new_output, [a, b, c, d, g])


def test_tree_distribution_two_layers():
    """
        [Distributive] ((A + B) * G) * C

        will produce
        
        AGC + BGC

        Note that (A+B) * G is contracted first.
    """

    for datatype in BACKEND_TYPES:
        T.set_backend(datatype)

        a = ad.Variable(name="a", shape=[3, 2])
        b = ad.Variable(name="b", shape=[3, 2])
        g = ad.Variable(name="g", shape=[2, 2])
        c = ad.Variable(name="c", shape=[2, 3])

        interm = ad.einsum('ik, kk->ik', a + b, g)
        output = ad.einsum('ik,kj->ij', interm, c)

        new_output = distribute_tree(output)
        assert isinstance(new_output, ad.AddNode)

        assert tree_eq(output, new_output, [a, b, c, g])


def test_tree_distribution_ppE():
    """
        [Distributive] ((A + B) + C) * G

        will produce
        
        AG + BG + CG

        Note that (A+B) has parent (A + B) + C.
    """

    for datatype in BACKEND_TYPES:
        T.set_backend(datatype)

        a = ad.Variable(name="a", shape=[3, 2])
        b = ad.Variable(name="b", shape=[3, 2])
        c = ad.Variable(name="c", shape=[3, 2])
        g = ad.Variable(name="g", shape=[2, 2])

        output = ad.einsum('ik,kk->ik', a + b + c, g)

        new_output = distribute_tree(output)
        assert isinstance(new_output, ad.AddNode)

        assert tree_eq(output, new_output, [a, b, c, g])


def test_copy_tree():
    """
        [Copy] Test copying a tree.
    """
    for datatype in BACKEND_TYPES:
        T.set_backend(datatype)

        a = ad.Variable(name="a", shape=[3, 2])
        b = ad.Variable(name="b", shape=[2, 3])

        c = ad.einsum('ik,kj->ij', a, b)
        output = ad.einsum('ik,ij->kj', a, c)

        new_node = copy_tree(output)
        # The cloned variable names must be different since the clone.
        assert new_node.name != output.name


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

    assert einsum_equal(x, y)


def test_prune_identity():
    for datatype in BACKEND_TYPES:
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
