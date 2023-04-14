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
from autohoot.utils import get_all_inputs, find_topo_sort
from autohoot.graph_ops.optimal_tree import generate_optimal_tree, split_einsum, get_common_ancestor, generate_optimal_tree_w_constraint
from tests.test_utils import tree_eq
from visualizer import print_computation_graph


def test_einsum_gen(backendopt):
    for datatype in backendopt:
        N = 10
        C = ad.Variable(name="C", shape=[N, N])
        I = ad.Variable(name="I", shape=[N, N, N, N])

        output = ad.einsum('pi,qj,ijkl,rk,sl->pqrs', C, C, I, C, C)
        new_output = generate_optimal_tree(output)
        assert tree_eq(output, new_output, [C, I])
        assert len(new_output.inputs) == 2


def test_einsum_gen_w_clone():
    A = ad.Variable(name="A", shape=[2, 2])
    output = ad.einsum('ab->ba', A.clone())
    new_output = generate_optimal_tree(output)
    assert tree_eq(output, new_output, [A])


def test_einsum_gen_corner_case(backendopt):
    """
    Note: Numpy contraction path cannot find the opt path for this expression.
        It will output the same expression as the input.
    --------    E    --------
    |       |       |       |
    a       b       c       d
    |       |       |       |
    A - e - B - f - C - g - D
    |       |       |       |
    h       i       j       k
    |       |       |       |
    """
    size = 5
    A = ad.Variable(name="A", shape=[size, size, size])
    B = ad.Variable(name="B", shape=[size, size, size, size])
    C = ad.Variable(name="C", shape=[size, size, size, size])
    D = ad.Variable(name="D", shape=[size, size, size])
    E = ad.Variable(name="E", shape=[size, size, size, size])

    output = ad.einsum('aeh,bfie,cgjf,dgk,abcd->hijk', A, B, C, D, E)
    new_output = generate_optimal_tree(output)

    for node in find_topo_sort([new_output]):
        if not isinstance(node, ad.VariableNode):
            assert (len(node.inputs) == 2)


def test_einsum_gen_custom(backendopt):
    for datatype in backendopt:
        a = ad.Variable(name="a", shape=[2, 2])
        b = ad.Variable(name="b", shape=[2, 5])
        c = ad.Variable(name="c", shape=[5, 2])

        output = ad.einsum('ij,jk,kl->il', a, b, c)
        new_output = generate_optimal_tree(output, path=[(1, 2), (0, 1)])
        assert tree_eq(output, new_output, [a, b, c])


def test_einsum_gen_custom_3operands(backendopt):
    for datatype in backendopt:
        a = ad.Variable(name="a", shape=[2, 2])
        b = ad.Variable(name="b", shape=[2, 5])
        c = ad.Variable(name="c", shape=[5, 2])
        d = ad.Variable(name="d", shape=[2, 2])

        output = ad.einsum('ij,jk,kl,lm->im', a, b, c, d)
        new_output = generate_optimal_tree(output, path=[(1, 2), (0, 1, 2)])
        assert tree_eq(output, new_output, [a, b, c, d])


def test_split_einsum(backendopt):

    A = ad.Variable(name="A", shape=[2, 2])
    B = ad.Variable(name="B", shape=[2, 2])
    C = ad.Variable(name="C", shape=[2, 2])
    D = ad.Variable(name="D", shape=[2, 2])
    E = ad.Variable(name="E", shape=[2, 2])

    einsum_node = ad.einsum("ab,bc,cd,de,ef->af", A, B, C, D, E)
    split_input_nodes = [A, B]
    new_einsum = split_einsum(einsum_node, split_input_nodes)
    assert len(new_einsum.inputs) == 3  # A, B, einsum(C, D, E)
    assert tree_eq(new_einsum, einsum_node, [A, B, C, D, E])


def test_get_common_ancestor_simple(backendopt):

    A = ad.Variable(name="A", shape=[3, 2])

    X1 = ad.Variable(name="X1", shape=[3, 4, 4])
    X2 = ad.Variable(name="X2", shape=[3, 2, 2])
    """
        The network and indices positions are as follows:

             g - A
                 |
        d        e
        |        |
        X1 - b - X2
        |        |
        i        j
    """
    einsum_node = ad.einsum('ge,bdi,bej->gdij', A, X1, X2)
    opt_einsum = generate_optimal_tree(einsum_node)
    sub_einsum = get_common_ancestor(opt_einsum, einsum_node.inputs, A)

    assert sorted(get_all_inputs(sub_einsum),
                  key=lambda node: node.name) == sorted(
                      [A, X2], key=lambda node: node.name)


def test_get_common_ancestor(backendopt):

    A = ad.Variable(name="A", shape=[3, 2])

    X1 = ad.Variable(name="X1", shape=[3, 2, 2])
    X2 = ad.Variable(name="X2", shape=[3, 3, 2, 2])
    X3 = ad.Variable(name="X3", shape=[3, 2, 2])
    """
        The network and indices positions are as follows:

                      g - A
                          |
        c        d        e
        |        |        |
        X1 - a - X2 - b - X3
        |        |        |
        h        i        j
                          |
                      l - A

    """
    einsum_node = ad.einsum('lj,ge,bej,abdi,ach->cdhigl', A, A, X3, X2, X1)
    opt_einsum = generate_optimal_tree(einsum_node)
    sub_einsum = get_common_ancestor(opt_einsum, einsum_node.inputs, A)

    assert sorted(get_all_inputs(sub_einsum),
                  key=lambda node: node.name) == sorted(
                      [A, A, X3], key=lambda node: node.name)


def test_get_common_ancestor_w_inv(backendopt):

    A = ad.Variable(name="A", shape=[3, 3])
    X = ad.Variable(name="X", shape=[3, 3, 3])
    inv = ad.tensorinv(ad.einsum("ab,ac->bc", A, A), ind=1)
    einsum_node = ad.einsum('abc,ad,ce->bce', X, A, inv)
    opt_einsum = generate_optimal_tree(einsum_node)
    sub_einsum = get_common_ancestor(opt_einsum, einsum_node.inputs, A)

    # sub_einsum should be ad.einsum('ad,abc->dbc',A,X), and shouldn't include the inv node.
    assert sorted(get_all_inputs(sub_einsum),
                  key=lambda node: node.name) == sorted(
                      [A, X], key=lambda node: node.name)


def test_get_common_ancester_intermediate_leaves(backendopt):

    a = ad.Variable(name="a", shape=[2, 2])
    b = ad.Variable(name="b", shape=[2, 2])
    c = ad.einsum("ab,bc->ac", a, b)
    d = ad.einsum("ab,ab->ab", c, c)

    ancester = get_common_ancestor(d, d.inputs, c)
    assert ancester == d


def test_get_common_ancester_dup(backendopt):
    a = ad.Variable(name="a", shape=[2, 2])
    aa = ad.einsum("ab,bc->ac", a, a)
    out = ad.einsum("ab,bc->ac", aa, a)
    ancester = get_common_ancestor(out, [aa, a], a)
    assert ancester == out


def test_split_einsum_dup(backendopt):
    for datatype in backendopt:

        A = ad.Variable(name="A", shape=[2, 2])
        B = ad.Variable(name="B", shape=[2, 2])

        einsum_node = ad.einsum("ab,bc,cd->ad", A, B, B)
        split_input_nodes = [A]
        new_einsum = split_einsum(einsum_node, split_input_nodes)

        assert len(new_einsum.inputs) == 2  # A, einsum(B, B)
        assert tree_eq(new_einsum, einsum_node, [A, B])


def test_optimal_tree_w_constraint(backendopt):
    for datatype in backendopt:

        A = ad.Variable(name="A", shape=[2, 2])
        B = ad.Variable(name="B", shape=[2, 2])
        C = ad.Variable(name="C", shape=[2, 2])

        einsum_node = ad.einsum("ab,bc,cd->ad", A, B, C)
        new_einsum = generate_optimal_tree_w_constraint(einsum_node, [B, C])

        # makes sure that the opt_einsum output is not a pure transpose
        assert len(new_einsum.inputs) == 2

        assert C in new_einsum.inputs
        einsum_intermediate, = [
            n for n in new_einsum.inputs if isinstance(n, ad.EinsumNode)
        ]
        assert B in einsum_intermediate.inputs
