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

import jax
import jax.numpy as np
import autodiff as ad


def make_jaxpr(fun):
    """
    Extracting the JAXPR and the constants of the input functions.
    Note that extracting the function constant values are necessary fo the graph construction.
    Therefore this function is used rather than the jax internal make_jaxpr.
    Reference: https://jax.readthedocs.io/en/latest/notebooks/Writing_custom_interpreters_in_Jax.html#1.-Tracing-a-function
    """
    from functools import wraps
    from jax import linear_util as lu
    from jax import tree_util
    from jax import api_util
    from jax.util import safe_map
    from jax.abstract_arrays import ShapedArray
    from jax.interpreters import partial_eval as pe

    def pv_like(x):
        # ShapedArrays are abstract values that carry around
        # shape and dtype information
        aval = ShapedArray(np.shape(x), np.result_type(x))
        return pe.PartialVal.unknown(aval)

    @wraps(fun)
    def jaxpr_const_maker(*args, **kwargs):
        # Set up fun for transformation
        wrapped = lu.wrap_init(fun)
        # Flatten input args
        jax_args, in_tree = tree_util.tree_flatten((args, kwargs))
        # Transform fun to accept flat args and return a flat list result
        jaxtree_fun, out_tree = api_util.flatten_fun(wrapped, in_tree)
        # Abstract and partial-val's flat args
        pvals = safe_map(pv_like, jax_args)
        # Trace function into Jaxpr
        jaxpr, _, consts = pe.trace_to_jaxpr(jaxtree_fun, pvals)
        return jaxpr, consts

    return jaxpr_const_maker


def parse_jax_dot_general(parameters, innodes):
    """
    Parse the JAX dot_general function.

    Parameters
    ----------
    parameters: A dict containing the parameters of the dot_general function.
        parameters['dimension_numbers'] is a tuple of tuples of the form
        ((lhs_contracting_dims, rhs_contracting_dims), (lhs_batch_dims, rhs_batch_dims)).
    innodes: The input nodes for the generated einsum node.

    Returns
    -------
    An einsum node equivalent to the dot_general function.

    jax dot_general reference:
        https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.dot_general.html?highlight=dot_general#jax.lax.dot_general
    Note: the dot_general is a bit different from tensordot because it has specific batch dimensions
    """
    from utils import indices_to_subscripts
    assert len(innodes) == 2
    node_A, node_B = innodes
    dim_numbers = parameters['dimension_numbers']
    contract_dims, batch_dims = dim_numbers

    A_contract_dims, B_contract_dims = contract_dims
    A_batch_dims, B_batch_dims = batch_dims

    A_noncontract_dims = tuple(
        sorted(
            set(range(len(innodes[0].shape))) - set(A_batch_dims) -
            set(A_contract_dims)))
    B_noncontract_dims = tuple(
        sorted(
            set(range(len(innodes[1].shape))) - set(B_batch_dims) -
            set(B_contract_dims)))

    assert len(A_contract_dims) == len(B_contract_dims)
    assert len(A_batch_dims) == len(B_batch_dims)

    dim = len(A_noncontract_dims) + len(B_noncontract_dims) + len(
        A_contract_dims) + len(A_batch_dims)
    input_indices_A = list(range(len(node_A.shape)))

    index_acc = len(node_A.shape)
    input_indices_B = [0] * len(node_B.shape)

    for i in range(len(node_B.shape)):
        if i in B_noncontract_dims:
            input_indices_B[i] = index_acc
            index_acc += 1
    for i in range(len(B_contract_dims)):
        input_indices_B[B_contract_dims[i]] = input_indices_A[
            A_contract_dims[i]]
    for i in range(len(B_batch_dims)):
        # Note: this part is not tested currently. Einsum is needed for this to be activated
        input_indices_B[B_batch_dims[i]] = input_indices_A[A_batch_dims[i]]

    assert index_acc == dim

    out_indices = [
        v for (i, v) in enumerate(input_indices_A) if i in A_batch_dims
    ]
    out_indices += [
        v for (i, v) in enumerate(input_indices_A) if i in A_noncontract_dims
    ]
    out_indices += [
        v for (i, v) in enumerate(input_indices_B) if i in B_noncontract_dims
    ]

    subscripts = indices_to_subscripts([input_indices_A, input_indices_B],
                                       out_indices, dim)
    return ad.einsum(subscripts, node_A, node_B)


def parse_jax_add(parameters, innodes):
    assert len(innodes) == 2
    return innodes[0] + innodes[1]


def parse_jax_sub(parameters, innodes):
    assert len(innodes) == 2
    return innodes[0] - innodes[1]


def parse_jax_mul(parameters, innodes):
    assert len(innodes) == 2
    return innodes[0] * innodes[1]


def parse_jax_transpose(parameters, innodes):
    assert len(innodes) == 1
    return ad.transpose(innodes[0], parameters['permutation'])


def parse_jax_xla_call(parameters, innodes):
    jaxpr = parameters['call_jaxpr']
    out_list = make_graph_from_subjaxpr(jaxpr, innodes)
    # Here we assume single output einsum.
    return out_list[0]


def make_graph_from_subjaxpr(jaxpr, inputs):
    """
    Transfer one JAX subpr to an AutoHoot graph.

    Parameters
    ----------
    jaxpr: The Jax PR.
    inputs: List of ad.Node.

    Returns 
    -------
    A list of ad.Node.
    """

    node_set = {}

    # Here we assume the input node (ad.Node) follow the same sequence as jaxpr definition.
    for i, var in enumerate(jaxpr.invars):
        node_set[str(var)] = inputs[i]

    for eqn in jaxpr.eqns:
        assert len(eqn.outvars) == 1
        outname = str(eqn.outvars[0])
        innodes = [node_set[str(var)] for var in eqn.invars]
        node_set[outname] = parse_jax_operator(eqn.primitive, eqn.params,
                                               innodes)

    out_list = [node_set[str(var)] for var in jaxpr.outvars]
    return out_list


def parse_jax_operator(operator, parameters, innodes):
    """
    Transfer one JAX operator to an operation node.

    Parameters
    ----------
    operator: The input jax operator.
    parameters: The set of parameters of the jax operator.
    innodes: List of nodes representing the inputs.

    Returns 
    -------
    An output node.
    """
    #TODO: einsum is not supported at this time
    if str(operator) == "dot_general":
        return parse_jax_dot_general(parameters, innodes)
    elif str(operator) == "add":
        return parse_jax_add(parameters, innodes)
    elif str(operator) == "sub":
        return parse_jax_sub(parameters, innodes)
    elif str(operator) == "mul":
        return parse_jax_mul(parameters, innodes)
    elif str(operator) == 'xla_call':
        return parse_jax_xla_call(parameters, innodes)
    elif str(operator) == 'transpose':
        return parse_jax_transpose(parameters, innodes)
    else:
        raise Exception(f'Jax {operator} parser not implemented.')


def make_graph(func, *inputs):
    """
    Make AutoHOOT graph based on the input function and inputs.

    Parameters
    ----------
    func: The input function.
    inputs: The input tensors of the input function.

    Returns 
    -------
    out_list: The out node list.
    variable_list: The input variables list.
    """
    jaxpr, consts = make_jaxpr(func)(*inputs)

    node_set = {}
    variable_list = []

    for i, var in enumerate(jaxpr.invars):
        variable = ad.Variable(name=str(var), shape=list(inputs[i].shape))
        node_set[str(var)] = variable
        variable_list.append(variable)

    for i, const in enumerate(jaxpr.constvars):
        node_set[str(const)] = ad.Constant(name=str(const),
                                           shape=list(consts[i].shape),
                                           value=consts[i])

    for eqn in jaxpr.eqns:
        assert len(eqn.outvars) == 1
        outname = str(eqn.outvars[0])
        innodes = [node_set[str(var)] for var in eqn.invars]
        node_set[outname] = parse_jax_operator(eqn.primitive, eqn.params,
                                               innodes)

    out_list = [node_set[str(var)] for var in jaxpr.outvars]
    return out_list, variable_list
