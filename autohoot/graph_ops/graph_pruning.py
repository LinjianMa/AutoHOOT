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
import itertools
import numpy as np
from autohoot import autodiff as ad
from autohoot.utils import PseudoNode
from autohoot.einsum_graph.graph_generator import generate_einsum_info
from autohoot.einsum_graph.graph_structure import UF
from opt_einsum.parser import parse_einsum_input


def prune_identity_nodes(einsum_node):
    """
        reduce the number of identity nodes in the
        einsum_node's inputs. Inplace update.

        Args:
            einsum_node: An fused einsum node.
    """
    if not (isinstance(einsum_node, ad.EinsumNode)):
        return

    uf_str, p_outnode, p_innodes = generate_einsum_info(einsum_node)
    whole_str = p_outnode.subscript + "".join(
        [node.subscript for node in p_innodes])

    p_identity_nodes = list(
        filter(lambda pnode: isinstance(pnode.node, ad.IdentityNode),
               p_innodes))
    p_variable_nodes = [
        pnode for pnode in p_innodes if pnode not in p_identity_nodes
    ]

    # each disjoint set in uf_identity represents the indices
    # linked by identity node
    uf_identity = UF(list(whole_str))
    for pnode in p_identity_nodes:
        uf_identity.connect(pnode.subscript[0], pnode.subscript[1])

    input_indices_set, output_indices_set = set(), set()
    for pnode in p_variable_nodes:
        # replace subscripts by the root chars
        sub_list = [uf_identity.root(char) for char in pnode.subscript]
        pnode.subscript = "".join(sub_list)
        input_indices_set |= set(sub_list)

    p_updated_inputs = p_variable_nodes
    out_sub_list = []
    for i, char in enumerate(p_outnode.subscript):
        uf_root_char = uf_identity.root(char)
        if uf_root_char in output_indices_set:
            # we cannot assign the same char to two indices in the
            # output. Therefore, assign a new char, and add one
            # identity node to the inputs to show the constraint.
            new_char = uf_str.cg.getchar()
            out_sub_list.append(new_char)
            p_identity_node = PseudoNode(node=ad.identity(
                einsum_node.shape[i]),
                                         subscript=f"{uf_root_char}{new_char}")
            p_updated_inputs.append(p_identity_node)
        else:
            # directly assign the root char to the subscripts
            out_sub_list.append(uf_root_char)
            output_indices_set.add(uf_root_char)
    p_outnode.subscript = "".join(out_sub_list)

    new_input_subs = [pnode.subscript for pnode in p_updated_inputs]
    new_subscripts = ",".join(new_input_subs) + "->" + p_outnode.subscript
    einsum_node.einsum_subscripts = new_subscripts
    einsum_node.set_inputs([pnode.node for pnode in p_updated_inputs])


def prune_scalar_nodes(einsum_node):
    """
        Remove the scalar input nodes of a einsum_node.
        Args:
            einsum_node: An fused einsum node.
        Return:
            both the scalar and the pruned einsum node.
    """
    in_subs, out_subs, _ = parse_einsum_input(
        (einsum_node.einsum_subscripts, *einsum_node.inputs))
    in_subs_list = in_subs.split(',')

    new_inputs, new_input_subs, scalars = [], [], []

    for i in range(len(in_subs_list)):
        if in_subs_list[i] == "" and isinstance(einsum_node.inputs[i],
                                                ad.ScalarNode):
            scalars.append(einsum_node.inputs[i].value)
        else:
            new_inputs.append(einsum_node.inputs[i])
            new_input_subs.append(in_subs_list[i])

    scalar = np.prod(scalars)

    new_subscripts = ",".join(new_input_subs) + "->" + out_subs
    output_node = ad.einsum(new_subscripts, *new_inputs)

    if scalar == 1.:
        return output_node
    else:
        return scalar * output_node


def prune_orthonormal_matmuls(einsum_node):
    """
    Remove the matrices of a einsum_node if M @ M.T like structures exist.
    Args:
        einsum_node: An fused einsum node.
    Return:
        An optimized einsum node.
    """

    # A map from the orthonormal matrix mode to (orthonormal_index, contraction_index)
    orthonormal_indices_map = {'column': (0, 1), 'row': (1, 0)}

    _, p_outnode, p_innodes = generate_einsum_info(einsum_node)
    subs_list = [pnode.subscript
                 for pnode in p_innodes] + [p_outnode.subscript]

    ortho_pnode_map = {}
    for pnode in p_innodes:
        if isinstance(pnode.node,
                      ad.MatrixNode) and pnode.node.orthonormal != None:
            nodename = pnode.node.name
            if nodename in ortho_pnode_map:
                ortho_pnode_map[nodename].append(pnode)
            else:
                ortho_pnode_map[nodename] = [pnode]

    for pnodes in ortho_pnode_map.values():
        if len(pnodes) < 2:
            continue

        remaining_pnodes = pnodes
        pnodes_subs = list(itertools.combinations(pnodes, 2))

        for pnodes_binary_input in pnodes_subs:
            if not set(pnodes_binary_input).issubset(set(remaining_pnodes)):
                continue

            pnode_A, pnode_B = pnodes_binary_input
            o_index, c_index = orthonormal_indices_map[
                pnode_A.node.orthonormal]
            # Criteria for the pruning: the o_index of two inputs are different,
            # and the c_index only appear in these two nodes.
            c_index_is_equal = pnode_A.subscript[c_index] == pnode_B.subscript[
                c_index]
            o_index_not_equal = pnode_A.subscript[
                o_index] != pnode_B.subscript[o_index]
            if not (c_index_is_equal and o_index_not_equal):
                continue
            num_subs_w_cindex = len(
                list(
                    filter(lambda subs: pnode_A.subscript[c_index] in subs,
                           subs_list)))
            if not num_subs_w_cindex == 2:
                continue
            remaining_pnodes = [
                pnode for pnode in remaining_pnodes
                if not pnode in pnodes_binary_input
            ]
            p_innodes = [
                pnode for pnode in p_innodes
                if not pnode in pnodes_binary_input
            ]

            i_node = ad.identity(pnode_A.node.shape[o_index])
            i_subs = f"{pnode_A.subscript[o_index]}{pnode_B.subscript[o_index]}"
            p_innodes.append(PseudoNode(node=i_node, subscript=i_subs))

    new_input_subs = [pnode.subscript for pnode in p_innodes]
    new_subscripts = ",".join(new_input_subs) + "->" + p_outnode.subscript
    new_inputs = [pnode.node for pnode in p_innodes]

    return ad.einsum(new_subscripts, *new_inputs)
