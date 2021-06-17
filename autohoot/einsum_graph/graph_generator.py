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
import copy
from opt_einsum.parser import parse_einsum_input
from autohoot import autodiff as ad
from autohoot.utils import PseudoNode
from autohoot.einsum_graph.graph_structure import DimInfo, UF


def cross_einsum_connect(uf, output_node, dims_info):
    """
        Link the literal relationship for an einsum op.
        
        Args: 
            uf: union find data structure.
            output_node: An einsum node.
            dims_info: A list of all the dimensions information including the output_node.
        
        Inputs of the einsum node can have duplicates.
    """
    assert (isinstance(output_node, ad.EinsumNode))
    # for child in output_node.inputs:
    #     assert (isinstance(child, ad.EinsumNode))

    in_subs, out_subs, _ = parse_einsum_input(
        (output_node.einsum_subscripts, *output_node.inputs))
    in_subs_list = in_subs.split(',')
    whole_str = out_subs + "".join(in_subs_list)

    record = {}

    for pos, pair in enumerate(zip(whole_str, dims_info)):
        char, litername = pair
        if char in record:
            # encode
            uf.connect(litername, record[char])
        else:
            record[char] = litername


def generate_einsum_info(einsum_node):
    """
        Generates the einsum information for easier programming.

        Args:
            einsum_node: All inputs must be unique.

        Returns:
            uf (type: graph_ops.graph_optimizer.UF): 
            the union_find set of the input

        Updates the subscript of the graph nodes affected.
        
    """
    assert (isinstance(einsum_node, ad.EinsumNode))

    pseudo_nodes = []
    einsum_node_dims_info = [
        DimInfo(node=einsum_node, dim_index=i)
        for i in range(len(einsum_node.shape))
    ]
    p_outnode = PseudoNode(node=einsum_node, dims_info=einsum_node_dims_info)
    pseudo_nodes.append(p_outnode)

    p_innodes = []
    for k, node in enumerate(einsum_node.inputs):
        dims_info = [
            DimInfo(node=node, dim_index=i, node_index=k)
            for i in range(len(node.shape))
        ]
        p_innode = PseudoNode(node=node, dims_info=dims_info)
        pseudo_nodes.append(p_innode)
        p_innodes.append(p_innode)

    all_dims_info = sum([node.dims_info for node in pseudo_nodes], [])

    # For any two dims with the same literal, get their pos and connect.
    uf = UF(all_dims_info)
    cross_einsum_connect(uf, einsum_node, all_dims_info)

    uf.assign()
    # Assign literals
    for node in pseudo_nodes:
        node.generate_subscript(uf)

    return uf, p_outnode, p_innodes


def get_disjoint_set(node):
    """
    Get the disjoint set information of the input einsum node.

    The returned set element has the following [key]:[value] structure:
    [connected_dims]:[output index].

    connected_dims: tuple of input node dims info connected by one char.
        Each element in the tuple is a DimInfo object.

    When the list of dims is connected by a contraction char, the output index will be -1.
    """
    def sort_hash(dim_info):
        return dim_info.name

    node_copy = copy.deepcopy(node)
    # this is used to normalize the output node name
    temp_out_name = "_temp_einsum"
    node_copy.name = temp_out_name

    uf, _, _ = generate_einsum_info(node_copy)
    # each set contains the dimension names connected by one char
    dset = uf.disjoint_set()
    dset_ret = {}

    for connected_dims in dset:
        if not any(temp_out_name == dim_info.node_name
                   for dim_info in connected_dims):
            # contracted char
            dset_ret[tuple(sorted(connected_dims, key=sort_hash))] = -1
        else:
            output_dim_info, = filter(
                lambda dim_info: dim_info.node_name == temp_out_name,
                connected_dims)
            # uncontracted char
            connected_dims_ret = tuple(
                sorted(list(
                    filter(
                        lambda dim_info: dim_info.node_name != temp_out_name,
                        connected_dims)),
                       key=sort_hash))
            dset_ret[connected_dims_ret] = output_dim_info.dim_index

    # Note: the value list of dset_ret will have an order of 0, ..., len(node.shape), -1.
    # It is determined because of the routine in generate_einsum_info.
    return dset_ret
