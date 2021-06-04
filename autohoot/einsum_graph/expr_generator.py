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
import logging
from autohoot import autodiff as ad
from autohoot.utils import PseudoNode
from autohoot.einsum_graph.graph_structure import DimInfo, UF
from autohoot.einsum_graph.graph_generator import cross_einsum_connect

FORMAT = '[%(asctime)-15s %(filename)s:%(lineno)s] %(message)s'

logger = logging.getLogger('optimizer')
logging.basicConfig(format=FORMAT)
logger.setLevel(logging.DEBUG)


def rewrite_einsum_expr(einsum_node):
    """
        Rewrites the einsum expression of a node.
        Inplace update.

        Args:
            einsum_node: Allow duplicate inputs of the einsum node.

        Returns:
            uf (type: graph_ops.graph_optimizer.UF): 
            the union_find set of the input
        
    """
    assert (isinstance(einsum_node, ad.EinsumNode))
    input_nodes = einsum_node.inputs

    # TODO: Get all the einsum nodes in the computation graph.
    # Note that the order matters!

    pseudo_nodes = []
    # Here einsum node has a temporary name so that the character assignment
    # order is consistent.
    einsum_node_dims_info = [
        DimInfo(node=einsum_node, dim_index=i, temp_node_name='_temp_einsum')
        for i in range(len(einsum_node.shape))
    ]
    pseudo_nodes.append(
        PseudoNode(node=einsum_node, dims_info=einsum_node_dims_info))

    for k, node in enumerate(einsum_node.inputs):
        dims_info = [
            DimInfo(node=node, dim_index=i, node_index=k)
            for i in range(len(node.shape))
        ]
        pseudo_nodes.append(PseudoNode(node=node, dims_info=dims_info))

    all_dims_info = sum([node.dims_info for node in pseudo_nodes], [])

    # For any two dims with the same literal, get their pos and connect.
    uf = UF(all_dims_info)
    cross_einsum_connect(uf, einsum_node, all_dims_info)

    uf.assign()
    # Assign literals
    for node in pseudo_nodes:
        node.generate_subscript(uf)

    einsum_node_subscript = pseudo_nodes[0].subscript

    # Remove the einsum node.
    pseudo_nodes.pop(0)

    # Sort based on both the node name and subscript.
    pseudo_nodes = sorted(pseudo_nodes,
                          key=lambda pnode: pnode.node.name + pnode.subscript)

    new_input_subs = [pnode.subscript for pnode in pseudo_nodes]
    new_subscripts = ",".join(new_input_subs) + "->" + einsum_node_subscript
    einsum_node.einsum_subscripts = new_subscripts
    einsum_node.set_inputs([pnode.node for pnode in pseudo_nodes])
    logger.info(f"Rewrite to new subscript: {new_subscripts}")

    return uf