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

import autodiff as ad
from utils import find_topo_sort, inner_product


def invert_dict(d):
    return dict((v, k) for k, v in d.items())


INDENT = "    "


def indent_line(line):
    return f'{INDENT}{line}\n'


def new_line(line):
    return f'{line}\n'


class SourceToSource():
    """Class to generate the source code
    Example usage:
        ```
        StS = SourceToSource()
        forward_str = StS.forward(y)
        gradient_str = StS.gradients(y, [x])
        hvp_str = StS.hvp(y, [x], [v])
        ```
    """
    def __init__(self):
        """Instance variables
            self.mid_name: middle variable names.
            self.input_index: index of the function input.
                e.g. x = input[input_index].
                after each assignment, input_index++ for the
                next assignment.
            self.forward_to_grad_map: a map mapping forward nodes to their grad nodes.
            self.grad_to_forward_map: a map mapping grad nodes to forward nodes.
            self.forward_to_hvp_map: a map mapping input nodes (forward+grad).
                to their hvp nodes
            self.hvp_to_forward_map: a map mapping hvp nodes
                to their input nodes (forward+grad).
        """
        self.mid_name = '_a'
        self.input_index = 0
        self.forward_to_grad_map = {}
        self.grad_to_forward_map = {}
        self.forward_to_hvp_map = {}
        self.hvp_to_forward_map = {}

    def _assign_next_midname(self):
        if self.mid_name[-1] < 'z':
            self.mid_name = self.mid_name[:-1] + \
                chr(ord(self.mid_name[-1]) + 1)
        else:
            self.mid_name = self.mid_name + 'a'

    def _assign_mid_variable(self, node):
        """Assign a middle variable.
            e.g. _a = T.transpose(x)
        """
        ret_str = f'{self.mid_name} = {node.s2s_expr(node.inputs)}'
        node.name = f'{self.mid_name}'
        self._assign_next_midname()
        return ret_str

    def _assign_init_variable(self, node):
        """Assign an init variable.
            e.g. x = inputs[0]
        """
        ret_str = f'{node.name} = inputs[{self.input_index}]'
        self.input_index += 1
        return ret_str

    def _assign_grad_variable(self, node):
        """Assign a gradient variable.
            e.g. _grad_a = T.dot(_grad_b, _g)
        """
        forward_node = self.grad_to_forward_map[node]
        ret_str = f'_grad{forward_node.name} = {node.s2s_expr(node.inputs)}'
        node.name = f'_grad{forward_node.name}'
        return ret_str

    def _sub_forward(self, output_node_list):
        """Forward pass subroutine"""
        file_string = ''
        topo_order = find_topo_sort(output_node_list)
        file_string += indent_line(f'# forward pass starts')
        for node in topo_order:
            if isinstance(node, ad.VariableNode):
                file_string += indent_line(self._assign_init_variable(node))
            elif isinstance(node, ad.OpNode):
                file_string += indent_line(self._assign_mid_variable(node))
        return file_string

    def _sub_gradients(self, output_node, node_list):
        """Gradient pass subroutine."""
        file_string = ''
        file_string += self._sub_forward([output_node])
        file_string += '\n'
        file_string += indent_line('# backward pass starts')

        self.forward_to_grad_map = ad.gradients_map(output_node)
        self.grad_to_forward_map = invert_dict(self.forward_to_grad_map)
        self.gradient_list = [
            self.forward_to_grad_map[node] for node in node_list
        ]
        self.topo_order_gradients = find_topo_sort(self.gradient_list)

        for node in self.topo_order_gradients:
            if node not in self.forward_to_grad_map.keys():
                if node not in self.forward_to_grad_map.values():
                    file_string += indent_line(self._assign_mid_variable(node))
                else:
                    file_string += indent_line(
                        self._assign_grad_variable(node))
        return file_string

    def _sub_gTv(self, vector_list):
        """Subroutine of g and v inner product."""
        file_string = '\n'
        file_string += indent_line(f'# inner product of g and v starts')
        for node in vector_list:
            file_string += indent_line(self._assign_init_variable(node))
        inner_product_node = inner_product(vector_list, self.gradient_list)
        topo_order = find_topo_sort([inner_product_node])
        for node in topo_order:
            if node not in self.topo_order_gradients and \
                    node is not inner_product_node and \
                    node not in vector_list:
                file_string += self._assign_mid_variable(node)
        file_string += indent_line(
            f'_gTv = {inner_product_node.s2s_expr(inner_product_node.inputs)}')
        inner_product_node.name = '_gTv'
        return inner_product_node, file_string

    def _sub_hvp(self, inner_product_node, node_list):
        """Subroutine of hvp."""
        file_string = '\n'
        file_string += indent_line(
            f'# backward pass of inner product of g and v starts')
        self.forward_to_hvp_map = ad.gradients_map(inner_product_node)
        self.hvp_to_forward_map = invert_dict(self.forward_to_hvp_map)
        hvp_nodes = [self.forward_to_hvp_map[node] for node in node_list]
        topo_order_hvps = find_topo_sort(hvp_nodes)
        for node in topo_order_hvps:
            if node not in self.forward_to_hvp_map.keys():
                if node not in self.forward_to_hvp_map.values():
                    file_string += indent_line(self._assign_mid_variable(node))
                else:
                    forward_node = self.hvp_to_forward_map[node]
                    file_string += indent_line(
                        f'_grad2{forward_node.name} = {node.s2s_expr(node.inputs)}'
                    )
                    node.name = f'_grad2{forward_node.name}'
        return file_string

    def import_lines(self, backend):
        file_string = ''
        file_string += f'import backend as T\n'
        file_string += f'T.set_backend(\'{backend}\')\n'
        if backend == 'jax':
            file_string += f'from utils import jit_decorator\n'
            file_string += f'@jit_decorator\n'
        return file_string

    def forward(self,
                output_node_list,
                function_name='forward',
                backend='numpy'):
        """Forward pass source code generation.
        function_name: the output function name
        backend: backend or jax
        """
        self.mid_name = '_a'
        self.input_index = 0

        file_string = ''
        file_string += self.import_lines(backend)

        file_string += new_line(f'def {function_name}(inputs):')
        file_string += self._sub_forward(output_node_list)
        # return expression
        returned_names = ",".join([node.name for node in output_node_list])
        file_string += indent_line(f'return [{returned_names}]')
        return file_string

    def gradients(self, output_node, node_list, backend='numpy'):
        """Gradients source code generation."""
        file_string = ''
        self.file_string = ''
        self.mid_name = '_a'
        self.input_index = 0
        file_string += self.import_lines(backend)
        file_string += f'def gradients(inputs):\n'
        file_string += self._sub_gradients(output_node, node_list)
        # return expression
        returned_grad_names = ",".join(
            [self.forward_to_grad_map[node].name for node in node_list])
        file_string += indent_line(f'return [{returned_grad_names}]')
        return file_string

    def hvp(self, output_node, node_list, vector_list, backend='numpy'):
        """Hvp source code generation."""
        file_string = ''
        self.mid_name = '_a'
        self.input_index = 0
        file_string += self.import_lines(backend)
        file_string += f'def hvp(inputs):\n'
        file_string += self._sub_gradients(output_node, node_list)
        inner_product_node, gtv_str = self._sub_gTv(vector_list)
        file_string += gtv_str
        file_string += self._sub_hvp(inner_product_node, node_list)
        # return expression
        returned_hvp_names = ",".join(
            [self.forward_to_hvp_map[node].name for node in node_list])
        file_string += indent_line(f'return [{returned_hvp_names}]')

        return file_string
