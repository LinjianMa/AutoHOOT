import autodiff as ad
from utils import find_topo_sort, inner_product


def invert_dict(d):
    return dict((v, k) for k, v in d.items())


INDENT = "    "


class SourceToSource():
    """Class to generate the source code
    Example usage:
        ```
        StS = SourceToSource()
        StS.forward(y, file=open("example_forward.py", "w"))
        StS.gradients(y, [x], file=open("example_grad.py", "w"))
        StS.hvp(y, [x], [v], file=open("example_hvp.py", "w"))
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
            self.file: output file name.
        """
        self.mid_name = '_a'
        self.input_index = 0
        self.forward_to_grad_map = {}
        self.grad_to_forward_map = {}
        self.forward_to_hvp_map = {}
        self.hvp_to_forward_map = {}
        self.file_string = ''

    def _print_to_file(self, input):
        self.file_string += input
        self.file_string += '\n'

    def _print_to_file_w_indent(self, input):
        self.file_string += INDENT
        self.file_string += input
        self.file_string += '\n'

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
        self._print_to_file_w_indent(
            f'{self.mid_name} = {node.s2s_expr(node.inputs)}')
        node.name = f'{self.mid_name}'
        self._assign_next_midname()

    def _assign_init_variable(self, node):
        """Assign an init variable.
            e.g. x = inputs[0]
        """
        self._print_to_file_w_indent(
            f'{node.name} = inputs[{self.input_index}]')
        self.input_index += 1

    def _assign_grad_variable(self, node):
        """Assign a gradient variable.
            e.g. _grad_a = T.dot(_grad_b, _g)
        """
        forward_node = self.grad_to_forward_map[node]
        self._print_to_file_w_indent(
            f'_grad{forward_node.name} = {node.s2s_expr(node.inputs)}')
        node.name = f'_grad{forward_node.name}'

    def _sub_forward(self, output_node_list):
        """Forward pass subroutine"""
        topo_order = find_topo_sort(output_node_list)
        self._print_to_file(f'\n{INDENT}# forward pass starts')
        for node in topo_order:
            if isinstance(node, ad.VariableNode):
                self._assign_init_variable(node)
            elif isinstance(node, ad.OpNode):
                self._assign_mid_variable(node)

    def _sub_gradients(self, output_node, node_list):
        """Gradient pass subroutine."""
        self._sub_forward([output_node])
        self._print_to_file(f'\n{INDENT}# backward pass starts')

        self.forward_to_grad_map = ad.gradients_map(output_node)
        self.grad_to_forward_map = invert_dict(self.forward_to_grad_map)
        self.gradient_list = [
            self.forward_to_grad_map[node] for node in node_list
        ]
        self.topo_order_gradients = find_topo_sort(self.gradient_list)

        for node in self.topo_order_gradients:
            if node not in self.forward_to_grad_map.keys():
                if node not in self.forward_to_grad_map.values():
                    self._assign_mid_variable(node)
                else:
                    self._assign_grad_variable(node)

    def _sub_gTv(self, vector_list):
        """Subroutine of g and v inner product."""
        self._print_to_file(f'\n{INDENT}# inner product of g and v starts')
        for node in vector_list:
            self._assign_init_variable(node)
        inner_product_node = inner_product(vector_list, self.gradient_list)
        topo_order = find_topo_sort([inner_product_node])
        for node in topo_order:
            if node not in self.topo_order_gradients and \
                    node is not inner_product_node and \
                    node not in vector_list:
                self._assign_mid_variable(node)
        self._print_to_file_w_indent(
            f'_gTv = {inner_product_node.s2s_expr(inner_product_node.inputs)}')
        inner_product_node.name = '_gTv'
        return inner_product_node

    def _sub_hvp(self, inner_product_node, node_list):
        """Subroutine of hvp."""
        self._print_to_file(
            f'\n{INDENT}# backward pass of inner product of g and v starts')
        self.forward_to_hvp_map = ad.gradients_map(inner_product_node)
        self.hvp_to_forward_map = invert_dict(self.forward_to_hvp_map)
        hvp_nodes = [self.forward_to_hvp_map[node] for node in node_list]
        topo_order_hvps = find_topo_sort(hvp_nodes)
        for node in topo_order_hvps:
            if node not in self.forward_to_hvp_map.keys():
                if node not in self.forward_to_hvp_map.values():
                    self._assign_mid_variable(node)
                else:
                    forward_node = self.hvp_to_forward_map[node]
                    self._print_to_file_w_indent(
                        f'_grad2{forward_node.name} = {node.s2s_expr(node.inputs)}'
                    )
                    node.name = f'_grad2{forward_node.name}'

    def _forward_head_print(self):
        self._print_to_file(f'import backend as T\n')

    def _jax_forward_head_print(self):
        self._print_to_file(f'import jax.numpy as T\n')
        self._print_to_file(f'from utils import jit_decorator\n')
        self._print_to_file(f'@jit_decorator')

    def forward(self,
                output_node_list,
                function_name='forward',
                backend='backend'):
        """Forward pass source code generation.
        function_name: the output function name
        backend: backend or jax
        """
        self.mid_name = '_a'
        self.input_index = 0

        if backend == 'backend':
            self._forward_head_print()
        elif backend == 'jax':
            self._jax_forward_head_print()
        else:
            raise NotImplementedError

        self._print_to_file(f'def {function_name}(inputs):')
        self._sub_forward(output_node_list)
        # return expression
        returned_names = ",".join([node.name for node in output_node_list])
        self._print_to_file_w_indent(f'return [{returned_names}]')

    def gradients(self, output_node, node_list):
        """Gradients source code generation."""
        self.file_string = ''
        self.mid_name = '_a'
        self.input_index = 0
        self._print_to_file(f'import backend as T\n')
        self._print_to_file(f'def gradients(inputs):')
        self._sub_gradients(output_node, node_list)
        # return expression
        returned_grad_names = ",".join(
            [self.forward_to_grad_map[node].name for node in node_list])
        self._print_to_file_w_indent(f'return [{returned_grad_names}]')

    def hvp(self, output_node, node_list, vector_list):
        """Hvp source code generation."""
        self.file_string = ''
        self.mid_name = '_a'
        self.input_index = 0
        self._print_to_file(f'import backend as T\n')
        self._print_to_file(f'def hvp(inputs):')
        self._sub_gradients(output_node, node_list)
        inner_product_node = self._sub_gTv(vector_list)
        self._sub_hvp(inner_product_node, node_list)
        # return expression
        returned_hvp_names = ",".join(
            [self.forward_to_hvp_map[node].name for node in node_list])
        self._print_to_file_w_indent(f'return [{returned_hvp_names}]')

    def __str__(self):
        return self.file_string
