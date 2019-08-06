import autodiff as ad
from utils import find_topo_sort, inner_product


def invert_dict(d):
    return dict((v, k) for k, v in d.items())


class SourceToSource():
    """Class to generate the source code"""

    def __init__(self):
        self.mid_name = '_a'
        self.input_index = 0
        self.forward_to_grad_map = {}
        self.grad_to_forward_map = {}
        self.forward_to_hvp_map = {}
        self.hvp_to_forward_map = {}
        self.file = None

    def _print_to_file(self, input):
        print(input, file=self.file)

    def _assign_next_midname(self):
        if self.mid_name[-1] < 'z':
            self.mid_name = self.mid_name[:-1] + chr(ord(self.mid_name[-1])+1)
        else:
            self.mid_name = self.mid_name + 'a'

    def _get_prev_midname(self):
        if self.mid_name[-1] > 'a':
            return self.mid_name[:-1] + chr(ord(self.mid_name[-1])-1)
        else:
            return self.mid_name[:-1]

    def _assign_mid_variable(self, node):
        self._print_to_file(
            f'    {self.mid_name} = {node.op.s2s_name(node.inputs, node)}')
        node.name = f'{self.mid_name}'
        self._assign_next_midname()

    def _assign_init_variable(self, node):
        self._print_to_file(f'    {node.name} = inputs[{self.input_index}]')
        self.input_index += 1

    def _assign_grad_variable(self, node):
        forward_node = self.grad_to_forward_map[node]
        self._print_to_file(
            f'    _grad{forward_node.name} = {node.op.s2s_name(node.inputs, node)}')
        node.name = f'_grad{forward_node.name}'

    def _sub_forward(self, output_node):
        topo_order = find_topo_sort([output_node])
        self._print_to_file(f'\n    # forward pass starts')
        for node in topo_order:
            if len(node.inputs) == 0:
                self._assign_init_variable(node)
            else:
                self._assign_mid_variable(node)

    def _sub_gradients(self, output_node, node_list):
        self._sub_forward(output_node)
        self._print_to_file(f'\n    # backward pass starts')

        self.forward_to_grad_map = ad.gradients_map(output_node, node_list)
        self.grad_to_forward_map = invert_dict(self.forward_to_grad_map)
        self.gradient_list = [self.forward_to_grad_map[node]
                              for node in node_list]
        self.topo_order_gradients = find_topo_sort(self.gradient_list)

        for node in self.topo_order_gradients:
            if node not in self.forward_to_grad_map.keys():
                if node not in self.forward_to_grad_map.values():
                    self._assign_mid_variable(node)
                else:
                    self._assign_grad_variable(node)

    def _sub_gTv(self, vector_list):
        self._print_to_file(f'\n    # inner product of g and v starts')
        for node in vector_list:
            self._assign_init_variable(node)
        inner_product_node = inner_product(vector_list, self.gradient_list)
        topo_order = find_topo_sort([inner_product_node])
        for node in topo_order:
            if node not in self.topo_order_gradients and \
                    node is not inner_product_node and \
                    node not in vector_list:
                self._assign_mid_variable(node)
        self._print_to_file(
            f'    _gTv = {inner_product_node.op.s2s_name(inner_product_node.inputs, inner_product_node)}')
        inner_product_node.name = '_gTv'
        return inner_product_node

    def _sub_hvp(self, inner_product_node, node_list):
        self._print_to_file(
            f'\n    # backward pass of inner product of g and v starts')
        self.forward_to_hvp_map = ad.gradients_map(
            inner_product_node, node_list)
        self.hvp_to_forward_map = invert_dict(self.forward_to_hvp_map)
        hvp_nodes = [self.forward_to_hvp_map[node] for node in node_list]
        topo_order_hvps = find_topo_sort(hvp_nodes)
        for node in topo_order_hvps:
            if node not in self.forward_to_hvp_map.keys():
                if node not in self.forward_to_hvp_map.values():
                    self._assign_mid_variable(node)
                else:
                    forward_node = self.hvp_to_forward_map[node]
                    self._print_to_file(
                        f'    _grad2{forward_node.name} = {node.op.s2s_name(node.inputs, node)}')
                    node.name = f'_grad2{forward_node.name}'

    def forward(self, output_node, file=None):
        self.mid_name = '_a'
        self.input_index = 0
        self.file = file
        self._print_to_file('import backend as T\n')
        self._print_to_file(f'def forward(inputs):')
        self._sub_forward(output_node)
        self._print_to_file(f'    return {self._get_prev_midname()}')
        self.file.flush()

    def gradients(self, output_node, node_list, file=None):
        self.mid_name = '_a'
        self.input_index = 0
        self.file = file
        self._print_to_file('import backend as T\n')
        self._print_to_file(f'def gradients(inputs):')
        self._sub_gradients(output_node, node_list)
        returned_grad_names = self.forward_to_grad_map[node_list[0]].name
        for node in node_list[1:]:
            returned_grad_names = returned_grad_names + \
                f', {self.forward_to_grad_map[node].name}'
        self._print_to_file(f'    return [{returned_grad_names}]')
        self.file.flush()

    def hvp(self, output_node, node_list, vector_list, file=None):
        self.mid_name = '_a'
        self.input_index = 0
        self.file = file
        self._print_to_file('import backend as T\n')
        self._print_to_file(f'def hvp(inputs):')
        self._sub_gradients(output_node, node_list)

        inner_product_node = self._sub_gTv(vector_list)
        self._sub_hvp(inner_product_node, node_list)

        returned_hvp_names = self.forward_to_hvp_map[node_list[0]].name
        for node in node_list[1:]:
            returned_hvp_names = returned_hvp_names + \
                f', {self.forward_to_hvp_map[node].name}'
        self._print_to_file(f'    return [{returned_hvp_names}]')
        self.file.flush()
