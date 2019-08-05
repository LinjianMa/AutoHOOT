import autodiff as ad
from utils import find_topo_sort, inner_product


def invert_dict(d):
    return dict((v,k) for k,v in d.items())


def subsource_forward(output_node):
    topo_order = find_topo_sort([output_node])
    print(f'\n    # forward pass starts')
    mid_name = '_a'
    input_index = 0
    for node in topo_order:
        if len(node.inputs) == 0:
            print(f'    {node.name} = inputs[{input_index}]')
            input_index += 1
        else:
            print(f'    {mid_name} = {node.op.s2s_name(node.inputs, node)}')
            node.name = f'{mid_name}'
            mid_name = f'_{chr(ord(mid_name[1])+1)}'
    return mid_name


def subsource_gradients(output_node, node_list):

    mid_name = subsource_forward(output_node)

    print(f'\n    # backward pass of starts')

    node_to_grad_map = ad.gradients_map(output_node, node_list)
    grad_to_node_map = invert_dict(node_to_grad_map)
    gradient_nodes = [node_to_grad_map[node] for node in node_list]
    topo_order_gradients = find_topo_sort(gradient_nodes)

    input_index = 0
    for node in topo_order_gradients:
        if not node in node_to_grad_map.keys():
            if not node in node_to_grad_map.values():
                print(f'    {mid_name} = {node.op.s2s_name(node.inputs, node)}')
                node.name = f'{mid_name}'
                mid_name = f'_{chr(ord(mid_name[1])+1)}'
            else:
                forward_node = grad_to_node_map[node]
                print(f'    _grad{forward_node.name} = {node.op.s2s_name(node.inputs, node)}')
                node.name = f'_grad{forward_node.name}'  
    return node_to_grad_map, mid_name              


def source_forward(output_node):
    print('import backend as T\n')
    print(f'def forward_pass(inputs):')
    mid_name = subsource_forward(output_node)
    print(f'    return _{chr(ord(mid_name[1])-1)}')


def source_gradients(output_node, node_list):

    print('import backend as T\n')
    print(f'def gradients(inputs):')
    node_to_grad_map,_ = subsource_gradients(output_node, node_list)
    returned_grad_names = node_to_grad_map[node_list[0]].name
    for node in node_list[1:]:
        returned_grad_names = returned_grad_names + f', {node_to_grad_map[node].name}'
    print(f'    return [{returned_grad_names}]')


def source_hvp(output_node, node_list, vector_list):

    print('import backend as T\n')
    print(f'def hvp(inputs):')
    node_to_grad_map, mid_name = subsource_gradients(output_node, node_list)
    gradient_list = [node_to_grad_map[node] for node in node_list]

    print(f'\n    # inner product of g and v starts')
    inner_product_node = inner_product(vector_list, gradient_list)
    print(f'    _gTv = {inner_product_node.op.s2s_name(inner_product_node.inputs, inner_product_node)}')
    inner_product_node.name = '_gTv'

    grad_to_hvp_map = ad.gradients_map(inner_product_node, node_list)
    hvp_to_grad_map = invert_dict(grad_to_hvp_map)
    hvp_nodes = [grad_to_hvp_map[node] for node in node_list]
    topo_order_hvps = find_topo_sort(hvp_nodes)

    print(f'\n    # backward pass of inner product of g and v starts')
    input_index = 2
    for node in topo_order_hvps:
        if not node in grad_to_hvp_map.keys():
            if len(node.inputs) == 0:
                print(f'    {node.name} = inputs[{input_index}]')
                input_index += 1
            else:
                if not node in grad_to_hvp_map.values():
                    print(f'    {mid_name} = {node.op.s2s_name(node.inputs, node)}')
                    node.name = f'{mid_name}'
                    mid_name = f'_{chr(ord(mid_name[1])+1)}'
                else:
                    forward_node = hvp_to_grad_map[node]
                    print(f'    _gradv{forward_node.name} = {node.op.s2s_name(node.inputs, node)}')
                    node.name = f'_gradv{forward_node.name}'           

    returned_hvp_names = grad_to_hvp_map[node_list[0]].name
    for node in node_list[1:]:
        returned_hvp_names = returned_hvp_names + f', {grad_to_hvp_map[node].name}'
    print(f'    return [{returned_hvp_names}]')  



x = ad.Variable(name="x")
H = ad.Variable(name="H")
v = ad.Variable(name="v")
y = ad.sum(x * (H @ x))
grads_x = ad.gradients(y, [x])
Hv, = ad.hvp(output_node=y, node_list=[x], vector_list=[v])


source_forward(y)
source_gradients(y, [x])
source_hvp(y, [x], [v])

