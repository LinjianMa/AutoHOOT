import numpy as np
import backend as T
from functools import reduce
from utils import einsum_grad_subscripts, find_topo_sort, topo_sort_dfs, sum_node_list, inner_product
from numpy.core.einsumfunc import _parse_einsum_input


class Node(object):
    """Node in a computation graph."""
    def __init__(self):
        """Constructor, new node is indirectly created by Op object __call__ method.

            Instance variables
            ------------------
            self.inputs: the list of input nodes.
            self.op: the associated op object,
                e.g. add object if this node is created by adding two other nodes.
            self.const_attr: the add or multiply constant,
                e.g. self.const_attr=5 if this node is created by x+5.
            self.name: node name for debugging purposes.
        """
        self.inputs = []
        self.op = None
        self.const_attr = None
        self.name = ""
        self.shape = None

    def __neg__(self):
        return negative(self)

    def __add__(self, other):
        if isinstance(other, Node):
            new_node = add(self, other)
        else:
            new_node = add_byconst(self, other)
        return new_node

    def __sub__(self, other):
        if isinstance(other, Node):
            new_node = sub(self, other)
        else:
            new_node = sub_byconst(self, other)
        return new_node

    def __rsub__(self, other):
        if isinstance(other, Node):
            new_node = sub(other, other)
        else:
            new_node = negative(sub_byconst(self, other))
        return new_node

    def __mul__(self, other):
        if isinstance(other, Node):
            new_node = mul(self, other)
        else:
            new_node = mul_byconst(self, other)
        return new_node

    # NOTE: currently only supports pow(variable, constant).
    def __pow__(self, other):
        return power(self, other)

    def __matmul__(self, other):
        return matmul(self, other)

    # TODOs:
    # def __div__(self, other): return anp.divide(  self, other)
    # def __mod__(self, other): return anp.mod(     self, other)
    # def __eq__(self, other): return anp.equal(self, other)
    # def __ne__(self, other): return anp.not_equal(self, other)
    # def __gt__(self, other): return anp.greater(self, other)
    # def __ge__(self, other): return anp.greater_equal(self, other)
    # def __lt__(self, other): return anp.less(self, other)
    # def __le__(self, other): return anp.less_equal(self, other)
    # def __abs__(self): return anp.abs(self)

    # Allow left-hand-side add and multiply.
    __radd__ = __add__
    __rmul__ = __mul__

    def __str__(self):
        """Allow print to display node name."""
        return self.name

    __repr__ = __str__


class OpNode(Node):
    """Op represents operations performed on nodes."""
    def compute(self, node, input_vals):
        """Given values of input nodes, compute the output value.
        Parameters
        ----------
        node: node that performs the compute.
        input_vals: values of input nodes.
        Returns
        -------
        An output value of the node.
        """
        raise NotImplementedError

    def transposed_vjp(self, node, output_grad):
        """Given value of output vector-jacobian product, compute transposed vjp contributions to each input node.
        Parameters
        ----------
        node: node that performs the transposed vjp.
        output_grad: value of output transposed vjp summed from children nodes' contributions
        Returns
        -------
        A list of transposed vjp contributions to each input node respectively.
        """
        raise NotImplementedError

    def s2s_expr(self, inputs):
        raise NotImplementedError


class VariableNode(Node):
    @staticmethod
    def create(*args, **kwargs):
        return VariableNode(*args, **kwargs)

    def __init__(self, name, shape):
        super().__init__()
        self.name = name
        self.shape = shape
        assert shape != None


class AddNode(OpNode):
    """A node that add two node together"""
    @staticmethod
    def create(*args, **kwargs):
        return AddNode(*args, **kwargs)

    def __init__(self, node_A, node_B):
        assert node_A.shape == node_B.shape
        self.inputs = [node_A, node_B]
        self.name = "(%s+%s)" % (node_A.name, node_B.name)
        self.shape = node_A.shape

    def compute(self, input_vals):
        """Compute the value of the self node given the input_vals"""
        assert len(input_vals) == 2
        # Don't allow broadcast.
        assert input_vals[0].shape == input_vals[1].shape
        return input_vals[0] + input_vals[1]

    def transposed_vjp(self, output_grad):
        return [output_grad, output_grad]

    def s2s_expr(self, inputs):
        """source_to_source expression: used for source generation"""
        assert len(inputs) == 2
        return "(%s + %s)" % (inputs[0].name, inputs[1].name)


class AddByConstNode(OpNode):
    """Node to element-wise add a nodes by a constant."""
    @staticmethod
    def create(*args, **kwargs):
        return AddByConstNode(*args, **kwargs)

    def __init__(self, node_A, const_val):
        self.const_attr = const_val
        self.inputs = [node_A]
        self.name = "(%s+%s)" % (node_A.name, str(const_val))
        self.shape = node_A.shape

    def compute(self, input_vals):
        """Given values of input node, return result of element-wise addition."""
        assert len(input_vals) == 1
        return input_vals[0] + self.const_attr

    def transposed_vjp(self, output_grad):
        return [output_grad]

    def s2s_expr(self, inputs):
        assert len(inputs) == 1
        return "(%s + %s)" % (inputs[0].name, self.const_attr)


class SubNode(OpNode):
    """Node to element-wise subtract two nodes."""
    @staticmethod
    def create(*args, **kwargs):
        return SubNode(*args, **kwargs)

    def __init__(self, node_A, node_B):
        assert node_A.shape == node_B.shape
        self.inputs = [node_A, node_B]
        self.name = "(%s-%s)" % (node_A.name, node_B.name)
        self.shape = node_A.shape

    def compute(self, input_vals):
        """Given values of two input nodes, return result of element-wise addition."""
        assert len(input_vals) == 2
        return input_vals[0] - input_vals[1]

    def transposed_vjp(self, output_grad):
        return [output_grad, -output_grad]

    def s2s_expr(self, inputs):
        assert len(inputs) == 2
        return "(%s - %s)" % (inputs[0].name, inputs[1].name)


class SubByConstNode(OpNode):
    """Node to element-wise add a nodes by a constant."""
    @staticmethod
    def create(*args, **kwargs):
        return SubByConstNode(*args, **kwargs)

    def __init__(self, node_A, const_val):
        self.const_attr = const_val
        self.inputs = [node_A]
        self.name = "(%s-%s)" % (node_A.name, str(const_val))
        self.shape = node_A.shape

    def compute(self, input_vals):
        """Given values of input node, return result of element-wise addition."""
        assert len(input_vals) == 1
        return input_vals[0] - self.const_attr

    def transposed_vjp(self, output_grad):
        return [output_grad]

    def s2s_expr(self, inputs):
        assert len(inputs) == 1
        return "(%s - %s)" % (inputs[0].name, self.const_attr)


class MulNode(OpNode):
    @staticmethod
    def create(*args, **kwargs):
        return MulNode(*args, **kwargs)

    def __init__(self, node_A, node_B, scalar_A=False, scalar_B=False):
        self.inputs = [node_A, node_B]
        self.scalar_A = scalar_A
        self.scalar_B = scalar_B
        self.name = "(%s * %s)" % (node_A.name, node_B.name)
        if scalar_A:
            self.shape = node_B.shape
        elif scalar_B:
            self.shape = node_A.shape
        else:
            assert node_A.shape == node_B.shape
            self.shape = node_A.shape

    def compute(self, input_vals):
        """Given values of two input nodes, return result of element-wise multiplication."""
        assert len(input_vals) == 2
        if self.scalar_A is False and self.scalar_B is False:
            assert input_vals[0].shape == input_vals[1].shape
        if self.scalar_A:
            assert input_vals[0].shape == ()
        if self.scalar_B:
            assert input_vals[1].shape == ()
        return input_vals[0] * input_vals[1]

    def transposed_vjp(self, output_grad):
        if self.scalar_A is False and self.scalar_B is True:
            return [
                mul(output_grad, self.inputs[1], False, True),
                sum(output_grad * self.inputs[0])
            ]
        elif self.scalar_A is True and self.scalar_B is False:
            return [
                sum(output_grad * self.inputs[1]),
                mul(output_grad, self.inputs[0], False, True)
            ]
        else:
            return [
                output_grad * self.inputs[1],
                output_grad * self.inputs[0],
            ]

    def s2s_expr(self, inputs):
        assert len(inputs) == 2
        return "(%s * %s)" % (inputs[0].name, inputs[1].name)


class MulByConstNode(OpNode):
    """Node to element-wise multiply a nodes by a constant."""
    @staticmethod
    def create(*args, **kwargs):
        return MulByConstNode(*args, **kwargs)

    def __init__(self, node_A, const_val):
        self.const_attr = const_val
        self.inputs = [node_A]
        self.name = "(%s*%s)" % (node_A.name, str(const_val))
        self.shape = node_A.shape

    def compute(self, input_vals):
        """Given values of input node, return result of element-wise multiplication."""
        assert len(input_vals) == 1
        return input_vals[0] * self.const_attr

    def transposed_vjp(self, output_grad):
        return [output_grad * self.const_attr]

    def s2s_expr(self, inputs):
        assert len(inputs) == 1
        return "(%s * %s)" % (inputs[0].name, self.const_attr)


class PowerNode(OpNode):
    """Node to element-wise power a nodes by a constant."""
    @staticmethod
    def create(*args, **kwargs):
        return PowerNode(*args, **kwargs)

    def __init__(self, node_A, const_val):
        self.const_attr = const_val
        self.inputs = [node_A]
        self.name = "T.power(%s, %s)" % (node_A.name, str(const_val))
        self.shape = node_A.shape

    def compute(self, input_vals):
        """Given values of input node, return result of element-wise multiplication."""
        assert len(input_vals) == 1
        return T.power(input_vals[0], self.const_attr)

    def transposed_vjp(self, output_grad):
        return [
            output_grad * self.const_attr *
            power(self.inputs[0], self.const_attr - 1)
        ]

    def s2s_expr(self, inputs):
        assert len(inputs) == 1
        return "T.power(%s, %s)" % (inputs[0].name, self.const_attr)


class MatMulNode(OpNode):
    """Node to matrix multiply two nodes."""
    @staticmethod
    def create(*args, **kwargs):
        return MatMulNode(*args, **kwargs)

    def __init__(self, node_A, node_B):
        """Create a new node that is the result a matrix multiple of two input nodes.

        Parameters
        ----------
        node_A: lhs of matrix multiply
        node_B: rhs of matrix multiply

        Returns
        -------
        Returns a node that is the result a matrix multiple of two input nodes.
        """
        self.inputs = [node_A, node_B]
        self.name = "T.dot(%s, %s)" % (node_A.name, node_B.name)

        # when both are matrices
        if len(node_A.shape) == 2 and len(node_B.shape) == 2:
            assert node_A.shape[1] == node_B.shape[0]
            self.shape = [node_A.shape[0], node_B.shape[1]]
        # vector matmul matrix
        elif len(node_A.shape) == 1 and len(node_B.shape) == 2:
            if node_A.shape[0] == node_B.shape[0]:
                self.shape = [node_B.shape[1]]
            # the case of outer product
            elif node_B.shape[0] == 1:
                self.shape = [node_A.shape[0], node_B.shape[1]]
        # matrix matmul vector
        elif len(node_A.shape) == 2 and len(node_B.shape) == 1:
            assert node_A.shape[1] == node_B.shape[0]
            self.shape = [node_A.shape[0]]
        # inner product
        else:
            assert node_A.shape[0] == node_B.shape[0]
            self.shape = [1]

    def compute(self, input_vals):
        """Given values of input selfs, return result of matrix multiplication."""
        assert len(input_vals) == 2
        assert T.is_tensor(input_vals[0])
        assert T.is_tensor(input_vals[1])
        return T.dot(input_vals[0], input_vals[1])

    def transposed_vjp(self, output_grad):
        """Given vjp of multiply self, return vjp contributions to each input.

        Useful formula: if Y=AB, then dA=dY B^T, dB=A^T dY
        """
        grad_A = matmul(output_grad, transpose(self.inputs[1]))
        grad_B = matmul(transpose(self.inputs[0]), output_grad)
        return [grad_A, grad_B]

    def s2s_expr(self, inputs):
        assert len(inputs) == 2
        return "T.dot(%s, %s)" % (inputs[0].name, inputs[1].name)


class EinsumNode(OpNode):
    """Node to perform einstein summation for two nodes."""
    @staticmethod
    def create(*args, **kwargs):
        return EinsumNode(*args, **kwargs)

    # TODO(yejiayu): Mark function staticmethod and change callsite.
    def _name_generator(self, subscripts, names):
        """Generate the einsum name for arbitary number of var names.

        Parameters
        ----------
        names: list of strings

        Returns
        -------
        Returns a einsum expression.

        """
        name = f"T.einsum('{subscripts}',"
        name += ",".join(names)
        name += ")"
        return name

    def __init__(self, subscripts, *nodes):
        """Create a new node that is the result a matrix multiple of two input nodes.

        Parameters
        ----------
        nodes: arbitary number of nodes

        Returns
        -------
        Returns a node that is the result of einsum.
        """
        self.einsum_subscripts = subscripts
        self.inputs = list(nodes)
        node_names = [node.name for node in nodes]
        self.name = self._name_generator(subscripts, node_names)
        self.subscripts = subscripts
        self.shape = self._output_shape(subscripts, nodes)

    def compute(self, input_vals):
        """Given values of input nodes, return result of matrix multiplication."""
        for val in input_vals:
            assert T.is_tensor(val)
        return T.einsum(self.einsum_subscripts, *input_vals)

    def _output_shape(self, subscripts, nodes):
        in_shapes = []
        for node in nodes:
            in_shapes = in_shapes + node.shape
        in_subs, out_subs, _ = _parse_einsum_input((subscripts, *nodes))
        if out_subs == '':
            return [1]
        in_subs_split = in_subs.split(',')
        in_subs_list = []
        for i in in_subs_split:
            if i is not '':
                in_subs_list = in_subs_list + list(i)
            else:
                in_subs_list = in_subs_list + ['']
        out_subs_list = list(out_subs)
        out_shape = []
        for out_sub in out_subs_list:
            for index, in_sub in enumerate(in_subs_list):
                if out_sub == in_sub:
                    out_shape.append(in_shapes[index])
                    break
        return out_shape

    def grad_einsum(self, argnum_wrt, node, output_grad):
        """

        Parameters
        ----------
        argnum_wrt: The node that is taken gradient w.r.t

        Returns
        -------
        Returns a einsum node.
        
        """
        in_subs, out_subs, _ = _parse_einsum_input(
            (node.einsum_subscripts, *node.inputs))
        in_subs_list = in_subs.split(',')

        op_num = argnum_wrt
        subs_wrt = in_subs_list[op_num]

        rest_of_ops = node.inputs[:op_num] + node.inputs[op_num + 1:]

        rest_of_subs = in_subs_list[:op_num] + in_subs_list[op_num + 1:]
        # This is non naked sum version first.
        new_input_subs = ','.join([out_subs] + rest_of_subs)
        new_operands = (output_grad, ) + tuple(rest_of_ops)
        new_subscripts = new_input_subs + '->' + subs_wrt
        return einsum(new_subscripts, *new_operands)

    def transposed_vjp(self, output_grad):

        if len(self.inputs) > 1:
            grad_einsums = [
                self.grad_einsum(i, self, output_grad)
                for i in range(len(self.inputs))
            ]
            return grad_einsums
        if len(self.inputs) == 1:
            return [einsum(self.einsum_subscripts, output_grad)]

    def s2s_expr(self, inputs):
        input_names = [inputvar.name for inputvar in inputs]
        return self._name_generator(self.einsum_subscripts, input_names)


class NormNode(OpNode):
    @staticmethod
    def create(*args, **kwargs):
        return NormNode(*args, **kwargs)

    def __init__(self, node, order=2, axis=None):
        self.order = order
        self.axis = axis
        self.inputs = [node]
        self.name = "T.norm(%s, %s, %s)" % (node.name, order, axis)
        if axis == None:
            self.shape = [1]
        else:
            raise NotImplementedError

    def compute(self, input_vals):
        assert len(input_vals) == 1
        assert T.is_tensor(input_vals[0])
        return T.norm(input_vals[0], self.order, self.axis)

    def transposed_vjp(self, output_grad):
        if self.axis is not None or self.order != 2:
            raise NotImplementedError
        return [
            mul(output_grad * norm(self.inputs[0])**(-1),
                self.inputs[0],
                scalar_A=True,
                scalar_B=False)
        ]

    def s2s_expr(self, inputs):
        assert len(inputs) == 1
        return "T.norm(%s, %s, %s)" % (inputs[0].name, self.order, self.axis)


class SumNode(OpNode):
    @staticmethod
    def create(*args, **kwargs):
        return SumNode(*args, **kwargs)

    def __init__(self, node, axis=None):
        self.axis = axis
        self.inputs = [node]
        self.name = "T.sum(%s, %s)" % (node.name, axis)
        if axis == None:
            self.shape = [1]
        else:
            raise NotImplementedError

    def compute(self, input_vals):
        assert len(input_vals) == 1
        assert T.is_tensor(input_vals[0])
        return T.sum(input_vals[0], self.axis)

    def transposed_vjp(self, output_grad):
        if self.axis != None:
            raise NotImplementedError
        return [
            mul(output_grad,
                oneslike(self.inputs[0]),
                scalar_A=True,
                scalar_B=False)
        ]

    def s2s_expr(self, inputs):
        assert len(inputs) == 1
        return "T.sum(%s, %s)" % (inputs[0].name, self.axis)


class TransposeNode(OpNode):
    @staticmethod
    def create(*args, **kwargs):
        return TransposeNode(*args, **kwargs)

    def __init__(self, node):
        self.inputs = [node]
        self.name = "T.transpose(%s)" % (node.name)
        assert len(node.shape) <= 2
        if len(node.shape) == 2:
            self.shape = [node.shape[1], node.shape[0]]
        else:
            self.shape = [1, node.shape[0]]

    def compute(self, input_vals):
        assert len(input_vals) == 1
        assert T.is_tensor(input_vals[0])
        return T.transpose(input_vals[0])

    def transposed_vjp(self, output_grad):
        return [transpose(output_grad)]

    def s2s_expr(self, inputs):
        assert len(inputs) == 1
        return "T.transpose(%s)" % (inputs[0].name)


class ZerosLikeNode(OpNode):
    """Op that represents a constant T.zeros_like."""
    @staticmethod
    def create(*args, **kwargs):
        return ZerosLikeNode(*args, **kwargs)

    def __init__(self, node_A):
        """Creates a node that represents a T.zeros array of same shape as node_A."""
        self.inputs = [node_A]
        self.name = "T.zeros_like(%s)" % node_A.name
        self.shape = node_A.shape

    def compute(self, input_vals):
        """Returns zeros_like of the same shape as input."""
        return T.zeros_like(input_vals[0])

    def transposed_vjp(self, output_grad):
        return [zeroslike(self.inputs[0])]

    def s2s_expr(self, inputs):
        assert len(inputs) == 1
        return "T.zeros_like(%s)" % (inputs[0].name)


class OnesLikeNode(OpNode):
    @staticmethod
    def create(*args, **kwargs):
        return OnesLikeNode(*args, **kwargs)

    def __init__(self, node_A):
        self.inputs = [node_A]
        self.name = "T.ones_like(%s)" % node_A.name
        self.shape = node_A.shape

    def compute(self, input_vals):
        """Returns ones_like of the same shape as input."""
        return T.ones_like(input_vals[0])

    def transposed_vjp(self, output_grad):
        return [zeroslike(self.inputs[0])]

    def s2s_expr(self, inputs):
        assert len(inputs) == 1
        return "T.ones_like(%s)" % (inputs[0].name)


class NegativeNode(OpNode):
    """Node that represents negating the input node"""
    @staticmethod
    def create(*args, **kwargs):
        return NegativeNode(*args, **kwargs)

    def __init__(self, node_A):
        """Creates a node that negates node_A."""
        self.inputs = [node_A]
        self.name = "(-%s)" % node_A.name
        self.shape = node_A.shape

    def s2s_expr(self, inputs):
        assert len(inputs) == 1
        return "(-%s)" % (inputs[0].name)

    def compute(self, input_vals):
        """Returns ones_like of the same shape as input."""
        assert (T.is_tensor(input_vals[0]))
        return -input_vals[0]

    def transposed_vjp(self, output_grad):
        return [-output_grad]


# Create global singletons of operators.
Variable = VariableNode.create
add = AddNode.create
mul = MulNode.create
sub = SubNode.create
add_byconst = AddByConstNode.create
mul_byconst = MulByConstNode.create
sub_byconst = SubByConstNode.create
matmul = MatMulNode.create
oneslike = OnesLikeNode.create
zeroslike = ZerosLikeNode.create
negative = NegativeNode.create
power = PowerNode.create
einsum = EinsumNode.create
norm = NormNode.create
sum = SumNode.create
transpose = TransposeNode.create


class Executor:
    """Executor computes values for a given subset of nodes in a computation graph."""
    def __init__(self, eval_node_list):
        """
        Parameters
        ----------
        eval_node_list: list of nodes whose values need to be computed.
        """
        self.eval_node_list = eval_node_list

    def run(self, feed_dict):
        """Computes values of nodes in eval_node_list given computation graph.
        Parameters
        ----------
        feed_dict: list of variable nodes whose values are supplied by user.

        Returns
        -------
        A list of values for nodes in eval_node_list.
        """
        node_to_val_map = dict(feed_dict)
        # Traverse graph in topological sort order and compute values for all
        # nodes.
        topo_order = find_topo_sort(self.eval_node_list)
        for node in topo_order:
            if node not in node_to_val_map:
                input_vals = [node_to_val_map[val] for val in node.inputs]
                result = node.compute(input_vals)
                node_to_val_map[node] = result

        # Collect node values.
        node_val_results = [
            node_to_val_map[node] for node in self.eval_node_list
        ]
        return node_val_results


def transposed_vjps_map(output_node, node_list, input_vector):
    """
    Return:
        a map mapping input nodes to their vjp contributions.
        node_to_output_grad[node] = [\frac{node}{input_vector}]
    Used in source generation.
    """
    node_to_output_grads_list = {}
    # Special note on initializing vjp of output_node as oneslike(output_node):
    # We are really taking a derivative of the scalar reduce_sum(output_node)
    # instead of the vector output_node. But this is the common case for loss
    # function.
    node_to_output_grads_list[output_node] = [input_vector]
    # a map from node to the vjp of that node
    node_to_output_grad = {}
    # Traverse graph in reverse topological order given the output_node that
    # we are taking vjp wrt.
    reverse_topo_order = reversed(find_topo_sort([output_node]))

    for node in reverse_topo_order:
        assert node in node_to_output_grads_list
        transposed_vjp = sum_node_list(node_to_output_grads_list[node])
        node_to_output_grad[node] = transposed_vjp
        for index, input in enumerate(node.inputs):
            input_transposed_vjp = node.transposed_vjp(transposed_vjp)[index]
            if input not in node_to_output_grads_list:
                node_to_output_grads_list[input] = [input_transposed_vjp]
            else:
                node_to_output_grads_list[input].append(input_transposed_vjp)
    return node_to_output_grad


def transposed_vjps(output_node, node_list, input_vector):
    """Take vector-jacobian product of output node with respect to each node in node_list.

    Parameters
    ----------
    output_node: output node that we are taking derivative of.
    node_list: list of nodes that we are taking derivative wrt.
    input_vector: input vector in the vjps.

    Returns
    -------
    mathematically, it is calculating (v^T @ J)^T
    A list of vjp values, one for each node in node_list respectively.
    The returned list shapes are the same as the node_list shapes.

    """
    node_to_output_grad = transposed_vjps_map(output_node, node_list,
                                              input_vector)
    # Collect results for vjps requested.
    grad_node_list = [node_to_output_grad[node] for node in node_list]
    return grad_node_list


def jvps(output_node, node_list, vector_list):
    """Take jacobian-vector product of output node with respect to each node in node_list.
    Reference: https://j-towns.github.io/2017/06/12/A-new-trick.html
    Note: we can achieve jvps by two vjps.
    Mathematically:
    g(v) = vjps(v) = (v^T @ J)^T
    (vector^T @ vjps(g(v)))^T = (vector^T @ J^T)^T = J @ vector

    Parameters
    ----------
    output_node: output node that we are taking derivative of.
    node_list: list of nodes that we are taking derivative wrt.
    vector_list: list of input vectors in the jvps

    Returns
    -------
    A list of jvp values, one for each node in node_list respectively.

    """
    assert len(node_list) == len(vector_list)
    list_length = len(node_list)
    # v is the intermediate variable for the first vjps pass
    v = oneslike(output_node)
    g_v = transposed_vjps(output_node, node_list, v)
    assert len(g_v) == list_length
    transposed_vjp_g = [
        transposed_vjps(g_v[i], [v], vector_list[i])[0]
        for i in range(list_length)
    ]
    return sum_node_list(transposed_vjp_g)


def jtjvps(output_node, node_list, vector_list):
    """
    Operator for the Gauss-Newton cg step:
    return J^T @ J @ v, where J is the Jacobian matrix
    """
    jvp_result = jvps(output_node, node_list, vector_list)
    return transposed_vjps(output_node, node_list, jvp_result)


def gradients_map(output_node, node_list):
    return transposed_vjps_map(output_node, node_list, oneslike(output_node))


def gradients(output_node, node_list):
    """ NOTE: currently this function only supports the case
        when output_node is a scalar.
        In our implementation, we are actually returning the
        transposed_vjps where the vector is oneslike(output_node)
        for this function.
        Mathematically, it is equal to the gradients
        ONLY WHEN the output is a scalar.
        Therefore, this function CANNOT be used to calculate the gradients
        when output_node is not a scalar.
    """
    assert output_node.shape == [1]
    ret_nodes = transposed_vjps(output_node, node_list, oneslike(output_node))
    for (ret_node, node) in zip(ret_nodes, node_list):
        assert ret_node.shape == node.shape
    return ret_nodes


def hvp(output_node, node_list, vector_list):
    """
    Hessian-Vector Product
    """
    gradient_list = gradients(output_node, node_list)
    g_v_inner_product = inner_product(vector_list, gradient_list)

    return gradients(g_v_inner_product, node_list)
