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


def Variable(name, shape=None):
    """User defined variables in an expression.
        e.g. x = Variable(name = "x")
    """
    placeholder_node = placeholder()
    placeholder_node.name = name
    placeholder_node.shape = shape
    assert (shape != None)
    return placeholder_node


class Op(object):
    """Op represents operations performed on nodes."""
    def __call__(self):
        """Create a new node and associate the op object with the node.

        Returns
        -------
        The new node object.
        """
        new_node = Node()
        new_node.op = self
        return new_node

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


class AddOp(Op):
    """Op to element-wise add two nodes."""
    def __call__(self, node_A, node_B):
        assert node_A.shape == node_B.shape
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        new_node.name = "(%s+%s)" % (node_A.name, node_B.name)
        new_node.shape = node_A.shape
        return new_node

    def s2s_expr(self, inputs, node):
        """source_to_source expression: used for source generation"""
        assert len(inputs) == 2
        return "(%s + %s)" % (inputs[0].name, inputs[1].name)

    def compute(self, node, input_vals):
        """Given values of two input nodes, return result of element-wise addition."""
        assert len(input_vals) == 2
        # Don't allow broadcast.
        assert input_vals[0].shape == input_vals[1].shape
        return input_vals[0] + input_vals[1]

    def transposed_vjp(self, node, output_grad):
        return [output_grad, output_grad]


class AddByConstOp(Op):
    """Op to element-wise add a nodes by a constant."""
    def __call__(self, node_A, const_val):
        new_node = Op.__call__(self)
        new_node.const_attr = const_val
        new_node.inputs = [node_A]
        new_node.name = "(%s+%s)" % (node_A.name, str(const_val))
        new_node.shape = node_A.shape
        return new_node

    def s2s_expr(self, inputs, node):
        assert len(inputs) == 1
        return "(%s + %s)" % (inputs[0].name, node.const_attr)

    def compute(self, node, input_vals):
        """Given values of input node, return result of element-wise addition."""
        assert len(input_vals) == 1
        return input_vals[0] + node.const_attr

    def transposed_vjp(self, node, output_grad):
        return [output_grad]


class SubOp(Op):
    """Op to element-wise subtract two nodes."""
    def __call__(self, node_A, node_B):
        assert node_A.shape == node_B.shape
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        new_node.name = "(%s-%s)" % (node_A.name, node_B.name)
        new_node.shape = node_A.shape
        return new_node

    def s2s_expr(self, inputs, node):
        assert len(inputs) == 2
        return "(%s - %s)" % (inputs[0].name, inputs[1].name)

    def compute(self, node, input_vals):
        """Given values of two input nodes, return result of element-wise addition."""
        assert len(input_vals) == 2
        return input_vals[0] - input_vals[1]

    def transposed_vjp(self, node, output_grad):
        return [output_grad, -output_grad]


class SubByConstOp(Op):
    """Op to element-wise add a nodes by a constant."""
    def __call__(self, node_A, const_val):
        new_node = Op.__call__(self)
        new_node.const_attr = const_val
        new_node.inputs = [node_A]
        new_node.name = "(%s-%s)" % (node_A.name, str(const_val))
        new_node.shape = node_A.shape
        return new_node

    def s2s_expr(self, inputs, node):
        assert len(inputs) == 1
        return "(%s - %s)" % (inputs[0].name, node.const_attr)

    def compute(self, node, input_vals):
        """Given values of input node, return result of element-wise addition."""
        assert len(input_vals) == 1
        return input_vals[0] - node.const_attr

    def transposed_vjp(self, node, output_grad):
        return [output_grad]


class MulOp(Op):
    """Op to element-wise multiply two nodes."""
    def __call__(self, node_A, node_B, scalar_A=False, scalar_B=False):
        """
        if one of the input node is a scalar rather than a tensor:
            e.g. 5 * [[1,2],[3,4]],
        set scalar_A / scalar_B to be True.
        It will affect the vjp expression.
        """
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        new_node.scalar_A = scalar_A
        new_node.scalar_B = scalar_B
        new_node.name = "(%s * %s)" % (node_A.name, node_B.name)
        if scalar_A:
            new_node.shape = node_B.shape
        elif scalar_B:
            new_node.shape = node_A.shape
        else:
            assert node_A.shape == node_B.shape
            new_node.shape = node_A.shape
        return new_node

    def s2s_expr(self, inputs, node):
        assert len(inputs) == 2
        return "(%s * %s)" % (inputs[0].name, inputs[1].name)

    def compute(self, node, input_vals):
        """Given values of two input nodes, return result of element-wise multiplication."""
        assert len(input_vals) == 2
        if node.scalar_A is False and node.scalar_B is False:
            assert input_vals[0].shape == input_vals[1].shape
        if node.scalar_A:
            assert input_vals[0].shape == ()
        if node.scalar_B:
            assert input_vals[1].shape == ()
        return input_vals[0] * input_vals[1]

    def transposed_vjp(self, node, output_grad):
        if node.scalar_A is False and node.scalar_B is True:
            return [
                mul(output_grad, node.inputs[1], False, True),
                sum(output_grad * node.inputs[0])
            ]
        elif node.scalar_A is True and node.scalar_B is False:
            return [
                sum(output_grad * node.inputs[1]),
                mul(output_grad, node.inputs[0], False, True)
            ]
        else:
            return [
                output_grad * node.inputs[1],
                output_grad * node.inputs[0],
            ]


class MulByConstOp(Op):
    """Op to element-wise multiply a nodes by a constant."""
    def __call__(self, node_A, const_val):
        new_node = Op.__call__(self)
        new_node.const_attr = const_val
        new_node.inputs = [node_A]
        new_node.name = "(%s*%s)" % (node_A.name, str(const_val))
        new_node.shape = node_A.shape
        return new_node

    def s2s_expr(self, inputs, node):
        assert len(inputs) == 1
        return "(%s * %s)" % (inputs[0].name, node.const_attr)

    def compute(self, node, input_vals):
        """Given values of input node, return result of element-wise multiplication."""
        assert len(input_vals) == 1
        return input_vals[0] * node.const_attr

    def transposed_vjp(self, node, output_grad):
        return [output_grad * node.const_attr]


class PowerOp(Op):
    """Op to element-wise power a nodes by a constant."""
    def __call__(self, node_A, const_val):
        new_node = Op.__call__(self)
        new_node.const_attr = const_val
        new_node.inputs = [node_A]
        new_node.name = "T.power(%s, %s)" % (node_A.name, str(const_val))
        new_node.shape = node_A.shape
        return new_node

    def s2s_expr(self, inputs, node):
        assert len(inputs) == 1
        return "T.power(%s, %s)" % (inputs[0].name, node.const_attr)

    def compute(self, node, input_vals):
        """Given values of input node, return result of element-wise multiplication."""
        assert len(input_vals) == 1
        return T.power(input_vals[0], node.const_attr)

    def transposed_vjp(self, node, output_grad):
        return [
            output_grad * node.const_attr *
            power(node.inputs[0], node.const_attr - 1)
        ]


class MatMulOp(Op):
    """Op to matrix multiply two nodes."""
    def __call__(self, node_A, node_B):
        """Create a new node that is the result a matrix multiple of two input nodes.

        Parameters
        ----------
        node_A: lhs of matrix multiply
        node_B: rhs of matrix multiply

        Returns
        -------
        Returns a node that is the result a matrix multiple of two input nodes.
        """
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        new_node.name = "T.dot(%s, %s)" % (node_A.name, node_B.name)

        # when both are matrices
        if len(node_A.shape) == 2 and len(node_B.shape) == 2:
            assert node_A.shape[1] == node_B.shape[0]
            new_node.shape = [node_A.shape[0], node_B.shape[1]]
        # vector matmul matrix
        elif len(node_A.shape) == 1 and len(node_B.shape) == 2:
            if node_A.shape[0] == node_B.shape[0]:
                new_node.shape = [node_B.shape[1]]
            # the case of outer product
            elif node_B.shape[0] == 1:
                new_node.shape = [node_A.shape[0], node_B.shape[1]]
        # matrix matmul vector
        elif len(node_A.shape) == 2 and len(node_B.shape) == 1:
            assert node_A.shape[1] == node_B.shape[0]
            new_node.shape = [node_A.shape[0]]
        # inner product
        else:
            assert node_A.shape[0] == node_B.shape[0]
            new_node.shape = [1]
        return new_node

    def s2s_expr(self, inputs, node):
        assert len(inputs) == 2
        return "T.dot(%s, %s)" % (inputs[0].name, inputs[1].name)

    def compute(self, node, input_vals):
        """Given values of input nodes, return result of matrix multiplication."""
        assert len(input_vals) == 2
        assert T.is_tensor(input_vals[0])
        assert T.is_tensor(input_vals[1])
        return T.dot(input_vals[0], input_vals[1])

    def transposed_vjp(self, node, output_grad):
        """Given vjp of multiply node, return vjp contributions to each input.

        Useful formula: if Y=AB, then dA=dY B^T, dB=A^T dY
        """
        grad_A = matmul(output_grad, transpose(node.inputs[1]))
        grad_B = matmul(transpose(node.inputs[0]), output_grad)
        return [grad_A, grad_B]


class EinsumOp(Op):
    """Op to perform einstein summation for two nodes."""
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

    def __call__(self, subscripts, *nodes):
        """Create a new node that is the result a matrix multiple of two input nodes.

        Parameters
        ----------
        nodes: arbitary number of nodes

        Returns
        -------
        Returns a node that is the result of einsum.
        """
        new_node = Op.__call__(self)
        new_node.einsum_subscripts = subscripts
        new_node.inputs = list(nodes)
        node_names = [node.name for node in nodes]
        new_node.name = self._name_generator(subscripts, node_names)
        new_node.shape = self._output_shape(subscripts, nodes)
        return new_node

    def s2s_expr(self, inputs, node):
        input_names = [inputvar.name for inputvar in inputs]
        return self._name_generator(node.einsum_subscripts, input_names)

    def compute(self, node, input_vals):
        """Given values of input nodes, return result of matrix multiplication."""
        for val in input_vals:
            assert T.is_tensor(val)
        return T.einsum(node.einsum_subscripts, *input_vals)

    def _output_shape(self, subscripts, nodes):
        in_shapes = []
        for node in nodes:
            in_shapes = in_shapes + node.shape
        in_subs, out_subs, _ = _parse_einsum_input((subscripts, *nodes))
        if out_subs == '':
            return [1]
        in_subs_list = list(in_subs)
        while ',' in in_subs_list:
            in_subs_list.remove(',')
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

    def transposed_vjp(self, node, output_grad):

        if len(node.inputs) > 1:
            grad_einsums = [
                self.grad_einsum(i, node, output_grad)
                for i in range(len(node.inputs))
            ]
            return grad_einsums
        if len(node.inputs) == 1:
            return [einsum(node.einsum_subscripts, output_grad)]


class NormOp(Op):
    def __call__(self, node, order=2, axis=None):
        new_node = Op.__call__(self)
        new_node.order = order
        new_node.axis = axis
        new_node.inputs = [node]
        new_node.name = "T.norm(%s, %s, %s)" % (node.name, order, axis)
        if axis == None:
            new_node.shape = [1]
        else:
            remaining_axis = list(set(node.shape) - set(axis)).sort()
            new_node.shape = [node.shape[i] for i in remaining_axis]
        return new_node

    def s2s_expr(self, inputs, node):
        assert len(inputs) == 1
        return "T.norm(%s, %s, %s)" % (inputs[0].name, node.order, node.axis)

    def compute(self, node, input_vals):
        assert len(input_vals) == 1
        assert T.is_tensor(input_vals[0])
        return T.norm(input_vals[0], node.order, node.axis)

    def transposed_vjp(self, node, output_grad):
        if node.axis is not None or node.order != 2:
            raise NotImplementedError
        return [
            mul(output_grad * norm(node.inputs[0])**(-1),
                node.inputs[0],
                scalar_A=True,
                scalar_B=False)
        ]


class SumOp(Op):
    def __call__(self, node, axis=None):
        new_node = Op.__call__(self)
        new_node.axis = axis
        new_node.inputs = [node]
        new_node.name = "T.sum(%s, %s)" % (node.name, axis)
        if axis == None:
            new_node.shape = [1]
        else:
            remaining_axis = list(set(node.shape) - set(axis)).sort()
            new_node.shape = [node.shape[i] for i in remaining_axis]
        return new_node

    def s2s_expr(self, inputs, node):
        assert len(inputs) == 1
        return "T.sum(%s, %s)" % (inputs[0].name, node.axis)

    def compute(self, node, input_vals):
        assert len(input_vals) == 1
        assert T.is_tensor(input_vals[0])
        return T.sum(input_vals[0], node.axis)

    def transposed_vjp(self, node, output_grad):
        if node.axis != None:
            raise NotImplementedError
        return [
            mul(output_grad,
                oneslike(node.inputs[0]),
                scalar_A=True,
                scalar_B=False)
        ]


class TransposeOp(Op):
    def __call__(self, node):
        new_node = Op.__call__(self)
        new_node.inputs = [node]
        new_node.name = "T.transpose(%s)" % (node.name)
        if len(node.shape) == 2:
            new_node.shape = [node.shape[1], node.shape[0]]
        else:
            new_node.shape = [1, node.shape[0]]
        return new_node

    def s2s_expr(self, inputs, node):
        assert len(inputs) == 1
        return "T.transpose(%s)" % (inputs[0].name)

    def compute(self, node, input_vals):
        assert len(input_vals) == 1
        assert T.is_tensor(input_vals[0])
        return T.transpose(input_vals[0])

    def transposed_vjp(self, node, output_grad):
        return [transpose(output_grad)]


class PlaceholderOp(Op):
    """Op to feed value to a nodes."""
    def __call__(self):
        """Creates a variable node."""
        new_node = Op.__call__(self)
        return new_node

    def compute(self, node, input_vals):
        """No compute function since node value is fed directly in Executor."""
        assert False, "placeholder values provided by feed_dict"

    def transposed_vjp(self, node, output_grad):
        """No vjp function since node has no inputs."""
        return None


class ZerosLikeOp(Op):
    """Op that represents a constant T.zeros_like."""
    def __call__(self, node_A):
        """Creates a node that represents a T.zeros array of same shape as node_A."""
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = "T.zeros_like(%s)" % node_A.name
        new_node.shape = node_A.shape
        return new_node

    def s2s_expr(self, inputs, node):
        assert len(inputs) == 1
        return "T.zeros_like(%s)" % (inputs[0].name)

    def compute(self, node, input_vals):
        """Returns zeros_like of the same shape as input."""
        return T.zeros_like(input_vals[0])

    def transposed_vjp(self, node, output_grad):
        return [zeroslike(node.inputs[0])]


class OnesLikeOp(Op):
    """Op that represents a constant T.ones_like."""
    def __call__(self, node_A):
        """Creates a node that represents a T.ones array of same shape as node_A."""
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = "T.ones_like(%s)" % node_A.name
        new_node.shape = node_A.shape
        return new_node

    def s2s_expr(self, inputs, node):
        assert len(inputs) == 1
        return "T.ones_like(%s)" % (inputs[0].name)

    def compute(self, node, input_vals):
        """Returns ones_like of the same shape as input."""
        return T.ones_like(input_vals[0])

    def transposed_vjp(self, node, output_grad):
        return [zeroslike(node.inputs[0])]


class NegativeOp(Op):
    """Op that represents a constant T.ones_like."""
    def __call__(self, node_A):
        """Creates a node that represents a T.ones array of same shape as node_A."""
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = "(-%s)" % node_A.name
        new_node.shape = node_A.shape
        return new_node

    def s2s_expr(self, inputs, node):
        assert len(inputs) == 1
        return "(-%s)" % (inputs[0].name)

    def compute(self, node, input_vals):
        """Returns ones_like of the same shape as input."""
        assert (T.is_tensor(input_vals[0]))
        return -input_vals[0]

    def transposed_vjp(self, node, output_grad):
        return [-output_grad]


# Create global singletons of operators.
add = AddOp()
mul = MulOp()
sub = SubOp()
add_byconst = AddByConstOp()
mul_byconst = MulByConstOp()
sub_byconst = SubByConstOp()
matmul = MatMulOp()
placeholder = PlaceholderOp()
oneslike = OnesLikeOp()
zeroslike = ZerosLikeOp()
negative = NegativeOp()
power = PowerOp()
einsum = EinsumOp()
norm = NormOp()
sum = SumOp()
transpose = TransposeOp()


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
                result = node.op.compute(node, input_vals)
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
            input_transposed_vjp = node.op.transposed_vjp(
                node, transposed_vjp)[index]
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
    assert (len(node_list) == len(vector_list))
    list_length = len(node_list)
    # v is the intermediate variable for the first vjps pass
    v = oneslike(output_node)
    g_v = transposed_vjps(output_node, node_list, v)
    assert (len(g_v) == list_length)
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
    assert (output_node.shape == [1])
    return transposed_vjps(output_node, node_list, oneslike(output_node))


def hvp(output_node, node_list, vector_list):
    """
    Hessian-Vector Product
    """
    gradient_list = gradients(output_node, node_list)
    g_v_inner_product = inner_product(vector_list, gradient_list)

    return gradients(g_v_inner_product, node_list)
