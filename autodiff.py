import numpy as np
import backend as T
from functools import reduce
from utils import einsum_grad_subscripts, find_topo_sort, topo_sort_dfs, sum_node_list


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


def Variable(name):
    """User defined variables in an expression.
        e.g. x = Variable(name = "x")
    """
    placeholder_node = placeholder()
    placeholder_node.name = name
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

    def vjp(self, node, output_grad):
        """Given value of output vector-jacobian product, compute vjp contributions to each input node.

        Parameters
        ----------
        node: node that performs the vjp.
        output_grad: value of output vjp summed from children nodes' contributions

        Returns
        -------
        A list of vjp contributions to each input node respectively.
        """
        raise NotImplementedError


class AddOp(Op):
    """Op to element-wise add two nodes."""

    def __call__(self, node_A, node_B):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        new_node.name = "(%s+%s)" % (node_A.name, node_B.name)
        return new_node

    def compute(self, node, input_vals):
        """Given values of two input nodes, return result of element-wise addition."""
        assert len(input_vals) == 2
        return input_vals[0] + input_vals[1]

    def vjp(self, node, output_grad):
        return [output_grad, output_grad]


class AddByConstOp(Op):
    """Op to element-wise add a nodes by a constant."""

    def __call__(self, node_A, const_val):
        new_node = Op.__call__(self)
        new_node.const_attr = const_val
        new_node.inputs = [node_A]
        new_node.name = "(%s+%s)" % (node_A.name, str(const_val))
        return new_node

    def compute(self, node, input_vals):
        """Given values of input node, return result of element-wise addition."""
        assert len(input_vals) == 1
        return input_vals[0] + node.const_attr

    def vjp(self, node, output_grad):
        return [output_grad]


class SubOp(Op):
    """Op to element-wise subtract two nodes."""

    def __call__(self, node_A, node_B):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        new_node.name = "(%s-%s)" % (node_A.name, node_B.name)
        return new_node

    def compute(self, node, input_vals):
        """Given values of two input nodes, return result of element-wise addition."""
        assert len(input_vals) == 2
        return input_vals[0] - input_vals[1]

    def vjp(self, node, output_grad):
        return [output_grad, -output_grad]


class SubByConstOp(Op):
    """Op to element-wise add a nodes by a constant."""

    def __call__(self, node_A, const_val):
        new_node = Op.__call__(self)
        new_node.const_attr = const_val
        new_node.inputs = [node_A]
        new_node.name = "(%s-%s)" % (node_A.name, str(const_val))
        return new_node

    def compute(self, node, input_vals):
        """Given values of input node, return result of element-wise addition."""
        assert len(input_vals) == 1
        return input_vals[0] - node.const_attr

    def vjp(self, node, output_grad):
        return [output_grad]


class MulOp(Op):
    """Op to element-wise multiply two nodes."""

    def __call__(self, node_A, node_B):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        new_node.name = "(%s*%s)" % (node_A.name, node_B.name)
        return new_node

    def compute(self, node, input_vals):
        """Given values of two input nodes, return result of element-wise multiplication."""
        assert len(input_vals) == 2
        return input_vals[0] * input_vals[1]

    def vjp(self, node, output_grad):
        return [output_grad * node.inputs[1], output_grad * node.inputs[0]]


class MulByConstOp(Op):
    """Op to element-wise multiply a nodes by a constant."""

    def __call__(self, node_A, const_val):
        new_node = Op.__call__(self)
        new_node.const_attr = const_val
        new_node.inputs = [node_A]
        new_node.name = "(%s*%s)" % (node_A.name, str(const_val))
        return new_node

    def compute(self, node, input_vals):
        """Given values of input node, return result of element-wise multiplication."""
        assert len(input_vals) == 1
        return input_vals[0] * node.const_attr

    def vjp(self, node, output_grad):
        return [output_grad * node.const_attr]


class PowerOp(Op):
    """Op to element-wise power a nodes by a constant."""

    def __call__(self, node_A, const_val):
        new_node = Op.__call__(self)
        new_node.const_attr = const_val
        new_node.inputs = [node_A]
        new_node.name = "(%s**%s)" % (node_A.name, str(const_val))
        return new_node

    def compute(self, node, input_vals):
        """Given values of input node, return result of element-wise multiplication."""
        assert len(input_vals) == 1
        return T.power(input_vals[0], node.const_attr)

    def vjp(self, node, output_grad):
        return [
            output_grad *
            node.const_attr *
            power(
                node.inputs[0],
                node.const_attr -
                1)]


class MatMulOp(Op):
    """Op to matrix multiply two nodes."""

    def __call__(self, node_A, node_B, trans_A=False, trans_B=False):
        """Create a new node that is the result a matrix multiple of two input nodes.

        Parameters
        ----------
        node_A: lhs of matrix multiply
        node_B: rhs of matrix multiply
        trans_A: whether to transpose node_A
        trans_B: whether to transpose node_B

        Returns
        -------
        Returns a node that is the result a matrix multiple of two input nodes.
        """
        new_node = Op.__call__(self)
        new_node.matmul_attr_trans_A = trans_A
        new_node.matmul_attr_trans_B = trans_B
        new_node.inputs = [node_A, node_B]
        new_node.name = "MatMul(%s,%s,%s,%s)" % (
            node_A.name, node_B.name, str(trans_A), str(trans_B))
        return new_node

    def compute(self, node, input_vals):
        """Given values of input nodes, return result of matrix multiplication."""
        assert len(input_vals) == 2
        assert T.is_tensor(input_vals[0])
        assert T.is_tensor(input_vals[1])
        tensor1 = T.transpose(
            input_vals[0]) if node.matmul_attr_trans_A else input_vals[0]
        tensor2 = T.transpose(
            input_vals[1]) if node.matmul_attr_trans_B else input_vals[1]
        return T.dot(tensor1, tensor2)

    def vjp(self, node, output_grad):
        """Given vjp of multiply node, return vjp contributions to each input.

        Useful formula: if Y=AB, then dA=dY B^T, dB=A^T dY
        """
        return [
            matmul(
                output_grad,
                node.inputs[1],
                trans_A=False,
                trans_B=not node.matmul_attr_trans_B),
            matmul(
                node.inputs[0],
                output_grad,
                trans_A=not node.matmul_attr_trans_A,
                trans_B=False)]


class EinsumOp(Op):
    """Op to perform einstein summation for two nodes."""

    def __call__(self, subscripts, node_A, node_B):
        """Create a new node that is the result a matrix multiple of two input nodes.

        Parameters
        ----------
        node_A: lhs of einsum
        node_B: rhs of einsum

        Returns
        -------
        Returns a node that is the result of einsum.
        """
        new_node = Op.__call__(self)
        new_node.einsum_subscripts = subscripts
        new_node.inputs = [node_A, node_B]
        new_node.name = "einsum(%s,%s,%s)" % (
            subscripts, node_A.name, node_B.name)
        return new_node

    def compute(self, node, input_vals):
        """Given values of input nodes, return result of matrix multiplication."""
        assert len(input_vals) == 2
        assert T.is_tensor(input_vals[0])
        assert T.is_tensor(input_vals[1])
        return T.einsum(node.einsum_subscripts, input_vals[0], input_vals[1])

    def vjp(self, node, output_grad):
        subscripts_dl = einsum_grad_subscripts(
            node.einsum_subscripts, left=True)
        subscripts_dr = einsum_grad_subscripts(
            node.einsum_subscripts, left=False)
        return [
            einsum(
                subscripts_dl,
                output_grad,
                node.inputs[1]),
            einsum(
                subscripts_dr,
                node.inputs[0],
                output_grad)]


class NormOp(Op):

    def __call__(self, node, order=2, axis=None):
        new_node = Op.__call__(self)
        new_node.order = order
        new_node.axis = axis
        new_node.inputs = [node]
        new_node.name = "norm(%s,%s,%s)" % (node.name, order, axis)
        return new_node

    def compute(self, node, input_vals):
        assert len(input_vals) == 1
        assert T.is_tensor(input_vals[0])
        return T.norm(input_vals[0], node.order, node.axis)

    def vjp(self, node, output_grad):
        if node.axis is not None or node.order != 2:
            raise NotImplementedError
        return [output_grad * norm(node.inputs[0])**(-1) * node.inputs[0]]


class SumOp(Op):

    def __call__(self, node, axis=None):
        new_node = Op.__call__(self)
        new_node.axis = axis
        new_node.inputs = [node]
        new_node.name = "sum(%s,%s)" % (node.name, axis)
        return new_node

    def compute(self, node, input_vals):
        assert len(input_vals) == 1
        assert T.is_tensor(input_vals[0])
        return T.sum(input_vals[0], node.axis)

    def vjp(self, node, output_grad):
        if node.axis != None:
            raise NotImplementedError
        return [output_grad * oneslike(node.inputs[0])]


class TransposeOp(Op):

    def __call__(self, node):
        new_node = Op.__call__(self)
        new_node.inputs = [node]
        new_node.name = "transpose(%s)" % (node.name)
        return new_node

    def compute(self, node, input_vals):
        assert len(input_vals) == 1
        assert T.is_tensor(input_vals[0])
        return T.transpose(input_vals[0])

    def vjp(self, node, output_grad):
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

    def vjp(self, node, output_grad):
        """No vjp function since node has no inputs."""
        return None


class ZerosLikeOp(Op):
    """Op that represents a constant T.zeros_like."""

    def __call__(self, node_A):
        """Creates a node that represents a T.zeros array of same shape as node_A."""
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = "Zeroslike(%s)" % node_A.name
        return new_node

    def compute(self, node, input_vals):
        """Returns zeros_like of the same shape as input."""
        assert (T.is_tensor(input_vals[0]))
        return T.zeros(input_vals[0].shape)

    def vjp(self, node, output_grad):
        return [zeroslike(node.inputs[0])]


class OnesLikeOp(Op):
    """Op that represents a constant T.ones_like."""

    def __call__(self, node_A):
        """Creates a node that represents a T.ones array of same shape as node_A."""
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = "Oneslike(%s)" % node_A.name
        return new_node

    def compute(self, node, input_vals):
        """Returns ones_like of the same shape as input."""
        return T.ones(input_vals[0].shape)

    def vjp(self, node, output_grad):
        return [zeroslike(node.inputs[0])]


class NegativeOp(Op):
    """Op that represents a constant T.ones_like."""

    def __call__(self, node_A):
        """Creates a node that represents a T.ones array of same shape as node_A."""
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = "(-%s)" % node_A.name
        return new_node

    def compute(self, node, input_vals):
        """Returns ones_like of the same shape as input."""
        assert (T.is_tensor(input_vals[0]))
        return -input_vals[0]

    def vjp(self, node, output_grad):
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
Sum = SumOp()
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
        node_val_results = [node_to_val_map[node]
                            for node in self.eval_node_list]
        return node_val_results


def vjps(output_node, node_list, input_vector):
    """Take vector-jacobian product of output node with respect to each node in node_list.

    Parameters
    ----------
    output_node: output node that we are taking derivative of.
    node_list: list of nodes that we are taking derivative wrt.

    Returns
    -------
    A list of vjp values, one for each node in node_list respectively.

    """

    # a map from node to a list of vjp contributions from each output node
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
        vjp = sum(node_to_output_grads_list[node])
        node_to_output_grad[node] = vjp
        for index, input in enumerate(node.inputs):
            input_vjp = node.op.vjp(node, vjp)[index]
            if input not in node_to_output_grads_list:
                node_to_output_grads_list[input] = [input_vjp]
            else:
                node_to_output_grads_list[input].append(input_vjp)

    # Collect results for vjps requested.
    grad_node_list = [node_to_output_grad[node] for node in node_list]
    return grad_node_list


def gradients(output_node, node_list):
    return vjps(output_node, node_list, oneslike(output_node))
