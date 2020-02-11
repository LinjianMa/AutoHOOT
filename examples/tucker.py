import copy
import autodiff as ad
import backend as T
from utils import CharacterGetter
from tensors.synthetic_tensors import init_rand_tucker
from graph_ops.graph_generator import split_einsum
from numpy.core.einsumfunc import _parse_einsum_input


def n_mode_eigendec(node, tensor, rank):
    """
    Eigendecomposition of mode-n unfolding of a input node.
    Used in Tucker decomposition to update the core tensor
    and one factor matrix.

    Parameters
    ----------
    node: the input einsum node. Note that it must be the EinsumNode
        of the core tensor node and one factor matrix node.
    tensor: the value of the input node
    rank: Tucker decomposition rank

    Returns
    -------
    1. the core tensor
    2. the corresponding factor matrix
    """
    assert isinstance(node, ad.EinsumNode)
    assert len(node.inputs) == 2

    in_subs, out_subs, _ = _parse_einsum_input(
        (node.einsum_subscripts, *node.inputs))
    core_subs, A_subs = in_subs.split(',')

    assert len(A_subs) == 2

    char_A_not_in_out = list(set(A_subs) - set(out_subs))[0]

    out_subs_2 = "".join([
        char if char not in A_subs else char_A_not_in_out for char in out_subs
    ])
    # used for inner product of tensor
    einstr = out_subs + "," + out_subs_2 + "->" + A_subs

    Y = T.einsum(einstr, tensor, tensor)
    U, _, _ = T.svd(Y)
    U = U[:, :rank]

    einstr = out_subs + "," + A_subs + "->" + core_subs
    core = T.einsum(einstr, tensor, U)
    return core, U


class TuckerGraph(object):
    """
    Produce a graph representing the Tucker decomposition.

    Note: current graph produces the decomposition with equidimensional core tensor.

    Parameters
    ----------
    dim: dimensionality of the input tensor
    size: the size of input tensor's each dim
    rank: the rank of the decomposition

    Variables
    ---------
    X: input tensor
    core: decomposed core tensor node
    A_list: a list of decomposed matrices nodes
    einsum_subscripts
    output: the output einsum node reconstructed from core and A_list
    residual: residual of the decomposition
    loss: Tucker decomposition loss

    """
    def __init__(self, dim, size, rank):

        cg = CharacterGetter()

        self.X = ad.Variable(name='X', shape=[size for _ in range(dim)])
        X_subscripts = "".join([cg.getchar() for _ in range(dim)])

        self.core = ad.Variable(name='core', shape=[rank for _ in range(dim)])
        core_subscripts = "".join([cg.getchar() for _ in range(dim)])

        self.A_list = []
        A_list_subscripts = []
        for i in range(dim):
            node = ad.Variable(name=f'A{i}', shape=[size, rank])
            self.A_list.append(node)
            A_list_subscripts.append(f"{X_subscripts[i]}{core_subscripts[i]}")

        input_subs = ','.join([
            subscripts for subscripts in A_list_subscripts + [core_subscripts]
        ])
        self.einsum_subscripts = input_subs + '->' + X_subscripts

        self.output = ad.einsum(self.einsum_subscripts,
                                *(self.A_list + [self.core]))

        self.residual = self.output - self.X

        self.intermediates, self.losses = [], []
        for i in range(dim):
            intermediate, loss = self._build_graph_w_intermediate(i)
            self.intermediates.append(intermediate)
            self.losses.append(loss)

    def _build_graph_w_intermediate(self, index):
        """
        rebuild the graph so that intermediate will be an input of output.
        """
        intermediate_set = {self.core, self.A_list[index]}
        split_input_nodes = list(set(self.output.inputs) - intermediate_set)
        output = split_einsum(self.output, split_input_nodes)

        # get the intermediate node
        intermediate = [
            node for node in output.inputs if isinstance(node, ad.EinsumNode)
        ][0]

        residual = output - self.X

        residual_shape = list(range(len(residual.shape)))
        loss = ad.tensordot(residual,
                            residual,
                            axes=[residual_shape, residual_shape])

        return intermediate, loss


def tucker_als_graph(dim, size, rank):
    """
    Build the graph used for Tucker ALS.

    Parameters
    ----------
    dim: dimensionality of the input tensor
    size: the size of input tensor's each dim
    rank: the rank of the decomposition

    Returns
    -------
    tg: an TuckerGraph object
    executors: list of executors. Each executor is used for
        one step of Tucker ALS
    intermediates: list of einsum nodes. Each node is the objective
        each Tucker ALS step optimized for
    """
    tg = TuckerGraph(dim, size, rank)

    executors = []

    for i in range(dim):

        core_A = tg.intermediates[i]
        hes = ad.hessian(tg.losses[i], [core_A])
        hes = hes[0][0]
        grad, = ad.gradients(tg.losses[i], [core_A])

        new_core_A = core_A - ad.tensordot(
            ad.tensorinv(hes), grad,
            [[i + dim for i in range(dim)], [i for i in range(dim)]])

        executor = ad.Executor([tg.losses[i], new_core_A])
        executors.append(executor)

    return tg, executors, tg.intermediates


def tucker_als(dim, size, rank, num_iter, input_val=[]):

    tg, executors, intermediates = tucker_als_graph(dim, size, rank)

    if input_val == []:
        A_val_list, core_val, X_val = init_rand_tucker(dim, size, rank)
    else:
        A_val_list, core_val, X_val = copy.deepcopy(input_val)

    for iter in range(num_iter):
        # als iterations
        for i in range(dim):

            feed_dict = dict(zip(tg.A_list, A_val_list))
            feed_dict.update({tg.core: core_val, tg.X: X_val})

            loss_val, new_core_A_val = executors[i].run(feed_dict=feed_dict)

            # update core_val and A_val_list[i] using SVD
            core_val, A_val_list[i] = n_mode_eigendec(intermediates[i],
                                                      new_core_A_val, rank)

        print(f'At iteration {iter} the loss is: {loss_val}')

    return A_val_list, core_val, X_val


if __name__ == "__main__":
    tucker_als(dim=4, size=5, rank=3, num_iter=5)
