import autodiff as ad
import backend as T
from utils import CharacterGetter
from tensors.synthetic_tensors import init_rand_tucker
from graph_ops.graph_generator import split_einsum
from numpy.core.einsumfunc import _parse_einsum_input

BACKEND_TYPES = ['numpy']


def n_mode_eigendec(node, tensor, rank):
    """
    Eigendecomposition of mode-n unfolding of a tensor
    """
    in_subs, out_subs, _ = _parse_einsum_input(
        (node.einsum_subscripts, *node.inputs))
    core_subs, A_subs = in_subs.split(',')

    assert len(A_subs) == 2

    char_A_not_in_out = list(set(A_subs) - set(out_subs))[0]

    out_subs_2 = "".join([
        char if char not in A_subs else char_A_not_in_out for char in out_subs
    ])

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

        self._recover_loss()

    def _recover_loss(self):
        self.output = ad.einsum(self.einsum_subscripts,
                                *(self.A_list + [self.core]))

        self.residual = self.output - self.X

        subscripts = "".join(
            [chr(ord('a') + i) for i in range(len(self.residual.shape))])
        self.loss = ad.einsum(f"{subscripts},{subscripts}->", self.residual,
                              self.residual)

    def rebuild_graph_w_intermediate(self, index):

        self._recover_loss()

        intermediate_set = {self.core, self.A_list[index]}
        split_input_nodes = list(set(self.output.inputs) - intermediate_set)
        self.output = split_einsum(self.output, split_input_nodes)

        # get the intermediate node
        intermediate = [
            node for node in self.output.inputs
            if isinstance(node, ad.EinsumNode)
        ][0]

        self.residual = self.output - self.X

        subscripts = "".join(
            [chr(ord('a') + i) for i in range(len(self.residual.shape))])
        self.loss = ad.einsum(f"{subscripts},{subscripts}->", self.residual,
                              self.residual)

        return intermediate


def tucker_als_graph(dim, size, rank):
    """
    explains here
    """
    tg = TuckerGraph(dim, size, rank)

    executors = []
    intermediates = []

    for i in range(dim):

        core_A = tg.rebuild_graph_w_intermediate(i)
        hes = ad.hessian(tg.loss, [core_A])
        hes = hes[0][0]
        grad, = ad.gradients(tg.loss, [core_A])

        new_core_A = core_A - ad.tensordot(
            ad.tensorinv(hes), grad,
            [[i + dim for i in range(dim)], [i for i in range(dim)]])

        executor = ad.Executor([tg.loss, new_core_A])
        executors.append(executor)
        intermediates.append(core_A)

    return tg, executors, intermediates


def tucker_als(dim, size, rank, num_iter, input_val=[]):
    tg, executors, intermediates = tucker_als_graph(dim, size, rank)

    if input_val == []:
        A_val_list, core_val, X_val = init_rand_tucker(dim, size, rank)
    else:
        A_val_list, core_val, X_val = input_val

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
