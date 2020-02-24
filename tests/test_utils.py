import autodiff as ad
import backend as T
import numpy as np  # This is used for generate random numbers.


def gen_dict(input_nodes):
    """Generates a random dict for executor to use.
    """
    feed_dict = {}
    for i_node in input_nodes:
        feed_dict[i_node] = T.tensor(np.asarray(np.random.rand(*i_node.shape)))
    return feed_dict


def float_eq(A, B, tol=1e-8):
    return (abs(T.to_numpy(A) - T.to_numpy(B)) < tol).all()


def tree_eq(out, new_out, input_nodes, tol=1e-8):
    """Compares whether two output (based on the same set of inputs are equal.
    """
    feed_dict = gen_dict(input_nodes)

    executor = ad.Executor([out])
    out_val, = executor.run(feed_dict=feed_dict)

    executor = ad.Executor([new_out])
    new_out_val, = executor.run(feed_dict=feed_dict)
    return float_eq(out_val, new_out_val, tol)
