import autodiff as ad
import backend as T
import time, copy
import numpy as np
from tensors.synthetic_tensors import init_rand_tucker
from examples.tucker import TuckerGraph, tucker_als, tucker_als_shared_exec

BACKEND_TYPES = ['numpy', 'tensorflow']

def test():
    dim, size, rank = 3, 400, 200

    for datatype in BACKEND_TYPES:
        T.set_backend(datatype)

        input_val = init_rand_tucker(dim, size, rank)
        A_val_list, _, X_val = input_val
        A1_val, A2_val, A3_val = A_val_list

        t0 = time.time()
        ret1 = T.einsum("abc,cl->abl", X_val, A3_val)
        t1 = time.time()
        print(t1 - t0)
        X_val_reshape = T.reshape(X_val, [X_val.shape[0]* X_val.shape[1], -1])
        ret2 = T.einsum('ac,cl->al', X_val_reshape, A3_val)
        t2 = time.time()
        print(t2 - t1)
test()