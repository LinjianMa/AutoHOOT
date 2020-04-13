import argparse

import autodiff as ad
import backend as T
import time
import numpy as np

from examples.mps_sparse_solve import dmrg_shared_exec_hvp


def rand_mps(num, rank, size):
    """
    Generate random MPS.
    """
    A_list = []
    A_list.append(T.random((rank, size)))
    for i in range(1, num - 1):
        A_list.append(T.random((rank, rank, size)))
    A_list.append(T.random((rank, size)))
    return A_list


def ham_rand_mpo(num, rank, size):
    H_list = []

    tensor = T.random((rank, size, size))
    tensor += T.einsum("abc->acb", tensor)
    H_list.append(tensor)

    for i in range(1, num - 1):
        tensor = T.random((rank, rank, size, size))
        tensor += T.einsum("abcd->abdc", tensor)
        H_list.append(tensor)

    tensor = T.random((rank, size, size))
    tensor += T.einsum("abc->acb", tensor)
    H_list.append(tensor)
    return H_list


def dmrg_als_benchmark_ctf(num, mpo_rank, max_mps_rank, size, num_iter,
                           num_inner_iter):
    T.set_backend('ctf')

    # dt
    h_tensors = ham_rand_mpo(num, mpo_rank, size)
    mps_tensors = rand_mps(num, max_mps_rank, size)

    dt_sweep_times = dmrg_shared_exec_hvp(h_tensors,
                                          mps_tensors,
                                          num_iter=num_iter,
                                          max_mps_rank=max_mps_rank,
                                          num_inner_iter=num_inner_iter)
    print(f'dt time is: {np.mean(dt_sweep_times)}')

    print('full summary')
    print(f'dt time is: {dt_sweep_times}')
    print('summary')
    print(f'dt time is: {np.mean(dt_sweep_times)}')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--num', type=int, default=3)
    parser.add_argument('--size', type=int, default=10)
    parser.add_argument('--rank', type=int, default=10)
    parser.add_argument('--numiter', type=int, default=3)
    parser.add_argument('--numinneriter', type=int, default=10)
    args, _ = parser.parse_known_args()
    print(args.num, args.size, args.rank, args.numiter, args.numinneriter)
    dmrg_als_benchmark_ctf(num=args.num,
                           size=args.size,
                           mpo_rank=args.rank,
                           max_mps_rank=args.rank,
                           num_iter=args.numiter,
                           num_inner_iter=args.numinneriter)
