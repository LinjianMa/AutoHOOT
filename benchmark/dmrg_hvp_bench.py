import argparse

import autodiff as ad
import backend as T
import time
import numpy as np

import quimb.tensor as qtn
from examples.mps import dmrg, dmrg_shared_exec
from examples.mps_sparse_solve import dmrg_shared_exec_hvp, dmrg_hvp_jax
from tensors.quimb_tensors import load_quimb_tensors


def dmrg_als_benchmark_numpy(num, mpo_rank, max_mps_rank, size, num_iter,
                             num_inner_iter):
    T.set_backend('numpy')

    h = qtn.MPO_rand_herm(num, mpo_rank, size)
    dmrg_quimb = qtn.DMRG2(h, bond_dims=[max_mps_rank])

    # dt
    h_tensors = load_quimb_tensors(h)
    mps_tensors = load_quimb_tensors(dmrg_quimb.state)

    dt_sweep_times = dmrg_shared_exec_hvp(h_tensors,
                                          mps_tensors,
                                          num_iter=num_iter,
                                          max_mps_rank=max_mps_rank,
                                          num_inner_iter=num_inner_iter)
    print(f'dt time is: {np.mean(dt_sweep_times)}')

    # dmrg based on quimb
    opts = {'max_bond': max_mps_rank}

    quimb_sweep_times = []
    for _ in range(num_iter):
        dt = dmrg_quimb.hvp_sweep_right(num_inner_iter=num_inner_iter,
                                        canonize=True,
                                        verbosity=0,
                                        **opts)
        quimb_sweep_times.append(dt)
    print(f'Quimb time is: {np.mean(quimb_sweep_times)}')

    print('full summary')
    print(f'dt time is: {dt_sweep_times}')
    print(f'Quimb time is: {quimb_sweep_times}')

    print('summary')
    print(f'dt time is: {np.mean(dt_sweep_times)}')
    print(f'Quimb time is: {np.mean(quimb_sweep_times)}')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--num', type=int, default=3)
    parser.add_argument('--size', type=int, default=10)
    parser.add_argument('--rank', type=int, default=10)
    parser.add_argument('--numiter', type=int, default=3)
    parser.add_argument('--numinneriter', type=int, default=10)
    args, _ = parser.parse_known_args()
    print(args.num, args.size, args.rank, args.numiter, args.numinneriter)
    dmrg_als_benchmark_numpy(num=args.num,
                             size=args.size,
                             mpo_rank=args.rank,
                             max_mps_rank=args.rank,
                             num_iter=args.numiter,
                             num_inner_iter=args.numinneriter)
