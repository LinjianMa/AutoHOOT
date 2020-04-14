import argparse

import autodiff as ad
import backend as T
import time

import quimb.tensor as qtn
from examples.mps import dmrg, dmrg_shared_exec
from examples.mps_sparse_solve import dmrg_shared_exec_sparse_solve
from tensors.quimb_tensors import load_quimb_tensors


def dmrg_als_benchmark_numpy(num, mpo_rank, max_mps_rank, size, num_iter):
    T.set_backend('numpy')

    h = qtn.MPO_rand_herm(num, mpo_rank, size)
    dmrg_quimb = qtn.DMRG2(h, bond_dims=[max_mps_rank])

    h_tensors = load_quimb_tensors(h)
    mps_tensors = load_quimb_tensors(dmrg_quimb.state)

    # dt
    _, _, time_dt = dmrg_shared_exec_sparse_solve(h_tensors,
                                                  mps_tensors,
                                                  num_iter=num_iter,
                                                  max_mps_rank=max_mps_rank,
                                                  return_time=True)

    # dmrg based on quimb
    opts = {'max_bond': max_mps_rank}

    time_quimb = []
    t0 = time.time()
    for _ in range(num_iter):
        dmrg_quimb.sweep_right(canonize=True, verbosity=0, **opts)
        time_quimb.append(time.time() - t0)
    print(dmrg_quimb.local_energies)
    print(dmrg_quimb.total_energies)

    print('summary')
    print(f'dt time is: {time_dt}')
    print(f'Quimb time is: {time_quimb}')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--num', type=int, default=6)
    parser.add_argument('--size', type=int, default=30)
    parser.add_argument('--rank', type=int, default=15)
    parser.add_argument('--numiter', type=int, default=1)
    args, _ = parser.parse_known_args()
    print(args.num, args.size, args.rank, args.numiter)
    dmrg_als_benchmark_numpy(num=args.num,
                             size=args.size,
                             mpo_rank=args.rank,
                             max_mps_rank=args.rank,
                             num_iter=args.numiter)
