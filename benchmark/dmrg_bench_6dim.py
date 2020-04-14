import argparse

import autodiff as ad
import backend as T
import time

import quimb.tensor as qtn
from examples.mps import dmrg, dmrg_shared_exec
from examples.mps_sparse_solve import dmrg_shared_exec_sparse_solve
from tensors.quimb_tensors import load_quimb_tensors


def dmrg_als_benchmark_numpy(num=6, size=10, mpo_rank=20, num_iter=5):
    T.set_backend('numpy')

    h = qtn.MPO_rand_herm(num, mpo_rank, size)

    for max_mps_rank in [5, 10, 15, 20, 25, 30, 35, 40]:
        dmrg_quimb = qtn.DMRG2(h, bond_dims=[max_mps_rank])

        h_tensors = load_quimb_tensors(h)
        mps_tensors = load_quimb_tensors(dmrg_quimb.state)

        # dt
        _, eigvals_dt, time_dt, matvecs_dt = dmrg_shared_exec_sparse_solve(
            h_tensors,
            mps_tensors,
            num_iter=num_iter,
            max_mps_rank=max_mps_rank,
            return_time=True)

        # dmrg based on quimb
        opts = {'max_bond': max_mps_rank}

        time_quimb = []
        matvecs_quimb = []
        t0 = time.time()
        for _ in range(num_iter):
            _, matvec_quimb = dmrg_quimb.sweep_right(canonize=True,
                                                     verbosity=0,
                                                     return_matvec_num=True,
                                                     **opts)
            time_quimb.append(time.time() - t0)
            matvecs_quimb.append(matvec_quimb)
        # print(dmrg_quimb.local_energies)
        # print(dmrg_quimb.total_energies)
        eigvals_quimb = [
            energies[-1] for energies in dmrg_quimb.local_energies
        ]

        print('summary')
        print(f'dt time is: {time_dt}')
        print(f'Quimb time is: {time_quimb}')
        print(f'dt energies is: {eigvals_dt}')
        print(f'Quimb energies is: {eigvals_quimb}')
        print(f'dt num matvecs is: {matvecs_dt}')
        print(f'Quimb num matvecs is: {matvecs_quimb}')


if __name__ == '__main__':
    dmrg_als_benchmark_numpy()
