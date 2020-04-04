import autodiff as ad
import backend as T
import time

import quimb.tensor as qtn
from examples.mps import dmrg, dmrg_shared_exec
from examples.mps_sparse_solve import dmrg_shared_exec_sparse_solve
from tensors.quimb_tensors import load_quimb_tensors

BACKEND_TYPES = ['numpy']
max_mps_rank = 15
num = 6
mpo_rank = 15
size = 30
num_iter = 1


def dmrg_als_benchmark_numpy():

    h = qtn.MPO_rand_herm(num, mpo_rank, size)
    dmrg_quimb = qtn.DMRG2(h, bond_dims=[max_mps_rank])

    h_tensors = load_quimb_tensors(h)
    mps_tensors = load_quimb_tensors(dmrg_quimb.state)

    # dt
    t0 = time.time()
    dmrg_shared_exec_sparse_solve(h_tensors,
                                  mps_tensors,
                                  num_iter=num_iter,
                                  max_mps_rank=max_mps_rank)
    time_dt = time.time() - t0
    print(f'dt time is: {time_dt/num_iter}')

    # dmrg based on quimb
    opts = {'max_bond': max_mps_rank}

    t0 = time.time()
    for _ in range(num_iter):
        dmrg_quimb.sweep_right(canonize=True, verbosity=0, **opts)
    print(dmrg_quimb.local_energies[-1])

    time_quimb = time.time() - t0
    print(f'Quimb time is: {time_quimb/num_iter}')

    print('summary')
    print(f'dt time is: {time_dt/num_iter}')
    print(f'Quimb time is: {time_quimb/num_iter}')


dmrg_als_benchmark_numpy()
