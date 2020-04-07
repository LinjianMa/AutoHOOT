import autodiff as ad
import backend as T
import time
import numpy as np
import tensorflow as tf

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


def dmrg_als_benchmark_tf(num, mpo_rank, max_mps_rank, size, num_iter,
                          num_inner_iter):
    T.set_backend('tensorflow')

    h = qtn.MPO_rand_herm(num, mpo_rank, size)
    for elem in h:
        elem.modify(data=tf.convert_to_tensor(value=np.float32(elem.data)))

    dmrg_quimb = qtn.DMRG2(h, bond_dims=[max_mps_rank])

    for elem in dmrg_quimb._k:
        elem.modify(data=tf.convert_to_tensor(value=np.float32(elem.data)))

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
                                        canonize=False,
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


dmrg_als_benchmark_tf(num=10,
                      mpo_rank=25,
                      max_mps_rank=25,
                      size=25,
                      num_iter=1,
                      num_inner_iter=10)
