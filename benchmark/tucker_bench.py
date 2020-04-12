import argparse

import autodiff as ad
import backend as T
import tensorly as tl
import time
import numpy as np

from tensors.synthetic_tensors import init_rand_tucker
from examples.tucker import tucker_als, tucker_als_shared_exec
from tensorly.decomposition import tucker as tucker_tensorly
from sktensor import dtensor
from sktensor.tucker import hooi as sk_tucker


def tucker_als_benchmark_numpy(dim, size, rank, num_iter):
    T.set_backend('numpy')

    input_val = init_rand_tucker(dim, size, rank)

    # als
    _, _, _, sweep_times_als = tucker_als(dim,
                                          size,
                                          rank,
                                          num_iter,
                                          input_val,
                                          calculate_loss=False,
                                          return_time=True)
    print(f'als time is: {np.mean(sweep_times_als)}')

    # dt
    _, _, _, sweep_times_dt = tucker_als_shared_exec(dim,
                                                     size,
                                                     rank,
                                                     num_iter,
                                                     input_val,
                                                     calculate_loss=False,
                                                     return_time=True)
    print(f'dt time is: {np.mean(sweep_times_dt)}')

    # tensorly
    _, _, input_tensor_val = init_rand_tucker(dim, size, rank)
    input_tensor = tl.tensor(input_tensor_val, dtype='float64')

    _, _, sweep_time_tensorly = tucker_tensorly(input_tensor,
                                                rank=rank,
                                                init='random',
                                                tol=0,
                                                n_iter_max=num_iter,
                                                verbose=0,
                                                return_time=True)
    print(f'Tensorly time is: {np.mean(sweep_time_tensorly)}')

    # sktensor
    _, _, input_tensor_val = init_rand_tucker(dim, size, rank)

    _, _, sweep_time_sktensor = sk_tucker(dtensor(input_tensor_val),
                                          rank=[rank for _ in range(dim)],
                                          maxIter=num_iter,
                                          init='random',
                                          return_time=True)
    print(f'sktensor time is: {np.mean(sweep_time_sktensor)}')

    print('full summary')
    print(f'als time is: {sweep_times_als}')
    print(f'dt time is: {sweep_times_dt}')
    print(f'Tensorly time is: {sweep_time_tensorly}')
    print(f'sktensor time is: {sweep_time_sktensor}')

    print('summary')
    print(f'als time is: {np.mean(sweep_times_als)}')
    print(f'dt time is: {np.mean(sweep_times_dt)}')
    print(f'Tensorly time is: {np.mean(sweep_time_tensorly)}')
    print(f'sktensor time is: {np.mean(sweep_time_sktensor)}')


def tucker_als_benchmark_ctf(dim, size, rank, num_iter):
    T.set_backend('ctf')

    input_val = init_rand_tucker(dim, size, rank)

    # dt
    _, _, _, sweep_times_dt = tucker_als_shared_exec(dim,
                                                     size,
                                                     rank,
                                                     num_iter,
                                                     input_val,
                                                     calculate_loss=False,
                                                     return_time=True)
    print(f'dt time is: {np.mean(sweep_times_dt)}')

    print('full summary')
    print(f'dt time is: {sweep_times_dt}')
    print('summary')
    print(f'dt time is: {np.mean(sweep_times_dt)}')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dim', type=int, default=3)
    parser.add_argument('--size', type=int, default=50)
    parser.add_argument('--rank', type=int, default=25)
    parser.add_argument('--numiter', type=int, default=3)
    parser.add_argument('--backend', type=str, default='numpy')
    args, _ = parser.parse_known_args()
    print(args.backend, args.dim, args.size, args.rank, args.numiter)
    if args.backend == 'numpy':
        tucker_als_benchmark_numpy(dim=args.dim,
                                   size=args.size,
                                   rank=args.rank,
                                   num_iter=args.numiter)
    if args.backend == 'ctf':
        tucker_als_benchmark_ctf(dim=args.dim,
                                 size=args.size,
                                 rank=args.rank,
                                 num_iter=args.numiter)
