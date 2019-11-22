import numpy as np


# A path cached einsum. optimize argument will be ignored
class _EinsumPathCached:
    def __init__(self):
        self.path = None

    def __call__(self, *args, **kwargs):
        if self.path is None:
            self.path = np.einsum_path(*args, **kwargs, optimize='optimal')
        kwargs['optimize'] = self.path[0]
        return np.einsum(*args, **kwargs)


einsum_pc = _EinsumPathCached()

# import time
# N = 10
# C = np.random.rand(N, N)
# I = np.random.rand(N, N, N, N)
# begin = time.time()
# for i in range(10):
#     einsum_pc('pi,qj,ijkl,rk,sl->pqrs', C, C, I, C, C)
# end = time.time()
# print(f'{end - begin}')
# begin = time.time()
# for i in range(10):
#     np.einsum('pi,qj,ijkl,rk,sl->pqrs', C, C, I, C, C, optimize='optimal')
# end = time.time()
# print(f'{end - begin}')
