import numpy as np


# This matains a global state.
# Only consider two expression that has the same subscripts and operand shapes
# to be the same einsum expression.
class _EinsumPathCached:
    def __init__(self):
        self.path = {}

    def __call__(self, *args, **kwargs):
        subscript = args[0]
        operands = args[1:]
        key = subscript
        key += '|'
        for operand in operands:
            key += '-'.join([str(dim) for dim in operand.shape])
            key += '|'

        if key not in self.path:
            self.path[key] = np.einsum_path(*args,
                                            **kwargs,
                                            optimize='optimal')[0]
        kwargs['optimize'] = self.path[key]
        return np.einsum(*args, **kwargs)


einsum_pc = _EinsumPathCached()

# import time
# N = 10
# C = np.random.rand(N, N)
# I = np.random.rand(N, N, N, N)
# begin = time.time()
# for i in range(10):
#     einsum_pc('pi,qj,ijkl,rk,sl->pqrs', C, C, I, C, C)
#     einsum_pc('pi,qj,ijko,rk,so->pqrs', C, C, I, C, C)
# end = time.time()
# print(einsum_pc.path)
# print(f'{end - begin}')
# begin = time.time()
# for i in range(10):
#     np.einsum('pi,qj,ijkl,rk,sl->pqrs', C, C, I, C, C, optimize='optimal')
# end = time.time()
# print(f'{end - begin}')
