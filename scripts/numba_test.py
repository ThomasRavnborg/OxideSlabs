import numpy as np
import numba
from numba import njit, prange
import time

@njit(parallel=True)
def prange_test(A):
    s = 0.0
    for i in prange(A.shape[0]):
        s += np.sin(A[i]) * np.cos(A[i])
    return s

N = 100_000_000
A = np.random.rand(N)

print("Threads:", numba.get_num_threads())
print("Layer:", numba.threading_layer())

# Warmup compilation
prange_test(A[:10])

t0 = time.time()
result = prange_test(A)
t1 = time.time()

print("Result:", result)
print("Elapsed:", t1 - t0, "seconds")