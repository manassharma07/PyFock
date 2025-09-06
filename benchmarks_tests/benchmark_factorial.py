import numpy as np
from pyfock.Integrals.integral_helpers import fastFactorial, fastFactorial_old
# from pyfock import Integrals
from numba import njit
from timeit import default_timer as timer

start=timer()

# 100 million evaluations of the boys function.
N = 100000000
np.random.seed(0)
n_vals = np.random.randint(0, 20, N)#.astype(float)


@njit(cache=True, fastmath=True, error_model="numpy")
def test_new(n_vals, N):
    x = np.zeros((N),dtype=np.int64)
    for i in range(N): # Million runs
        x[i] = fastFactorial(n_vals[i])
    return x

@njit(cache=True, fastmath=True, error_model="numpy")
def test_old(n_vals, N):
    x = np.zeros((N),dtype=np.int64)
    for i in range(N): # Million runs
        x[i] = fastFactorial_old(n_vals[i])
    return x

duration = timer() - start
print('Duration preliminaries: ',duration)

start=timer()
x_new = test_new(n_vals, N) # Cached
duration = timer() - start
print('Duration new: ',duration)
# Duration new:  0.3832744919927791
start=timer()
x_old = test_old(n_vals, N) # Cached
duration = timer() - start
print('Duration old: ',duration)
print(abs(x_new-x_old).max())
# Duration old:  1.1192696560174227
