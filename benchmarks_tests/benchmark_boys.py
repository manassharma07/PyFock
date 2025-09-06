import numpy as np
from pyfock.Integrals.integral_helpers import Fboys, Fboys_old, Fboys_jjgoings
# from pyfock import Integrals
from numba import njit
from timeit import default_timer as timer


# 100 million evaluations of the boys function.
N = 100000000
np.random.seed(0)
n_vals = np.random.randint(0, 10, N)#.astype(float)
x_vals = np.random.uniform(0.0, 35.0, N)


@njit(cache=True, fastmath=True, error_model="numpy")
def test_new(x_vals, n_vals, N):
    x = np.zeros((N))
    for i in range(N): # Million runs
        x[i] = Fboys(n_vals[i], x_vals[i])
    return x

@njit(cache=True, fastmath=True, error_model="numpy")
def test_old(x_vals, n_vals, N):
    x = np.zeros((N))
    for i in range(N): # Million runs
        x[i] = Fboys_old(n_vals[i], x_vals[i])
    return x

@njit(cache=True, fastmath=True, error_model="numpy")
def test_jjgoings(x_vals, n_vals, N):
    x = np.zeros((N))
    for i in range(N): # Million runs
        x[i] = Fboys_jjgoings(n_vals[i], x_vals[i])
    return x

start=timer()
x_new = test_new(x_vals, n_vals, N) # Cached
duration = timer() - start
print('Duration new: ',duration)
# Duration new:  4.23387262201868

start=timer()
x_old = test_old(x_vals, n_vals, N) # Gets compiled at runtime (cant be cached)
duration = timer() - start
print('Duration old: ',duration)
print(abs(x_new-x_old).max())
# Duration old:  11.750024079985451

start=timer()
x_jjgoings = test_jjgoings(x_vals, n_vals, N) # Gets compiled at runtime (cant be cached)
duration = timer() - start
print('Duration jjgoings: ',duration)
print(abs(x_jjgoings-x_old).max())
# Duration :  36.23037774604745

# Results look good compared to here: https://github.com/adabbott/Research_Notes/blob/13327abe82a3576dd95df038e24007645071bc13/Quax_dev_archive/integrals_dev/tei_trials/teis_trial7/taketa_test/test.py