#!/usr/bin/env python3
"""
Strong scaling plot for PyFock vs PySCF.
Includes breakdown of PyFock timings into J and XC.
"""

import numpy as np
import matplotlib.pyplot as plt

# Raw data
cores = np.array([1, 2, 4, 8, 16, 32, 48, 64])
time_pyfock_total = np.array([8265.656, 4300.769, 2505.805, 1371.796,
                              840.332, 587.816, 527.458, 476.245])
time_pyfock_J = np.array([5186.064, 2647.520, 1572.094, 814.812,
                          448.830, 273.503, 209.090, 145.506])
time_pyfock_XC = np.array([1246.379, 653.984, 340.340, 190.672,
                           125.339, 103.408, 110.146, 120.418])
time_pyscf_total  = np.array([15698.977, 8119.609, 4344.811, 2490.722,
                              1455.911, 965.482, 829.167, 773.742])

# Choose max number of cores to display
max_cores = 64  # change to 32, 16, etc. if needed
mask = cores <= max_cores

cores = cores[mask]
time_pyfock_total = time_pyfock_total[mask]
time_pyfock_J = time_pyfock_J[mask]
time_pyfock_XC = time_pyfock_XC[mask]
time_pyscf_total = time_pyscf_total[mask]

# Ideal scaling reference (PyFock total, 1-core baseline)
t1 = time_pyfock_total[0]
ideal_scaling = t1 / cores

# Plot
plt.figure(figsize=(8, 6))

plt.loglog(cores, time_pyfock_total, 'o-', label="PyFock (Total)", linewidth=2, markersize=8)
plt.loglog(cores, time_pyfock_J, 's--', label="PyFock (ERI)", linewidth=2, markersize=7)
plt.loglog(cores, time_pyfock_XC, 'd--', label="PyFock (XC)", linewidth=2, markersize=7)
plt.loglog(cores, time_pyscf_total, '^-.', label="PySCF (Total)", linewidth=2, markersize=7)
plt.loglog(cores, ideal_scaling, 'k:', label="Ideal Scaling", linewidth=1.5)

plt.xlabel("Number of Cores", fontsize=13)
plt.ylabel("Wall Time (s)", fontsize=13)
plt.title("Strong Scaling: PyFock vs PySCF (with J and XC breakdown)", fontsize=14)
plt.xticks(cores, cores)
plt.grid(True, which="both", ls="--", alpha=0.6)
plt.legend(fontsize=11)
plt.tight_layout()
plt.show()
