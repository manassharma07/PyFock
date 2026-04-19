#!/usr/bin/env python3
"""
Strong scaling plot for PyFock vs PySCF.
Includes breakdown of PyFock timings into J and XC.
"""

import numpy as np
import matplotlib.pyplot as plt

# Raw data
cores = np.array([1, 2, 4, 8, 16, 32])
time_pyfock_total  = np.array([5396.46, 2924.39, 1614.15, 919.29, 604.89, 440.88])
time_pyfock_J = np.array([2307.32, 1185.91, 700.09, 361.76, 203.68, 126.27])
time_pyfock_XC = np.array([1250.26, 705.14, 341.10, 191.41, 131.99, 103.39])
time_pyscf_total = np.array([14121.08, 7275.02, 3871.92, 2164.48, 1305.71, 871.30])

# Choose max number of cores to display
max_cores = 32  # change to 32, 16, etc. if needed
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
