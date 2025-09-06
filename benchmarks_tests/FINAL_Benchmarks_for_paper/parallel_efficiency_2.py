#!/usr/bin/env python3
"""
Parallel efficiency plot for PyFock and PySCF.
"""

import numpy as np
import matplotlib.pyplot as plt

# Raw data
cores = np.array([1, 2, 4, 8, 16, 32, 48, 64])
time_pyfock = np.array([8265.656, 4300.769, 2505.805, 1371.796,
                        840.332, 587.816, 527.458, 476.245])
time_pyscf  = np.array([15698.977, 8119.609, 4344.811, 2490.722,
                        1455.911, 965.482, 829.167, 773.742])

# Choose max number of cores to display
max_cores = 64  # change this value to 32, 16, etc.
mask = cores <= max_cores

cores = cores[mask]
time_pyfock = time_pyfock[mask]
time_pyscf = time_pyscf[mask]

# Efficiency calculation
eta_pyfock = time_pyfock[0] / (cores * time_pyfock)
eta_pyscf  = time_pyscf[0]  / (cores * time_pyscf)

# Plot
plt.figure(figsize=(7, 5))
plt.plot(cores, eta_pyfock, 'o-', linewidth=2, markersize=8, label="PyFock")
plt.plot(cores, eta_pyscf, 's--', linewidth=2, markersize=7, label="PySCF")

plt.axhline(1.0, color='k', linestyle=':', linewidth=1.2, label="Ideal")

plt.xlabel("Number of Cores", fontsize=13)
plt.ylabel("Parallel Efficiency", fontsize=13)
plt.title("Parallel Efficiency: PyFock vs PySCF", fontsize=14)
plt.xticks(cores, cores)
plt.ylim(0, 1.1)
plt.grid(True, ls="--", alpha=0.6)
plt.legend(fontsize=11)
plt.tight_layout()
plt.show()
