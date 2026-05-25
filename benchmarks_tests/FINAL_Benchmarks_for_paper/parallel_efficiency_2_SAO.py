#!/usr/bin/env python3
"""
Parallel efficiency plot for PyFock and PySCF.
"""
import numpy as np
import matplotlib.pyplot as plt

# Raw data
cores = np.array([1, 2, 4, 8, 16, 24, 32])

time_pyscf = np.array([
    13191.86, 6684.68, 3432.17, 1859.45, 1006.68, 724.96, 586.13
])
time_pyfock_J = np.array([
    2056.62, 1092.38, 713.90, 355.90, 187.48, 147.10, 119.59
])

time_pyfock_XC = np.array([
    1089.47, 600.01, 298.32, 169.99, 112.85, 92.33, 90.46
])

time_pyfock = np.array([
    4771.88, 2594.16, 1550.15, 851.76, 535.78, 444.94, 403.04
])

# Choose max number of cores to display
max_cores = 32  # change this value to 32, 16, etc.
mask = cores <= max_cores
cores = cores[mask]
time_pyfock = time_pyfock[mask]
time_pyfock_J = time_pyfock_J[mask]
time_pyfock_XC = time_pyfock_XC[mask]
time_pyscf = time_pyscf[mask]

# Efficiency calculation
eta_pyfock = time_pyfock[0] / (cores * time_pyfock)
eta_pyfock_J = time_pyfock_J[0] / (cores * time_pyfock_J)
eta_pyfock_XC = time_pyfock_XC[0] / (cores * time_pyfock_XC)
eta_pyscf  = time_pyscf[0]  / (cores * time_pyscf)
print(eta_pyfock)
print(eta_pyfock_J)
print(eta_pyfock_XC)
print(eta_pyscf)

# Plot
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(cores, eta_pyfock, 'o-', linewidth=2, markersize=8, label="PyFock")
ax.plot(cores, eta_pyfock_J, '^--', linewidth=2, markersize=8, label="PyFock (ERI)")
ax.plot(cores, eta_pyfock_XC, 'v--', linewidth=2, markersize=8, label="PyFock (XC)")
ax.plot(cores, eta_pyscf, 's-', linewidth=2, markersize=7, label="PySCF")
ax.axhline(1.0, color='k', linestyle=':', linewidth=2.4, label="Ideal")

ax.set_xlabel("Number of Cores", fontsize=16, fontweight='bold')
ax.set_ylabel("Parallel Efficiency", fontsize=16, fontweight='bold')
ax.set_title("Parallel Efficiency: PyFock vs PySCF", fontsize=18, fontweight='bold')
ax.set_xticks(cores)
ax.set_xticklabels(cores, fontsize=14, fontweight='bold')
ax.set_ylim(0, 1.1)

# Make y-axis tick labels larger and bold
ax.tick_params(axis='y', labelsize=14)
for label in ax.get_yticklabels():
    label.set_fontweight('bold')

# Make plot border (spines) thicker
for spine in ax.spines.values():
    spine.set_linewidth(2)

ax.grid(True, ls="--", alpha=0.6, linewidth=1.6)
ax.legend(fontsize=13)
plt.tight_layout()
plt.show()