import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# -----------------
# Data from the table (PyFock)
# -----------------
nbf = np.array([125, 250, 500, 800, 1175, 1900, 2500, 3475], dtype=float)
total_time = np.array([0.240, 0.629, 2.348, 5.508, 11.662, 28.205, 55.254, 103.132], dtype=float)
j_time = np.array([0.049, 0.186, 0.937, 2.188, 5.273, 11.879, 25.985, 46.390], dtype=float)
xc_time = np.array([0.108, 0.299, 0.930, 2.114, 3.778, 8.648, 13.007, 21.473], dtype=float)

# -----------------
# PySCF total wall times
# -----------------
pyscf_total_time = np.array([0.325, 1.477, 5.386, 12.869, 27.131, 62.184, 120.327, 232.729], dtype=float)

# Function to fit
def scaling_law(N, a, p):
    return a * N**p

# Log–log linear regression
def loglog_fit(N, t):
    logN = np.log(N)
    logt = np.log(t)
    coeffs = np.polyfit(logN, logt, 1)
    p = coeffs[0]
    a = np.exp(coeffs[1])
    return a, p

# Nonlinear least squares fit
def nonlinear_fit(N, t):
    popt, _ = curve_fit(scaling_law, N, t, p0=(1e-6, 2))
    a, p = popt
    return a, p

# Fit all datasets
results = {}
for label, data in zip(["Total", "ERI", "XC", "PySCF Total"], 
                       [total_time, j_time, xc_time, pyscf_total_time]):
    a_log, p_log = loglog_fit(nbf, data)
    a_nonlin, p_nonlin = nonlinear_fit(nbf, data)
    results[label] = {
        "log-log": (a_log, p_log),
        "nonlinear": (a_nonlin, p_nonlin)
    }

# Print results
for label, vals in results.items():
    print(f"{label}:")
    print(f"  Log–log fit: a = {vals['log-log'][0]:.6e}, p = {vals['log-log'][1]:.4f}")
    print(f"  Nonlinear fit: a = {vals['nonlinear'][0]:.6e}, p = {vals['nonlinear'][1]:.4f}")
    print()

# -----------------
# Plotting function
# -----------------
def plot_fit(method_name):
    plt.figure(figsize=(7,5))
    for label, data, color in zip(["Total", "ERI", "XC", "PySCF Total"], 
                                  [total_time, j_time, xc_time, pyscf_total_time],
                                  ["tab:blue", "tab:orange", "tab:green", "tab:red"]):
        a, p = results[label][method_name]
        N_fit = np.linspace(min(nbf), max(nbf), 200)
        plt.plot(nbf, data, 'o', color=color, label=f"{label}")
        if label=='XC' or label=='ERI':
            plt.plot(N_fit, scaling_law(N_fit, a, p), '--', color=color, 
                    label=f"{label} fit (N^{p:.2f})")
        else:
            plt.plot(N_fit, scaling_law(N_fit, a, p), '-', color=color, 
                label=f"{label} fit (N^{p:.2f})")

    plt.xlabel(r"No. of Basis Functions ($N_{bf}$)", fontsize=15, weight='bold')
    plt.ylabel("Wall Time per Iteration (s)", fontsize=15, weight='bold')
    plt.xticks(fontsize=13, fontweight='bold')
    plt.yticks(fontsize=13, fontweight='bold')
    plt.title("Scaling Behavior of PyFock vs PySCF\nKS-DFT Calculations on Water Clusters", 
              fontsize=16, weight='bold', pad=15)

    for spine in plt.gca().spines.values():
        spine.set_linewidth(1.8)

    plt.legend(fontsize=12, prop={'weight': 'bold'})
    plt.grid(True)
    plt.tight_layout()

# Plot for both fit methods
plot_fit("nonlinear")
plot_fit("log-log")
plt.show()
