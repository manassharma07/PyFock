import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# -----------------
# PySCF data
# -----------------
nbf = np.array([25, 125, 250, 500, 800, 1175, 1900, 2500, 3475], dtype=float)
total_time = np.array([0.059873508, 0.351985034, 1.545247587, 5.655018284, 13.58711019,
                       28.77391837, 67.70287694, 129.9493543, 262.5715926], dtype=float)

# Scaling law function
def scaling_law(N, a, p):
    return a * N**p

# -----------------
# Method 1: Log–log linear regression
# -----------------
def loglog_fit(N, t):
    logN = np.log(N)
    logt = np.log(t)
    coeffs = np.polyfit(logN, logt, 1)
    p = coeffs[0]
    a = np.exp(coeffs[1])
    return a, p

# -----------------
# Method 2: Nonlinear least squares
# -----------------
def nonlinear_fit(N, t):
    popt, _ = curve_fit(scaling_law, N, t, p0=(1e-6, 2))
    a, p = popt
    return a, p

# Fit results
a_log, p_log = loglog_fit(nbf, total_time)
a_nonlin, p_nonlin = nonlinear_fit(nbf, total_time)

# Print results
print("PySCF Total Time Scaling:")
print(f"  Log–log fit: a = {a_log:.6e}, p = {p_log:.4f}")
print(f"  Nonlinear fit: a = {a_nonlin:.6e}, p = {p_nonlin:.4f}")

# -----------------
# Plotting function
# -----------------
def plot_fit(a, p, method_name):
    plt.figure(figsize=(7,5))
    N_fit = np.linspace(min(nbf), max(nbf), 200)

    # Plot data and fit
    plt.plot(nbf, total_time, 'o', color="tab:blue", label="PySCF Data")
    plt.plot(N_fit, scaling_law(N_fit, a, p), '-', color="tab:blue", 
             label=f"Fit (N^{p:.2f})")

    # Axis labels
    plt.xlabel(r"No. of Basis Functions ($N_{bf}$)", fontsize=15, weight='bold')
    plt.ylabel("Wall Time per Iteration (s)", fontsize=15, weight='bold')

    # Bold tick labels
    plt.xticks(fontsize=13, fontweight='bold')
    plt.yticks(fontsize=13, fontweight='bold')

    # Multiline title
    plt.title(f"Scaling Behavior of PySCF\nKS-DFT Calculations on Water Clusters\n({method_name} fit)",
              fontsize=16, weight='bold', pad=15)

    # Thick border
    for spine in plt.gca().spines.values():
        spine.set_linewidth(1.8)

    # Bold legend
    plt.legend(fontsize=12, prop={'weight': 'bold'})
    plt.grid(True)
    plt.tight_layout()

# Plot for both methods
plot_fit(a_log, p_log, "Log–log")
plot_fit(a_nonlin, p_nonlin, "Nonlinear")

plt.show()
