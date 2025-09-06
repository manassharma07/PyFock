import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# -----------------
# Data from the table (PyFock)
# -----------------
nbf = np.array([25, 125, 250, 500, 800, 1175, 1900, 2500, 3475], dtype=float)
total_time = np.array([0.091779917, 0.251879828, 0.829753874, 3.32694805, 7.512214625,
                       17.15226698, 41.32841003, 82.38475342, 154.2059581], dtype=float)
j_time = np.array([0.013566, 0.08225514, 0.35319341, 1.91062789, 4.40820322,
                   10.75980486, 24.7638173, 54.0653728, 97.1046283], dtype=float)
xc_time = np.array([0.01539, 0.10870471, 0.33446765, 0.95263881, 1.99390185,
                    3.76085967, 8.61741557, 12.6850703, 21.373439], dtype=float)

# -----------------
# PySCF total wall times
# -----------------
pyscf_total_time = np.array([0.059873508, 0.351985034, 1.545247587, 5.655018284, 13.58711019,
                             28.77391837, 67.70287694, 129.9493543, 262.5715926], dtype=float)

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
for label, data in zip(["Total", "J", "XC", "PySCF Total"], 
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
    for label, data, color in zip(["Total", "J", "XC", "PySCF Total"], 
                                  [total_time, j_time, xc_time, pyscf_total_time],
                                  ["tab:blue", "tab:orange", "tab:green", "tab:red"]):
        a, p = results[label][method_name]
        N_fit = np.linspace(min(nbf), max(nbf), 200)
        plt.plot(nbf, data, 'o', color=color, label=f"{label}")
        if label=='XC' or label=='J':
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
