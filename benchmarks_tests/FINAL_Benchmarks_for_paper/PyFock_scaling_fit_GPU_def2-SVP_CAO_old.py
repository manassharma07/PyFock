import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# -----------------
# Updated data from the table
# -----------------
nbf = np.array([125, 250, 500, 800, 1175, 1900, 2500, 3475], dtype=float)
total_time = np.array([0.116, 0.202, 0.461, 0.811, 1.526, 3.285, 5.879, 10.650], dtype=float)
j_time = np.array([0.026, 0.051, 0.158, 0.342, 0.771, 1.857, 3.772, 7.131], dtype=float)
xc_time = np.array([0.052, 0.088, 0.167, 0.263, 0.413, 0.777, 1.075, 1.639], dtype=float)

# Scaling law function
def scaling_law(N, a, p):
    return a * N**p

# -----------------
# Method 1: Log–log linear regression
# -----------------
def loglog_fit(N, t):
    logN = np.log(N)
    logt = np.log(t)
    coeffs = np.polyfit(logN, logt, 1)  # slope, intercept
    p = coeffs[0]
    a = np.exp(coeffs[1])
    return a, p

# -----------------
# Method 2: Nonlinear least squares
# -----------------
def nonlinear_fit(N, t):
    popt, _ = curve_fit(scaling_law, N, t, p0=(1e-6, 2))  # initial guess
    a, p = popt
    return a, p

# Fit for each quantity
results = {}
for label, data in zip(["Total", "ERI", "XC"], [total_time, j_time, xc_time]):
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
# Plotting
# -----------------
def plot_fit(method_name):
    plt.figure(figsize=(7,5))
    for label, data, color in zip(["Total", "ERI", "XC"], 
                                  [total_time, j_time, xc_time],
                                  ["tab:blue", "tab:orange", "tab:green"]):
        a, p = results[label][method_name]
        N_fit = np.linspace(min(nbf), max(nbf), 200)
        plt.plot(nbf, data, 'o', color=color, label=f"{label}")
        # plt.plot(N_fit, scaling_law(N_fit, a, p), '-', color=color, 
        #          label=f"{label} fit (N^{p:.2f})")
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

    plt.title("Scaling Behavior of PyFock\nKS-DFT Calculations on Water Clusters", 
              fontsize=16, weight='bold', pad=15)

    for spine in plt.gca().spines.values():
        spine.set_linewidth(1.8)

    plt.legend(fontsize=12, prop={'weight': 'bold'})
    plt.grid(True)
    plt.tight_layout()

# Plot both methods
plot_fit("nonlinear")
plot_fit("log-log")
plt.show()
