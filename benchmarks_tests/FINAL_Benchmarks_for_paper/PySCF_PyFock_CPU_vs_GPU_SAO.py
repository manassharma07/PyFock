import matplotlib.pyplot as plt
import numpy as np

# ==== CONFIG FLAGS ====
x_axis_choice = "water"  # "water" or "basis"
log_scale = False        # True for log scale, False for linear
plot_total_only = False  # True for just total times, False for breakdown

# ==== DATA ====
water_molecules = np.array([47, 76, 100, 139])
basis_functions = np.array([1128, 1824, 2400, 3336])

# PySCF data
total_pyscf = np.array([27.13, 62.18, 120.33, 232.73])

# GPU/CPU data
total_gpu = np.array([1.53, 3.29, 5.88, 10.65])
total_cpu = np.array([11.66, 28.21, 55.25, 103.13])

j_gpu = np.array([0.77, 1.86, 3.77, 7.13])
j_cpu = np.array([5.27, 11.88, 25.99, 46.39])

xc_gpu = np.array([0.41, 0.78, 1.08, 1.64])
xc_cpu = np.array([3.78, 8.65, 13.01, 21.47])

# Derived values for "Other" = Total - (J + XC)
other_gpu = total_gpu - (j_gpu + xc_gpu)
other_cpu = total_cpu - (j_cpu + xc_cpu)

# ==== Choose x-axis ====
if x_axis_choice == "water":
    x_values = water_molecules
    x_label = "Number of Water Molecules"
    # Create subscript labels for water molecules
    x_tick_labels = [f"(H$_2$O)$_{{{n}}}$" for n in water_molecules]
    # x_tick_labels = [fr"$\mathbf{{(H_2O)}}_{{{n}}}$" for n in water_molecules]
elif x_axis_choice == "basis":
    x_values = basis_functions
    x_label = "Number of Basis Functions"
    x_tick_labels = [str(n) for n in basis_functions]
else:
    raise ValueError("x_axis_choice must be 'water' or 'basis'")

# ==== Plot ====
width = 6  # Reduced width to accommodate three bars
fig, ax = plt.subplots(figsize=(10, 6.3))

if plot_total_only:
    # Plot just total times - PySCF first, then PyFock CPU and GPU
    bars_pyscf = ax.bar(x_values - width, total_pyscf, width, label="PySCF (CPU)", color="tab:green", edgecolor='black', linewidth=1.2)
    bars_cpu = ax.bar(x_values, total_cpu, width, label="PyFock (CPU)", color="tab:red", edgecolor='black', linewidth=1.2)
    bars_gpu = ax.bar(x_values + width, total_gpu, width, label="PyFock (GPU)", color="tab:blue", edgecolor='black', linewidth=1.2)
else:
    # Stacked bars: J, XC, Other (keeping original layout but adjusting positions)
    bars_pyscf = ax.bar(x_values - width, total_pyscf, width, label="PySCF (CPU, Total)", color="tab:green", edgecolor='black', linewidth=1.2)
    
    bars_cpu = ax.bar(x_values, j_cpu, width, label="PyFock ERI (CPU)", color="tab:orange", edgecolor='black', linewidth=1.2)
    ax.bar(x_values, xc_cpu, width, bottom=j_cpu, label="PyFock XC (CPU)", color="tab:red", edgecolor='black', linewidth=1.2)
    ax.bar(x_values, other_cpu, width, bottom=j_cpu+xc_cpu, label="PyFock Other (CPU)", color="tab:gray", edgecolor='black', linewidth=1.2)

    bars_gpu = ax.bar(x_values + width, j_gpu, width, label="PyFock ERI (GPU)", color="tab:orange", alpha=0.6, edgecolor='black', linewidth=1.2)
    ax.bar(x_values + width, xc_gpu, width, bottom=j_gpu, label="PyFock XC (GPU)", color="tab:red", alpha=0.6, edgecolor='black', linewidth=1.2)
    ax.bar(x_values + width, other_gpu, width, bottom=j_gpu+xc_gpu, label="PyFock Other (GPU)", color="tab:gray", alpha=0.6, edgecolor='black', linewidth=1.2)

# Labels & settings
ax.set_xlabel(x_label, fontsize=16, fontweight='bold')
ax.set_ylabel("Time per Iteration (s)", fontsize=16, fontweight='bold')

if log_scale:
    ax.set_yscale("log")

title = "PySCF (CPU) vs PyFock (CPU and GPU)" if not plot_total_only else "PySCF vs PyFock Total Time per Iteration"
ax.set_title(title, fontsize=16, fontweight='bold')

# Set custom x-tick labels
ax.set_xticks(x_values)
ax.set_xticklabels(x_tick_labels)

# Tick labels
ax.tick_params(axis='both', labelsize=14)
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontweight('bold')

# Thicker border
for spine in ax.spines.values():
    spine.set_linewidth(1.5)

# Legend styling
legend = ax.legend(fontsize=12)
for text in legend.get_texts():
    text.set_fontweight('bold')

# ==== Annotate total times ====
if plot_total_only:
    for x, val in zip(x_values - width, total_pyscf):
        ax.text(x*1.005, val * 1.005, f"{val:.1f}", ha='center', va='bottom', fontsize=12, fontweight='bold', rotation=0)

    for x, val in zip(x_values, total_cpu):
        ax.text(x*1.005, val * 1.005, f"{val:.1f}", ha='center', va='bottom', fontsize=12, fontweight='bold', rotation=0)
    
    for x, val in zip(x_values + width, total_gpu):
        ax.text(x*1.005, val * 1.005, f"{val:.1f}", ha='center', va='bottom', fontsize=12, fontweight='bold', rotation=0)
else:
    # For breakdown view, annotate total times
    for x, val in zip(x_values - width, total_pyscf):
        ax.text(x*1.005, val * 1.005, f"{val:.1f}", ha='center', va='bottom', fontsize=12, fontweight='bold', rotation=0)

    for x, val in zip(x_values, total_cpu):
        ax.text(x*1.005, val * 1.005, f"{val:.1f}", ha='center', va='bottom', fontsize=12, fontweight='bold', rotation=0)
    
    for x, val in zip(x_values + width, total_gpu):
        ax.text(x*1.005, val * 1.005, f"{val:.1f}", ha='center', va='bottom', fontsize=12, fontweight='bold', rotation=0)

plt.tight_layout()
plt.show()