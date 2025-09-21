import matplotlib.pyplot as plt
import numpy as np

# ==== CONFIG FLAGS ====
x_axis_choice = "water"  # "water" or "basis"
log_scale = False        # True for log scale, False for linear
plot_total_only = True  # True for just total times, False for breakdown

# ==== DATA ====
water_molecules = np.array([47, 76, 100, 139])
basis_functions = np.array([1175, 1900, 2500, 3475])

# PySCF data
total_pyscf = np.array([28.77391837, 67.70287694, 129.9493543, 262.5715926])

# GPU/CPU data
total_gpu_old = np.array([1.815790464, 4.110073972, 7.665055265, 14.48421162])
total_gpu = np.array([1.53661779, 3.292493198, 5.90542049, 10.71981104])
total_cpu = np.array([17.15226698, 41.32841003, 82.38475342, 154.2059581])

j_gpu_old = np.array([0.896369033, 2.172111774, 4.52707693, 8.54460349])
j_gpu = np.array([0.784328794, 1.87488702, 3.823785722, 7.140085827])
j_cpu = np.array([10.75980461, 24.7638173, 54.06537282, 97.10462831])

xc_gpu_old = np.array([0.48043671, 0.991713577, 1.396709837, 2.243173538])
xc_gpu = np.array([0.413739246, 0.780230838, 1.078757818, 1.639881885])
xc_cpu = np.array([3.76085967, 8.61741557, 12.6850703, 21.373439])

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
    
    bars_cpu = ax.bar(x_values, j_cpu, width, label="PyFock J (CPU)", color="tab:orange", edgecolor='black', linewidth=1.2)
    ax.bar(x_values, xc_cpu, width, bottom=j_cpu, label="PyFock XC (CPU)", color="tab:red", edgecolor='black', linewidth=1.2)
    ax.bar(x_values, other_cpu, width, bottom=j_cpu+xc_cpu, label="PyFock Other (CPU)", color="tab:gray", edgecolor='black', linewidth=1.2)

    bars_gpu = ax.bar(x_values + width, j_gpu, width, label="PyFock J (GPU)", color="tab:orange", alpha=0.6, edgecolor='black', linewidth=1.2)
    ax.bar(x_values + width, xc_gpu, width, bottom=j_gpu, label="PyFock XC (GPU)", color="tab:red", alpha=0.6, edgecolor='black', linewidth=1.2)
    ax.bar(x_values + width, other_gpu, width, bottom=j_gpu+xc_gpu, label="PyFock Other (GPU)", color="tab:gray", alpha=0.6, edgecolor='black', linewidth=1.2)

# Labels & settings
ax.set_xlabel(x_label, fontsize=16, fontweight='bold')
ax.set_ylabel("Time per Iteration (s)", fontsize=16, fontweight='bold')

if log_scale:
    ax.set_yscale("log")

title = "PySCF vs PyFock Timing Breakdown" if not plot_total_only else "PySCF vs PyFock Total Time per Iteration"
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
        ax.text(x, val * 1.01, f"{val:.0f}", ha='center', va='bottom', fontsize=12, fontweight='bold', rotation=0)

    for x, val in zip(x_values, total_cpu):
        ax.text(x, val * 1.01, f"{val:.0f}", ha='center', va='bottom', fontsize=12, fontweight='bold', rotation=0)
    
    for x, val in zip(x_values + width, total_gpu):
        ax.text(x, val * 1.01, f"{val:.0f}", ha='center', va='bottom', fontsize=12, fontweight='bold', rotation=0)
else:
    # For breakdown view, annotate total times
    for x, val in zip(x_values - width, total_pyscf):
        ax.text(x, val * 1.01, f"{val:.0f}", ha='center', va='bottom', fontsize=12, fontweight='bold', rotation=0)

    for x, val in zip(x_values, total_cpu):
        ax.text(x, val * 1.01, f"{val:.0f}", ha='center', va='bottom', fontsize=12, fontweight='bold', rotation=0)
    
    for x, val in zip(x_values + width, total_gpu):
        ax.text(x, val * 1.01, f"{val:.0f}", ha='center', va='bottom', fontsize=12, fontweight='bold', rotation=0)

plt.tight_layout()
plt.show()