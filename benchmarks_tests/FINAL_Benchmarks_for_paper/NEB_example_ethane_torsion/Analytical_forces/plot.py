import numpy as np
import matplotlib.pyplot as plt

# ---------------------------
# Data from your NEB results
# ---------------------------
dihedral = np.array([180.0, 183.9, 203.8, 221.5, 240.0, 256.2, 273.0, 293.9, 300.0])
energy_kcal = np.array([0.0000, 0.0042, 0.9511, 2.1880, 2.7791, 2.3167, 1.1827, 0.0483, 0.0000])

# ---------------------------
# Create smooth curve (optional)
# ---------------------------
from scipy.interpolate import make_interp_spline

x_smooth = np.linspace(dihedral.min(), dihedral.max(), 300)
spline = make_interp_spline(dihedral, energy_kcal, k=3)
y_smooth = spline(x_smooth)

# ---------------------------
# Plot styling
# ---------------------------
plt.figure(figsize=(8, 6), dpi=300)

# Main curve
plt.plot(x_smooth, y_smooth, linewidth=3)

# Data points
plt.scatter(dihedral, energy_kcal, s=60, zorder=3)

# Highlight key points
plt.scatter([180, 300], [0, 0], s=120, marker='o', zorder=4, label='Staggered')
plt.scatter([240], [2.7791], s=120, marker='^', zorder=4, label='Eclipsed (TS)')

# Labels and title
plt.xlabel("Dihedral Angle (°)", fontsize=16, fontweight='bold')
plt.ylabel("Relative Energy (kcal/mol)", fontsize=16, fontweight='bold')
plt.title("Ethane Torsional Barrier", fontsize=18, fontweight='bold')

# Ticks
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

# Grid
plt.grid(True, linestyle='--', linewidth=0.8, alpha=0.6)

# Annotation for barrier
# plt.annotate(
#     "Barrier ≈ 2.78 kcal/mol",
#     xy=(240, 2.78),
#     xytext=(245, 2.2),
#     arrowprops=dict(arrowstyle="->", linewidth=1.5),
#     fontsize=12,
#     fontweight='bold'
# )

# Legend
plt.legend(fontsize=12, frameon=False)

plt.xlim(160, 320)

# Tight layout
plt.tight_layout()

# Save figure
plt.savefig("ethane_torsional_barrier.png", dpi=600)
# plt.savefig("ethane_torsional_barrier.pdf")

plt.show()