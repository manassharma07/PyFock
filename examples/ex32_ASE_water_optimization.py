from pathlib import Path

from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write, read
from ase.optimize import BFGS

from pyfock import PyFockCalculator


water = Atoms(
    symbols="OHH",
    positions=[
        [0.000, 0.000, 0.119],
        [0.000, 0.763, -0.477],
        [0.000, -0.763, -0.477],
    ],
)

calc = PyFockCalculator(
    functional="PBE",
    basis="def2-SVP",
    auxbasis="def2-universal-jfit",
    # Set dispersion=True and install torch-dftd to add D3 corrections.
    # dispersion=True,
    # dispersion_kwargs={"damping": "bj", "device": "cpu"},
    ncores=4,
    DF=True,
    save_ao_values=True,
    sao = True,
    conv_crit=1e-7,
    directory="ase_water_optimization",
)
water.calc = calc

energy = water.get_potential_energy()
forces = water.get_forces()

print(f"Initial energy: {energy:.12f} eV")
print("Initial forces (eV/Å):")
for sym, f in zip(water.get_chemical_symbols(), forces):
    print(f"{sym:2s}  {f[0]:12.6f}  {f[1]:12.6f}  {f[2]:12.6f}")
print(f"Initial HOMO-LUMO gap: {calc.get_homo_lumo_gap():.6f} eV")
print("PyFock energy components:", calc.pyfock_results)

# Geometry optimization
print("Starting geometry optimization...\n")

opt = BFGS(
    water,
    trajectory="ase_water_optimization/water_opt.traj",
    # logfile="ase_water_optimization/opt.log"
)

opt.run(fmax=0.02)

print("\nOptimization finished.\n")

# Final energy
energy_final = water.get_potential_energy()
print(f"Final energy: {energy_final:.6f} eV")
print()

# Optimized coordinates
print("Optimized geometry (Å):")
for sym, pos in zip(water.get_chemical_symbols(), water.positions):
    print(f"{sym:2s}  {pos[0]:10.6f}  {pos[1]:10.6f}  {pos[2]:10.6f}")

print()

# Final forces
forces = water.get_forces()
print("Final forces (eV/Å):")
for sym, f in zip(water.get_chemical_symbols(), forces):
    print(f"{sym:2s}  {f[0]:12.6f}  {f[1]:12.6f}  {f[2]:12.6f}")

print()
print(f"Maximum force component: {abs(forces).max():.6f} eV/Å")

# Save the optimized structure and calculation results in extXYZ format
write("ase_water_optimization/water_optimized.extxyz", water)

# Convert trajectory to extxyz
traj = read("ase_water_optimization/water_opt.traj", index=":")
write("ase_water_optimization/water_opt.extxyz", traj, format="extxyz")
