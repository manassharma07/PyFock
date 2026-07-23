# End-to-end test of the ASE calculator with analytical forces:
#   1. default force_mode="analytical" works and matches numerical forces
#   2. unsupported configs (HF) automatically fall back to numerical
#   3. a short geometry optimization runs with analytical forces
import shutil
import numpy as np
from timeit import default_timer as timer

from ase import Atoms
from ase.optimize import BFGS

from pyfock import PyFockCalculator

water_positions = [
    [0.000, 0.000, 0.119],
    [0.000, 0.763, -0.477],
    [0.000, -0.763, -0.477],
]


def make_water():
    return Atoms(symbols="OHH", positions=[list(p) for p in water_positions])


common = dict(
    functional="PBE",
    basis="def2-SVP",
    auxbasis="def2-universal-jfit",
    ncores=4,
    DF=True,
    save_ao_values=True,
    sao=True,
    conv_crit=1e-9,
)

# ---------- 1a. analytical (default) ----------
shutil.rmtree("ase_test_ana", ignore_errors=True)
water = make_water()
water.calc = PyFockCalculator(directory="ase_test_ana", **common)
start = timer()
e_ana = water.get_potential_energy()
f_ana = water.get_forces()
t_ana = timer() - start
method_ana = water.calc.pyfock_results.get("force_method_used")
print("\n[analytical] method used:", method_ana, " time: %.2f s" % t_ana)
assert method_ana == "analytical", f"expected analytical, got {method_ana}"

# ---------- 1b. numerical (explicit) ----------
shutil.rmtree("ase_test_num", ignore_errors=True)
water = make_water()
water.calc = PyFockCalculator(directory="ase_test_num", force_mode="numerical", **common)
start = timer()
e_num = water.get_potential_energy()
f_num = water.get_forces()
t_num = timer() - start
method_num = water.calc.pyfock_results.get("force_method_used")
print("[numerical ] method used:", method_num, " time: %.2f s" % t_num)
assert method_num == "numerical", f"expected numerical, got {method_num}"

print("\nEnergy diff (eV):", abs(e_ana - e_num))
print("Max force diff analytical vs numerical (eV/Ang):", np.abs(f_ana - f_num).max())
assert np.abs(f_ana - f_num).max() < 1e-3, "analytical and numerical ASE forces disagree"

# ---------- 2. HF fallback ----------
shutil.rmtree("ase_test_hf", ignore_errors=True)
h2 = Atoms(symbols="HH", positions=[[0, 0, 0], [0, 0, 0.74]])
h2.calc = PyFockCalculator(
    directory="ase_test_hf",
    functional="HF",
    basis="sto-3g",
    ncores=4,
    conv_crit=1e-9,
)
f_hf = h2.get_forces()
method_hf = h2.calc.pyfock_results.get("force_method_used")
print("\n[HF fallback] method used:", method_hf)
print("HF forces (eV/Ang):\n", f_hf)
assert method_hf == "numerical", f"expected numerical fallback for HF, got {method_hf}"

# ---------- 3. short BFGS optimization with analytical forces ----------
shutil.rmtree("ase_test_opt", ignore_errors=True)
water = make_water()
water.calc = PyFockCalculator(directory="ase_test_opt", **common)
start = timer()
opt = BFGS(water, logfile="-")
converged = opt.run(fmax=0.02, steps=15)
t_opt = timer() - start
f_final = water.get_forces()
fmax_final = np.sqrt((f_final**2).sum(axis=1)).max()
print("\nOptimization converged:", converged)
print("Final fmax (eV/Ang):", fmax_final)
print("Optimization wall time: %.1f s" % t_opt)
assert converged and fmax_final < 0.02, "BFGS did not converge within 15 steps"

print("\nALL ASE TESTS PASSED")
