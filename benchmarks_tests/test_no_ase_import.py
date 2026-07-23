# Verify that pyfock works without ASE installed (ASE is optional):
# blocks all 'ase' imports via a meta-path hook, then imports pyfock,
# computes integrals and runs a small DFT SCF, and finally checks that
# accessing PyFockCalculator raises a helpful ImportError.
import importlib.abc
import os
import sys

ncores = 2
os.environ["OMP_NUM_THREADS"] = str(ncores)
os.environ["OPENBLAS_NUM_THREADS"] = str(ncores)
os.environ["MKL_NUM_THREADS"] = str(ncores)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(ncores)
os.environ["NUMEXPR_NUM_THREADS"] = str(ncores)


class AseBlocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "ase" or fullname.startswith("ase."):
            raise ImportError(f"No module named {fullname!r} (blocked to simulate missing ASE)")
        return None


assert "pyfock" not in sys.modules, "run this test in a fresh interpreter"
for mod in list(sys.modules):
    if mod == "ase" or mod.startswith("ase."):
        del sys.modules[mod]
sys.meta_path.insert(0, AseBlocker())

# 1. ASE really is blocked
try:
    import ase  # noqa: F401
    raise AssertionError("ase import should have been blocked")
except ImportError:
    pass

# 2. pyfock imports fine without ASE
import numpy as np
from pyfock import Basis, DFT, DFT_Grad, Mol, Integrals  # noqa: F401
print("pyfock imported successfully without ASE")

# 3. Integrals work
mol = Mol(atoms=[["H", 0.0, 0.0, 0.0], ["H", 0.0, 0.0, 0.74]])
basis = Basis(mol, {"all": Basis.load(mol=mol, basis_name="sto-3g")})
S = Integrals.overlap_mat_symm(basis)
T = Integrals.kin_mat_symm(basis)
print("Overlap/kinetic integrals OK, S shape:", S.shape)

# 4. A small DFT SCF works
dft_obj = DFT(mol, basis, xc="LDA", gridsLevel=3)
dft_obj.conv_crit = 1e-6
dft_obj.ncores = ncores
energy, dmat = dft_obj.scf()
print("DFT SCF without ASE OK, energy:", energy)
assert dft_obj.converged

# 5. Analytical gradients work
res = DFT_Grad(dft_obj, verbose=False).calculate()
print("Analytical forces without ASE OK:\n", np.asarray(res["forces"]))

# 6. PyFockCalculator access gives a helpful error
try:
    from pyfock import PyFockCalculator  # noqa: F401
    raise AssertionError("PyFockCalculator import should have failed without ASE")
except ImportError as exc:
    message = str(exc)
    print("PyFockCalculator error message:", message)
    assert "pip install ase" in message

print("\nNO-ASE TEST PASSED")
