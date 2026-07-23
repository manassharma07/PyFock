# Full analytical MGGA force validation:
#   - native PyFock MGGA: analytical vs PyFock numerical (same native energy surface)
#   - pylibxc MGGA      : analytical vs PySCF analytical (independent reference)
# Functional: TPSS (and optionally r2SCAN).
import os
import sys
import numpy as np
from timeit import default_timer as timer

ncores = 4
os.environ["OMP_NUM_THREADS"] = str(ncores)
os.environ["OPENBLAS_NUM_THREADS"] = str(ncores)
os.environ["MKL_NUM_THREADS"] = str(ncores)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(ncores)
os.environ["NUMEXPR_NUM_THREADS"] = str(ncores)

from pyfock import Basis, DFT, DFT_Grad, DFT_NumGrad, Mol
from pyscf import dft as pyscf_dft
from pyscf import gto

basis_set_name = "def2-SVP"
auxbasis_name = "def2-universal-jfit"
xyz_filename = "H2O.xyz"

# functional name -> (pyscf xc string, pyfock funcid list)
FUNCS = {
    "TPSS": ("TPSS", [202, 231]),
}
if len(sys.argv) > 1 and sys.argv[1] == "r2scan":
    FUNCS = {"r2SCAN": ("R2SCAN", [497, 498])}


def build_pyscf(xc_string):
    mol_pyscf = gto.Mole()
    mol_pyscf.atom = xyz_filename
    mol_pyscf.basis = basis_set_name
    mol_pyscf.cart = False
    mol_pyscf.verbose = 0
    mol_pyscf.build()
    mf = pyscf_dft.rks.RKS(mol_pyscf).density_fit(auxbasis=auxbasis_name)
    mf.xc = xc_string
    mf.init_guess = "atom"
    dm0 = mf.init_guess_by_atom(mol_pyscf)
    mf.conv_tol = 1e-11
    mf.grids.level = 3
    energy = mf.kernel(dm0=dm0)
    forces = -mf.nuc_grad_method().kernel()
    return energy, forces, mf.grids, dm0


def build_pyfock(funcid, grids, dm0, use_libxc):
    mol = Mol(coordfile=xyz_filename)
    basis = Basis(mol, {"all": Basis.load(mol=mol, basis_name=basis_set_name)})
    auxbasis = Basis(mol, {"all": Basis.load(mol=mol, basis_name=auxbasis_name)})
    dft_obj = DFT(mol, basis, auxbasis, xc=funcid, grids=grids)
    dft_obj.conv_crit = 1e-11
    dft_obj.max_itr = 60
    dft_obj.ncores = ncores
    dft_obj.save_ao_values = True
    dft_obj.isDF = True
    dft_obj.DF_algo = 10
    dft_obj.XC_algo = 2
    dft_obj.sortGrids = False
    dft_obj.xc_bf_screen = True
    dft_obj.threshold_schwarz = 1e-9
    dft_obj.strict_schwarz = False
    dft_obj.cholesky = True
    dft_obj.orthogonalize = True
    dft_obj.sao = True
    dft_obj.use_gpu = False
    dft_obj.use_libxc = use_libxc
    dft_obj.dmat = dm0
    energy, _ = dft_obj.scf()
    return dft_obj, energy


for name, (xc_string, funcid) in FUNCS.items():
    print("\n" + "=" * 70)
    print(f"Functional: {name}  (pyfock funcid {funcid})")
    print("=" * 70)

    energy_pyscf, forces_pyscf, grids, dm0 = build_pyscf(xc_string)

    # ---- native PyFock MGGA ----
    dft_native, e_native = build_pyfock(funcid, grids, dm0, use_libxc=False)
    start = timer()
    res_native = DFT_Grad(dft_native, verbose=False).calculate()
    DFT_Grad(dft_native, verbose=False).calculate()  # warm
    t_ana = timer() - start
    f_native = res_native["forces"]

    # numerical reference on the native surface
    num = DFT_NumGrad(dft_native, step_size=1e-3, step_unit="bohr", method="central",
                      use_fixed_grids=True, verbose=False)
    f_native_num = num.calculate()["forces"]

    print(f"\n[native]  E(PyFock) - E(PySCF) = {e_native - energy_pyscf:.3e}")
    print(f"[native]  max |F_analytical - F_numerical| = {np.abs(f_native - f_native_num).max():.3e}")
    print(f"[native]  max |F_analytical - F_PySCF|     = {np.abs(f_native - forces_pyscf).max():.3e}")

    # ---- pylibxc PyFock MGGA ----
    dft_libxc, e_libxc = build_pyfock(funcid, grids, dm0, use_libxc=True)
    res_libxc = DFT_Grad(dft_libxc, verbose=False).calculate()
    f_libxc = res_libxc["forces"]

    print(f"\n[libxc ]  E(PyFock) - E(PySCF) = {e_libxc - energy_pyscf:.3e}")
    print(f"[libxc ]  max |F_analytical - F_PySCF|     = {np.abs(f_libxc - forces_pyscf).max():.3e}")

    print("\nPySCF forces (Ha/Bohr):\n", np.array2string(forces_pyscf, precision=8))
    print("PyFock native analytical forces:\n", np.array2string(f_native, precision=8))
    print("PyFock libxc  analytical forces:\n", np.array2string(f_libxc, precision=8))

    assert np.abs(f_native - f_native_num).max() < 5e-5, "native analytical vs numerical too large"
    assert np.abs(f_libxc - forces_pyscf).max() < 5e-5, "libxc analytical vs PySCF too large"

print("\nALL MGGA FORCE TESTS PASSED")
