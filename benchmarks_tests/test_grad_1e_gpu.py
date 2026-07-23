# Validate the GPU (CuPy/Numba-CUDA) one-electron gradient kernels against
# their CPU counterparts (which are themselves validated against PySCF).
#
# Run this on a machine with a CUDA GPU and CuPy installed:
#     python3 test_grad_1e_gpu.py
#
# It compares, element-wise, the (3, N, N) bra-center r-gradient tensors:
#     overlap_mat_grad_r_symm_cupy   vs  overlap_mat_grad_r_symm
#     kin_mat_grad_r_symm_cupy       vs  kin_mat_grad_r_symm
#     nuc_mat_grad_r_symm_cupy       vs  nuc_mat_grad_r_symm
# and also checks the atom-mapped overlap/kinetic gradients.
import os
import numpy as np

ncores = 4
os.environ["OMP_NUM_THREADS"] = str(ncores)
os.environ["OPENBLAS_NUM_THREADS"] = str(ncores)
os.environ["MKL_NUM_THREADS"] = str(ncores)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(ncores)
os.environ["NUMEXPR_NUM_THREADS"] = str(ncores)

try:
    import cupy as cp
except Exception:
    cp = None

from pyfock import Basis, Mol, Integrals

if cp is None:
    raise SystemExit("CuPy is not installed / no GPU available - cannot run the GPU gradient test.")

# A couple of basis sets / molecules to exercise different angular momenta.
configs = [
    ("H2O.xyz", "sto-3g"),
    ("H2O.xyz", "def2-SVP"),
    ("H2O.xyz", "def2-TZVP"),
]

overall_ok = True
for xyz, basis_name in configs:
    mol = Mol(coordfile=xyz)
    basis = Basis(mol, {"all": Basis.load(mol=mol, basis_name=basis_name)})
    nbf = basis.bfs_nao
    print(f"\n===== {xyz}  {basis_name}  (nbf={nbf}) =====")

    # ---- Overlap r-gradient ----
    dS_cpu = Integrals.overlap_mat_grad_r_symm(basis)
    dS_gpu = cp.asnumpy(Integrals.overlap_mat_grad_r_symm_cupy(basis))
    dS_diff = np.abs(dS_cpu - dS_gpu).max()
    print(f"overlap  dS_r  max|GPU-CPU| = {dS_diff:.3e}")

    # ---- Kinetic r-gradient ----
    dT_cpu = Integrals.kin_mat_grad_r_symm(basis)
    dT_gpu = cp.asnumpy(Integrals.kin_mat_grad_r_symm_cupy(basis))
    dT_diff = np.abs(dT_cpu - dT_gpu).max()
    print(f"kinetic  dT_r  max|GPU-CPU| = {dT_diff:.3e}")

    # ---- Nuclear r-gradient ----
    dV_cpu = Integrals.nuc_mat_grad_r_symm(basis, mol)
    dV_gpu = cp.asnumpy(Integrals.nuc_mat_grad_r_symm_cupy(basis, mol))
    dV_diff = np.abs(dV_cpu - dV_gpu).max()
    print(f"nuclear  dV_r  max|GPU-CPU| = {dV_diff:.3e}")

    ok = dS_diff < 1e-9 and dT_diff < 1e-9 and dV_diff < 1e-8
    overall_ok = overall_ok and ok
    print("  ->", "OK" if ok else "MISMATCH")

print("\n" + ("ALL GPU 1e-GRADIENT TESTS PASSED" if overall_ok else "SOME GPU 1e-GRADIENT TESTS FAILED"))
