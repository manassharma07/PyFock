# Validate the GPU 1e-gradient CUDA kernels using Numba's CUDA SIMULATOR
# (NUMBA_ENABLE_CUDASIM=1), which runs cuda.jit code on the CPU. This lets us
# check the kernel logic and math WITHOUT a physical GPU (no CuPy needed).
#
# The simulator is slow (pure-Python per thread), so use a tiny system.
# We call the internal *_cuda kernels directly with numpy arrays (the host
# wrappers are CuPy-only), and compare against the CPU reference routines.
import os
os.environ["NUMBA_ENABLE_CUDASIM"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
from numba import cuda

# The CUDA simulator's jit() does not accept the `max_registers` kwarg that the
# real-GPU kernels use (it is a hardware-only hint). Strip it so the modules
# import and run under the simulator. This only affects this CPU test harness.
_orig_jit = cuda.jit


def _patched_jit(*args, **kwargs):
    kwargs.pop('max_registers', None)
    return _orig_jit(*args, **kwargs)


cuda.jit = _patched_jit

import importlib
from pyfock import Basis, Mol, Integrals
# Use importlib because the package re-exports same-named functions that shadow
# the submodule attributes.
ov_mod = importlib.import_module('pyfock.Integrals.overlap_mat_grad_r_symm_cupy')
kin_mod = importlib.import_module('pyfock.Integrals.kin_mat_grad_r_symm_cupy')
nuc_mod = importlib.import_module('pyfock.Integrals.nuc_mat_grad_r_symm_cupy')


def pack(basis):
    nbf = basis.bfs_nao
    maxnprim = max(basis.bfs_nprim)
    coeffs = np.zeros((nbf, maxnprim))
    expnts = np.zeros((nbf, maxnprim))
    prim_norms = np.zeros((nbf, maxnprim))
    for i in range(nbf):
        for j in range(basis.bfs_nprim[i]):
            coeffs[i, j] = basis.bfs_coeffs[i][j]
            expnts[i, j] = basis.bfs_expnts[i][j]
            prim_norms[i, j] = basis.bfs_prim_norms[i][j]
    return (
        np.array(basis.bfs_coords, dtype=np.float64),
        np.array(basis.bfs_contr_prim_norms, dtype=np.float64),
        np.array(basis.bfs_lmn, dtype=np.int64),
        np.array(basis.bfs_nprim, dtype=np.int64),
        coeffs, prim_norms, expnts,
    )


xyz = "H2O.xyz"
basis_name = "sto-3g"   # s and p functions; tiny (7 bfs) so the simulator finishes quickly
mol = Mol(coordfile=xyz)
basis = Basis(mol, {"all": Basis.load(mol=mol, basis_name=basis_name)})
nbf = basis.bfs_nao
print(f"CUDA simulator validation on {xyz} / {basis_name} (nbf={nbf})")

coords, cpn, lmn, nprim, coeffs, prim_norms, expnts = pack(basis)
a, b, c, d = 0, nbf, 0, nbf
both = True  # full symmetric matrix

tpb = (8, 8)
bpg = ((nbf + 7) // 8, (nbf + 7) // 8)

# ---------------- Overlap ----------------
dS_cpu = Integrals.overlap_mat_grad_r_symm(basis)
dS_sim = np.zeros((3, nbf, nbf))
ov_mod.overlap_mat_grad_r_symm_internal_cuda[bpg, tpb](
    coords, cpn, lmn, nprim, coeffs, prim_norms, expnts, a, b, c, d,
    False, False, both, False, dS_sim)
print("overlap  dS_r  max|SIM-CPU| =", np.abs(dS_cpu - dS_sim).max())

# ---------------- Kinetic ----------------
dT_cpu = Integrals.kin_mat_grad_r_symm(basis)
dT_sim = np.zeros((3, nbf, nbf))
kin_mod.kin_mat_grad_r_symm_internal_cuda[bpg, tpb](
    coords, cpn, lmn, nprim, coeffs, prim_norms, expnts, a, b, c, d,
    False, False, both, False, dT_sim)
print("kinetic  dT_r  max|SIM-CPU| =", np.abs(dT_cpu - dT_sim).max())

# ---------------- Nuclear ----------------
dV_cpu = Integrals.nuc_mat_grad_r_symm(basis, mol)
dV_sim = np.zeros((3, nbf, nbf))
Z = np.array(mol.Zcharges, dtype=np.float64)
coordsBohrs = np.array(mol.coordsBohrs, dtype=np.float64)
dummy = np.zeros((1, 1), dtype=np.float64)
nuc_mod.nuc_mat_grad_r_symm_internal_cuda[bpg, tpb](
    coords, cpn, lmn, nprim, coeffs, prim_norms, expnts, a, b, c, d,
    Z, coordsBohrs, mol.natoms, False, False, both, False, dummy, False, dV_sim)
print("nuclear  dV_r  max|SIM-CPU| =", np.abs(dV_cpu - dV_sim).max())

ok = (np.abs(dS_cpu - dS_sim).max() < 1e-9 and
      np.abs(dT_cpu - dT_sim).max() < 1e-9 and
      np.abs(dV_cpu - dV_sim).max() < 1e-8)
print("\n" + ("CUDA-SIMULATOR VALIDATION PASSED" if ok else "CUDA-SIMULATOR VALIDATION FAILED"))
