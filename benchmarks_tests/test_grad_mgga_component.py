# Component-level finite-difference validation of the analytical MGGA XC
# gradient (tau term) at FIXED density matrix. Compares -2*sum_{mu in A}
# dexc_dbf  against the finite difference of Exc (eval_xc_2) for a meta-GGA
# functional, using both the native PyFock functionals and pylibxc.
import os
import sys
import numpy as np

ncores = 4
os.environ["OMP_NUM_THREADS"] = str(ncores)
os.environ["OPENBLAS_NUM_THREADS"] = str(ncores)
os.environ["MKL_NUM_THREADS"] = str(ncores)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(ncores)
os.environ["NUMEXPR_NUM_THREADS"] = str(ncores)

import numba
numba.set_num_threads(ncores)

from pyfock import Basis, Mol, Integrals, Data
from pyscf import dft as pyscf_dft
from pyscf import gto

basis_set_name = "def2-SVP"
xyz_filename = "H2O.xyz"
funcid = [202, 231]  # MGGA_X_TPSS + MGGA_C_TPSS

# Generate a grid via PySCF and a plausible (converged-ish) density via PySCF
mol_pyscf = gto.Mole()
mol_pyscf.atom = xyz_filename
mol_pyscf.basis = basis_set_name
mol_pyscf.cart = True   # use Cartesian to match PyFock CAO directly
mol_pyscf.verbose = 0
mol_pyscf.build()
mf = pyscf_dft.rks.RKS(mol_pyscf)
mf.xc = "TPSS"
mf.grids.level = 3
mf.conv_tol = 1e-10
mf.kernel()
grids = mf.grids
D = mf.make_rdm1()  # Cartesian AO density (symmetric)

mol = Mol(coordfile=xyz_filename)
basis = Basis(mol, {"all": Basis.load(mol=mol, basis_name=basis_set_name)})
assert basis.bfs_nao == D.shape[0], (basis.bfs_nao, D.shape)

coords_grid = np.asarray(grids.coords)
weights_grid = np.asarray(grids.weights)
ngrids = coords_grid.shape[0]
blocksize = 5000
nblocks = ngrids // blocksize
bfs_atoms = np.asarray(basis.bfs_atoms, dtype=np.int64)


def analytical_grad_xc(use_libxc):
    lnz, cnz = Integrals.bf_val_helpers.nonzero_ao_indices(basis, coords_grid, blocksize, nblocks, ngrids)
    dexc_dbf = Integrals.eval_xc_grad_2(basis, D, weights_grid, coords_grid, funcid=funcid,
                                        use_libxc=use_libxc, ncores=ncores, blocksize=blocksize,
                                        list_nonzero_indices=lnz, count_nonzero_indices=cnz)
    grad = np.zeros((mol.natoms, 3))
    np.add.at(grad, bfs_atoms, -2.0 * dexc_dbf.T)
    return grad


def exc_energy(coords_ang, use_libxc):
    m = Mol(atoms=[[s, *coords_ang[i]] for i, s in enumerate(mol.atomicSpecies)], charge=mol.charge)
    b = Basis(m, {"all": Basis.load(mol=m, basis_name=basis_set_name)})
    lnz, cnz = Integrals.bf_val_helpers.nonzero_ao_indices(b, coords_grid, blocksize, nblocks, ngrids)
    exc, _ = Integrals.eval_xc_2(b, D, weights_grid, coords_grid, funcid, use_libxc,
                                 ncores=ncores, blocksize=blocksize, list_nonzero_indices=lnz,
                                 count_nonzero_indices=cnz, print_nelec=False)
    return exc


coords0 = np.array(mol.coords, dtype=np.float64)
step = 1e-4
step_ang = step / Data.Angs2BohrFactor

for use_libxc in (False, True):
    label = "libxc" if use_libxc else "native"
    grad_ana = analytical_grad_xc(use_libxc)
    targets = [(0, 2), (1, 0), (2, 1)]
    print(f"\n[{label}] XC gradient component check (fixed D):")
    max_diff = 0.0
    for (ia, d) in targets:
        cp = coords0.copy(); cp[ia, d] += step_ang
        cm = coords0.copy(); cm[ia, d] -= step_ang
        fd = (exc_energy(cp, use_libxc) - exc_energy(cm, use_libxc)) / (2 * step)
        ana = grad_ana[ia, d]
        diff = abs(fd - ana)
        max_diff = max(max_diff, diff)
        print(f"  atom {ia} dir {d}: FD={fd:.10f}  analytical={ana:.10f}  diff={diff:.2e}")
    print(f"  [{label}] max abs diff: {max_diff:.2e}")
