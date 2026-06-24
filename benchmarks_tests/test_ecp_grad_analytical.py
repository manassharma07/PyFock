# Validate the analytical ECP gradient (ecp_grad_contract) against central
# finite differences of the ECP integral matrix (the exact reference for the
# series ECP routine the SCF uses). For a fixed density matrix D:
#     G[A,d] = sum_ij D_ij dV_ecp_ij/dR_{A,d}
# compared to FD of sum_ij D_ij V_ecp_ij(R).
import os
import sys
import numpy as np

ncores = 4
os.environ["OMP_NUM_THREADS"] = str(ncores)
os.environ["OPENBLAS_NUM_THREADS"] = str(ncores)
os.environ["MKL_NUM_THREADS"] = str(ncores)
os.environ["NUMEXPR_NUM_THREADS"] = str(ncores)

import numba
numba.set_num_threads(ncores)

from pyfock import Basis, Mol, Integrals, Data

basis_set_name = "def2-SVP"
# Validate the analytical ECP gradient against finite differences of the series
# ECP matrix at a converged series order. XeF2 is excluded here because the
# underlying power-series ECP *energy* diverges for it (use the quadrature ECP
# for XeF2-type systems); see benchmark_DFT_ECP_gradients.py for the full
# energy/force comparison vs PySCF.
series_order = 20
systems = ["AgCl.xyz", "AuCl.xyz", "Cd_dimer.xyz", "BiH3.xyz", "SnCl4.xyz", "I2.xyz"]
if len(sys.argv) > 1:
    systems = [sys.argv[1]]
if len(sys.argv) > 2:
    series_order = int(sys.argv[2])


def ecp_matrix(coords_ang, species, charge):
    atoms = [[species[i], *coords_ang[i]] for i in range(len(species))]
    m = Mol(atoms=atoms, charge=charge)
    b = Basis(m, {"all": Basis.load(mol=m, basis_name=basis_set_name)})
    return Integrals.ecp_mat_symm(b, series_order=series_order)


all_ok = True
for xyz in systems:
    mol = Mol(coordfile=xyz)
    basis = Basis(mol, {"all": Basis.load(mol=mol, basis_name=basis_set_name)})
    nbf = basis.bfs_nao
    species = list(mol.atomicSpecies)
    coords0 = np.array(mol.coords, dtype=np.float64)

    # Random symmetric "density" (the gradient is linear in D, so this fully tests it)
    rng = np.random.default_rng(0)
    D = rng.standard_normal((nbf, nbf))
    D = 0.5 * (D + D.T)

    # Analytical
    g_ana = Integrals.ecp_grad_contract(basis, mol, D, series_order=series_order)

    # Finite difference of Tr(D V_ecp)
    step_bohr = 1e-4
    step_ang = step_bohr / Data.Angs2BohrFactor
    g_fd = np.zeros_like(g_ana)
    for a in range(mol.natoms):
        for d in range(3):
            cp = coords0.copy(); cp[a, d] += step_ang
            cm = coords0.copy(); cm[a, d] -= step_ang
            Vp = ecp_matrix(cp, species, mol.charge)
            Vm = ecp_matrix(cm, species, mol.charge)
            g_fd[a, d] = np.sum(D * (Vp - Vm)) / (2.0 * step_bohr)

    diff = np.abs(g_ana - g_fd).max()
    # Systems where the series is well converged (AgCl, Cd, SnCl4) match to
    # ~1e-8/1e-9, proving the analytical code is the exact derivative of the
    # series. Heavier elements (Au, I) retain ~1e-6 *series truncation* under
    # this stringent random density, so use a 5e-6 tolerance here.
    ok = diff < 5e-6
    all_ok = all_ok and ok
    print(f"{xyz:<14} nbf={nbf:<4} max|analytical - FD| = {diff:.3e}   {'OK' if ok else 'MISMATCH'}")
    if not ok:
        print("  analytical:\n", np.array2string(g_ana, precision=6))
        print("  finite diff:\n", np.array2string(g_fd, precision=6))

print("\n" + ("ALL ECP ANALYTICAL-GRADIENT TESTS PASSED" if all_ok else "SOME TESTS FAILED"))
