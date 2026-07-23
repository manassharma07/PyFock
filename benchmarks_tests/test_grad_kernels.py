# Quick finite-difference validation of the contracted DF gradient kernels
# (3c2e and 2c2e) and the AO value/grad/hess evaluator.
import os
import numpy as np

ncores = 4
os.environ["OMP_NUM_THREADS"] = str(ncores)
os.environ["OPENBLAS_NUM_THREADS"] = str(ncores)
os.environ["MKL_NUM_THREADS"] = str(ncores)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(ncores)
os.environ["NUMEXPR_NUM_THREADS"] = str(ncores)

from pyfock import Basis, Mol, Integrals
import numba
numba.set_num_threads(ncores)

basis_set_name = 'def2-SVP'
auxbasis_name = 'def2-universal-jfit'
xyz = 'H2O.xyz'

mol = Mol(coordfile=xyz)
basis = Basis(mol, {'all': Basis.load(mol=mol, basis_name=basis_set_name)})
auxbasis = Basis(mol, {'all': Basis.load(mol=mol, basis_name=auxbasis_name)})

nbf = basis.bfs_nao
naux = auxbasis.bfs_nao
print('nbf:', nbf, 'naux:', naux)

rng = np.random.default_rng(42)
D = rng.normal(size=(nbf, nbf))
D = 0.5 * (D + D.T)
c = rng.normal(size=naux)

# ----- analytical contracted gradients -----
g3c = Integrals.rys_3c2e_grad_contract(basis, auxbasis, D, c, schwarz=True, threshold_schwarz=1e-13)
g2c = Integrals.rys_2c2e_grad_contract(auxbasis, c)

# ----- finite differences -----
import copy
step = 1e-5  # Bohr
from pyfock import Data
step_ang = step / Data.Angs2BohrFactor

coords0 = np.array(mol.coords, dtype=np.float64)


def build(coords_ang):
    atoms = []
    for ia, sym in enumerate(mol.atomicSpecies):
        atoms.append([sym, *coords_ang[ia]])
    m = Mol(atoms=atoms, charge=mol.charge)
    b = Basis(m, {'all': Basis.load(mol=m, basis_name=basis_set_name)})
    ab = Basis(m, {'all': Basis.load(mol=m, basis_name=auxbasis_name)})
    return m, b, ab


def e3c(coords_ang):
    m, b, ab = build(coords_ang)
    ints = Integrals.rys_3c2e_symm_test(b, ab, schwarz=False)
    return np.einsum('ijp,ij,p->', ints, D, c)


def e2c(coords_ang):
    m, b, ab = build(coords_ang)
    ints = Integrals.rys_2c2e_symm(ab)
    return np.einsum('pq,p,q->', ints, c, c)


g3c_num = np.zeros((mol.natoms, 3))
g2c_num = np.zeros((mol.natoms, 3))
for ia in range(mol.natoms):
    for d in range(3):
        cp = coords0.copy(); cp[ia, d] += step_ang
        cm = coords0.copy(); cm[ia, d] -= step_ang
        g3c_num[ia, d] = (e3c(cp) - e3c(cm)) / (2 * step)
        g2c_num[ia, d] = (e2c(cp) - e2c(cm)) / (2 * step)

print('\n3c2e contracted gradient (analytical):\n', g3c)
print('3c2e contracted gradient (numerical):\n', g3c_num)
print('3c2e max abs diff:', np.abs(g3c - g3c_num).max())

print('\n2c2e contracted gradient (analytical):\n', g2c)
print('2c2e contracted gradient (numerical):\n', g2c_num)
print('2c2e max abs diff:', np.abs(g2c - g2c_num).max())

# ----- nuclear attraction contracted gradient vs reference contraction -----
g_nuc = Integrals.rys_nuc_grad_contract(basis, mol, D, schwarz=True, threshold=1e-14)
dV_ref = Integrals.nuc_mat_grad_r_symm(basis, mol, wrt_atoms=True)
g_nuc_ref = np.einsum('adij,ij->ad', dV_ref, D)
print('\nNuclear contracted gradient (rys):\n', g_nuc)
print('Nuclear contracted gradient (reference):\n', g_nuc_ref)
print('Nuclear grad max abs diff:', np.abs(g_nuc - g_nuc_ref).max())

# ----- AO grad/hess check via finite differences of value/grad -----
coords_test = rng.normal(size=(20, 3)) * 1.5 + np.array([1.0, 0.0, 0.0])
bfs_coords = np.array([basis.bfs_coords])
bfs_contr_prim_norms = np.array([basis.bfs_contr_prim_norms])
bfs_lmn = np.array([basis.bfs_lmn])
bfs_nprim = np.array([basis.bfs_nprim])
maxnprim = max(basis.bfs_nprim)
bfs_coeffs = np.zeros([nbf, maxnprim]); bfs_expnts = np.zeros([nbf, maxnprim]); bfs_prim_norms = np.zeros([nbf, maxnprim])
for i in range(nbf):
    for j in range(basis.bfs_nprim[i]):
        bfs_coeffs[i, j] = basis.bfs_coeffs[i][j]
        bfs_expnts[i, j] = basis.bfs_expnts[i][j]
        bfs_prim_norms[i, j] = basis.bfs_prim_norms[i][j]
idx = np.arange(nbf)

val, grad, hess = Integrals.bf_val_helpers.eval_bfs_grad_and_hess_sparse_internal_serial(
    bfs_coords[0], bfs_contr_prim_norms[0], bfs_nprim[0], bfs_lmn[0], bfs_coeffs, bfs_prim_norms, bfs_expnts, coords_test, idx)

val_ref, grad_ref = Integrals.bf_val_helpers.eval_bfs_and_grad_sparse_internal_serial(
    bfs_coords[0], bfs_contr_prim_norms[0], bfs_nprim[0], bfs_lmn[0], bfs_coeffs, bfs_prim_norms, bfs_expnts, coords_test, idx)

print('\nAO value diff vs existing evaluator:', np.abs(val - val_ref).max())
print('AO grad diff vs existing evaluator:', np.abs(grad - grad_ref).max())

# hess via FD of grad
h = 1e-6
hess_num = np.zeros((6, coords_test.shape[0], nbf))
comp = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]
for ic, (a, b) in enumerate(comp):
    cp = coords_test.copy(); cp[:, b] += h
    cm = coords_test.copy(); cm[:, b] -= h
    _, gp = Integrals.bf_val_helpers.eval_bfs_and_grad_sparse_internal_serial(
        bfs_coords[0], bfs_contr_prim_norms[0], bfs_nprim[0], bfs_lmn[0], bfs_coeffs, bfs_prim_norms, bfs_expnts, cp, idx)
    _, gm = Integrals.bf_val_helpers.eval_bfs_and_grad_sparse_internal_serial(
        bfs_coords[0], bfs_contr_prim_norms[0], bfs_nprim[0], bfs_lmn[0], bfs_coeffs, bfs_prim_norms, bfs_expnts, cm, idx)
    hess_num[ic] = (gp[a] - gm[a]) / (2 * h)

print('AO hess max abs diff vs FD:', np.abs(hess - hess_num).max())
print('AO hess max abs value:', np.abs(hess).max())
