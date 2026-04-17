from pyfock import Basis
from pyfock import Mol
from pyfock import Integrals
from timeit import default_timer as timer
import numpy as np
import os

from pyscf import gto
import numba

ncores = 4
bench_GPU = False

numba.set_num_threads(ncores)
os.environ['OMP_NUM_THREADS'] = str(ncores)
os.environ["OPENBLAS_NUM_THREADS"] = str(ncores)
os.environ["MKL_NUM_THREADS"] = str(ncores)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(ncores)
os.environ["NUMEXPR_NUM_THREADS"] = str(ncores)

basis_set_name = 'sto-3g'
xyzFilename = 'H2O.xyz'

mol = Mol(coordfile=xyzFilename)
basis = Basis(mol, {'all': Basis.load(mol=mol, basis_name=basis_set_name)})

print('\n\n\n')
print('CrysX-PyFock')
print('NAO: ', basis.bfs_nao)
start = timer()
dV = Integrals.nuc_mat_grad_symm(basis, mol)
print(dV)
duration = timer() - start
print('Matrix dimensions: ', dV.shape)
print('Duration for dV using PyFock: ', duration)

if bench_GPU:
    print('\n\n\n')
    print('CrysX-PyFock (GPU)')
    print('NAO: ', basis.bfs_nao)
    start = timer()
    dV_gpu = Integrals.nuc_mat_symm_cupy(basis, mol)
    print(dV_gpu)
    duration = timer() - start
    print('Matrix dimensions: ', dV_gpu.shape)
    print('Duration for dV using PyFock (GPU): ', duration)
    import cupy as cp
    print('Difference b/w CPU and GPU version: ', abs(dV - cp.asnumpy(dV_gpu)).max())


def build_nuc_grad_pyscf(mol):
    """
    Build the full nuclear attraction gradient tensor: (natm, 3, nao, nao)

    dV[A, x, i, j] = d/dR_{A,x} <i| V_nuc |j>

    where V_nuc = sum_C -Z_C / |r - R_C|

    Three types of contributions when differentiating w.r.t. atom A:
      1) Bra contribution: basis function i sits on atom A
         -> derivative of the Gaussian centered on A
      2) Ket contribution: basis function j sits on atom A
         -> derivative of the Gaussian centered on A
      3) Nuclear center contribution: nucleus C = A
         -> derivative of the operator -Z_A / |r - R_A|

    For (1) and (2), we use int1e_ipnuc which differentiates the bra
    basis function w.r.t. its center, summing over ALL nuclei in the operator.
    This is already the full sum over C for the operator part.

    For (3), we use int1e_iprinv with rinv placed at nucleus A, which
    differentiates the 1/|r-R_C| operator w.r.t. R_C.

    Key PySCF conventions:
      - int1e_ipnuc[x,i,j] = (d/dR_i chi_i | V_nuc | chi_j)
        where d/dR_i means derivative w.r.t. the center of chi_i.
        This equals -d/dR_A <i|V|j> when i is on atom A (for the bra part).
        NOTE: This is NOT symmetric. It only differentiates the bra side.

      - int1e_iprinv[x,i,j] = (d/dR chi_i | 1/|r-R_C| | chi_j)
        with R_C set by with_rinv_at_nucleus.
        By translational invariance of the 3-center integral:
        d/dR_A + d/dR_B + d/dR_C of <A|1/|r-C||B> = 0
        So d/dR_C <i|1/|r-C||j> = -d/dR_A <i|1/|r-C||j> - d/dR_B <i|1/|r-C||j>
        The bra derivative is int1e_iprinv, so:
        d/dR_C = -(iprinv + iprinv^T)
        And the nuclear potential piece: d/dR_C <i|-Z_C/|r-C||j> = Z_C * (iprinv + iprinv^T)
    """
    nao = mol.nao
    natm = mol.natm
    dV_full = np.zeros((natm, 3, nao, nao))

    # ---------------------------------------------------------------
    # Contributions 1 & 2: derivative w.r.t. AO basis centers
    # ---------------------------------------------------------------
    # int1e_ipnuc (NOT symmetric version - we need the raw non-symmetric integral)
    # h1[x, i, j] = <d_x chi_i | V_nuc | chi_j>
    # The actual derivative d/dR_A <i|V|j> (bra part) = -h1[:, p0:p1, :]
    # because moving the center is opposite to moving the coordinate.
    #
    # Use 'int1e_ipnuc' (not 'int1e_ipnuc_sph' or symmetric version)
    h1 = mol.intor('int1e_ipnuc', comp=3)  # (3, nao, nao) - NOT symmetric

    for iatom in range(natm):
        _, _, p0, p1 = mol.aoslice_by_atom()[iatom]
        # Bra on atom A: d/dR_A <i|V|j> = -h1[:, p0:p1, :]
        dV_full[iatom, :, p0:p1, :] -= h1[:, p0:p1, :]
        # Ket on atom A: d/dR_A <i|V|j> = -h1[:, :, p0:p1]^T
        # By the same logic: <i|V|d_x chi_j> for j on atom A
        # = h1^T contribution, with the same sign convention
        dV_full[iatom, :, :, p0:p1] -= h1[:, :, p0:p1].transpose(0, 2, 1).transpose(0, 2, 1)
        # Actually, we need to think about this more carefully.
        # <i | V | d_x chi_j> = <d_x chi_j | V | i>* = <d_x chi_j | V | i> (real)
        # = h1[:, p0:p1, :] with j in bra... no, h1 has bra=i, ket=j.
        # The ket derivative: we need (i | V | d_x j) which is h1 with roles swapped.
        # For real basis: <i|V|d_x j> = <d_x j|V|i> = h1[x, j_idx, i] 
        # where j_idx runs over functions on atom A.
        # So ket contribution to dV[A, x, i, j'] = -h1[x, j', i] = -h1^T[x, i, j']

    # Redo this properly:
    dV_full = np.zeros((natm, 3, nao, nao))
    
    for iatom in range(natm):
        _, _, p0, p1 = mol.aoslice_by_atom()[iatom]
        # Bra derivative: d/dR_A for basis i on atom A
        # <d/dR_A chi_i | V | chi_j> = h1[:, p0:p1, :]
        # d/dR_A <i|V|j> = -h1[:, p0:p1, :] (negative because d/dR_A of Gaussian
        # centered at R_A picks up minus sign relative to coordinate derivative)
        dV_full[iatom, :, p0:p1, :] += -h1[:, p0:p1, :]
        
        # Ket derivative: d/dR_A for basis j on atom A  
        # <chi_i | V | d/dR_A chi_j> = <d/dR_A chi_j | V | chi_i> (real, V hermitian)
        #                             = h1[:, p0:p1, :] but with second index being i
        # So: <i | V | d/dR_A j> = h1[x, j, i] for j in [p0,p1)
        # In matrix form for all i: it's h1[:, p0:p1, :]^T placed at [:, :, p0:p1]
        dV_full[iatom, :, :, p0:p1] += -h1[:, p0:p1, :].transpose(0, 2, 1)

    # ---------------------------------------------------------------
    # Contribution 3: derivative w.r.t. nuclear position R_C
    # ---------------------------------------------------------------
    # For each nucleus C (= atom A), we need:
    #   d/dR_C <i| -Z_C/|r-R_C| |j>
    #
    # Using translational invariance of the integral <i|1/|r-C||j>:
    #   d/dR_A + d/dR_B + d/dR_C = 0  (for each primitive triple A,B,C)
    #   => d/dR_C = -(d/dR_A + d/dR_B)
    #
    # int1e_iprinv with rinv at C gives:
    #   iprinv[x,i,j] = <d_x chi_i | 1/|r-R_C| | chi_j>
    #
    # d/dR_A <i|1/|r-C||j> = -iprinv[x,i,j]  (same sign logic as ipnuc)
    # d/dR_B <i|1/|r-C||j> = -iprinv[x,j,i] = -iprinv^T[x,i,j]
    #
    # d/dR_C <i|1/|r-C||j> = -(d/dR_A + d/dR_B) = iprinv + iprinv^T
    #
    # For V_C = -Z_C/|r-R_C|:
    #   d/dR_C <i|V_C|j> = -Z_C * (iprinv + iprinv^T)
    #
    # Wait, let's be more careful with signs.
    # <i| -Z_C/|r-C| |j> = -Z_C * <i|1/|r-C||j>
    # d/dR_C [-Z_C <i|1/|r-C||j>] = -Z_C * d/dR_C <i|1/|r-C||j>
    #                               = -Z_C * [iprinv + iprinv^T]
    
    for iatom in range(natm):
        with mol.with_rinv_at_nucleus(iatom):
            iprinv = mol.intor('int1e_iprinv', comp=3)  # (3, nao, nao)
        Zc = mol.atom_charge(iatom)
        # d/dR_C <i| -Zc/|r-Rc| |j> = -Zc * (iprinv + iprinv^T)
        dV_full[iatom] += -Zc * (iprinv + iprinv.transpose(0, 2, 1))

    return dV_full


# Comparison with PySCF
molPySCF = gto.Mole()
molPySCF.atom = xyzFilename
molPySCF.basis = basis_set_name
molPySCF.cart = True
molPySCF.build()

start = timer()
dV_pyscf_full = build_nuc_grad_pyscf(molPySCF)
duration = timer() - start

print('\n\nPySCF')
print('Duration for dV (full) using PySCF: ', duration)
print(dV_pyscf_full)
print('Matrix dimensions (full dV): ', dV_pyscf_full.shape)

# ---------------------------------------------------------------
# Verify PySCF reference passes translational invariance FIRST
# ---------------------------------------------------------------
# print('\n--- Sanity checks on PySCF reference ---')
# pyscf_trans_inv = abs(dV_pyscf_full.sum(axis=0)).max()
# print(f'PySCF translational invariance: max |sum_A dV[A]| = {pyscf_trans_inv:.6e}')
# if pyscf_trans_inv > 1e-8:
#     print('WARNING: PySCF reference FAILS translational invariance!')
#     print('The reference construction is likely incorrect.')
#     print('Trying alternative sign conventions...\n')
    
#     # Try all sign combinations for the three contributions
#     h1 = molPySCF.intor('int1e_ipnuc', comp=3)
    
#     best_trans_inv = 1e99
#     best_signs = None
#     best_dV = None
    
#     for s_bra in [+1, -1]:
#         for s_ket in [+1, -1]:
#             for s_nuc in [+1, -1]:
#                 dV_test = np.zeros_like(dV_pyscf_full)
                
#                 for iatom in range(molPySCF.natm):
#                     _, _, p0, p1 = molPySCF.aoslice_by_atom()[iatom]
#                     dV_test[iatom, :, p0:p1, :] += s_bra * h1[:, p0:p1, :]
#                     dV_test[iatom, :, :, p0:p1] += s_ket * h1[:, p0:p1, :].transpose(0, 2, 1)
                
#                 for iatom in range(molPySCF.natm):
#                     with molPySCF.with_rinv_at_nucleus(iatom):
#                         iprinv = molPySCF.intor('int1e_iprinv', comp=3)
#                     Zc = molPySCF.atom_charge(iatom)
#                     dV_test[iatom] += s_nuc * Zc * (iprinv + iprinv.transpose(0, 2, 1))
                
#                 ti = abs(dV_test.sum(axis=0)).max()
#                 symm = max(abs(dV_test[a, d] - dV_test[a, d].T).max() 
#                           for a in range(molPySCF.natm) for d in range(3))
#                 diff_pyfock = abs(dV_test - dV).max()
                
#                 if ti < best_trans_inv:
#                     best_trans_inv = ti
#                     best_signs = (s_bra, s_ket, s_nuc)
#                     best_dV = dV_test.copy()
                
#                 print(f'  signs=({s_bra:+d},{s_ket:+d},{s_nuc:+d}): '
#                       f'trans_inv={ti:.4e}, symm={symm:.4e}, diff_pyfock={diff_pyfock:.4e}')
    
#     print(f'\nBest signs: {best_signs} with trans_inv={best_trans_inv:.6e}')
#     dV_pyscf_full = best_dV

print('\n--- Final comparison ---')
print('Difference b/w PyFock (CPU) and PySCF: ', abs(dV_pyscf_full - dV).max())

# # Detailed per-atom, per-direction comparison
# print('\nPer-atom, per-direction max differences:')
# for iatom in range(molPySCF.natm):
#     for idir, dirstr in enumerate(['x', 'y', 'z']):
#         diff = abs(dV_pyscf_full[iatom, idir] - dV[iatom, idir]).max()
#         print(f'  Atom {iatom}, dir {dirstr}: max diff = {diff:.6e}')

# # Symmetry check on PyFock result
# print('\nSymmetry check on PyFock result (dV[A,x,i,j] vs dV[A,x,j,i]):')
# for iatom in range(mol.natoms):
#     for idir, dirstr in enumerate(['x', 'y', 'z']):
#         mat = dV[iatom, idir]
#         antisymm = abs(mat - mat.T).max()
#         print(f'  Atom {iatom}, dir {dirstr}: max |dV - dV^T| = {antisymm:.6e}')

# # Symmetry check on PySCF result
# print('\nSymmetry check on PySCF result (dV[A,x,i,j] vs dV[A,x,j,i]):')
# for iatom in range(molPySCF.natm):
#     for idir, dirstr in enumerate(['x', 'y', 'z']):
#         mat = dV_pyscf_full[iatom, idir]
#         antisymm = abs(mat - mat.T).max()
#         print(f'  Atom {iatom}, dir {dirstr}: max |dV - dV^T| = {antisymm:.6e}')

# # Translational invariance check
# print('\nTranslational invariance check (sum over atoms should be ~0):')
# dV_sum = dV.sum(axis=0)
# dV_pyscf_sum = dV_pyscf_full.sum(axis=0)
# print(f'  PyFock:  max |sum_A dV[A]| = {abs(dV_sum).max():.6e}')
# print(f'  PySCF:   max |sum_A dV[A]| = {abs(dV_pyscf_sum).max():.6e}')

# # Additional: compare undifferentiated nuclear matrix
# print('\n--- Undifferentiated nuclear matrix comparison ---')
# from pyfock import Integrals as IntMod
# V_pyfock = IntMod.nuc_mat_symm(basis, mol)
# V_pyscf = molPySCF.intor_symmetric('int1e_nuc')
# print(f'Undifferentiated V diff: {abs(V_pyfock - V_pyscf).max():.6e}')

# # Numerical gradient check
# print('\n--- Numerical gradient check (finite difference) ---')
# delta = 1e-5  # Bohr
# dV_numerical = np.zeros_like(dV)
# V_ref = IntMod.nuc_mat_symm(basis, mol)

# for iatom in range(mol.natoms):
#     for idir in range(3):
#         # Forward
#         mol_fwd = Mol(coordfile=xyzFilename)
#         coords_fwd = mol_fwd.coordsBohrs.copy()
#         coords_fwd[iatom, idir] += delta
#         mol_fwd.coordsBohrs = coords_fwd
#         # Need to also update basis for shifted geometry?
#         # Actually for nuclear potential gradient, we need to shift:
#         # 1) basis center if basis is on this atom
#         # 2) nuclear position
#         # This is complex - let's just use PySCF for numerical check
        
#         molp = molPySCF.copy()
#         coords_bohr = molPySCF.atom_coords().copy()  # in Bohr
#         coords_bohr[iatom, idir] += delta
#         # Rebuild with new coords
#         atom_list = []
#         for ia in range(molPySCF.natm):
#             sym = molPySCF.atom_symbol(ia)
#             atom_list.append([sym, coords_bohr[ia].tolist()])
#         molp = gto.Mole()
#         molp.atom = atom_list
#         molp.basis = basis_set_name
#         molp.cart = True
#         molp.unit = 'Bohr'
#         molp.build()
#         Vp = molp.intor_symmetric('int1e_nuc')
        
#         # Backward
#         coords_bohr2 = molPySCF.atom_coords().copy()
#         coords_bohr2[iatom, idir] -= delta
#         atom_list2 = []
#         for ia in range(molPySCF.natm):
#             sym = molPySCF.atom_symbol(ia)
#             atom_list2.append([sym, coords_bohr2[ia].tolist()])
#         molm = gto.Mole()
#         molm.atom = atom_list2
#         molm.basis = basis_set_name
#         molm.cart = True
#         molm.unit = 'Bohr'
#         molm.build()
#         Vm = molm.intor_symmetric('int1e_nuc')
        
#         dV_numerical[iatom, idir] = (Vp - Vm) / (2.0 * delta)

# print('Numerical gradient computed.')
# print(f'PyFock  vs numerical: max diff = {abs(dV - dV_numerical).max():.6e}')
# print(f'PySCF   vs numerical: max diff = {abs(dV_pyscf_full - dV_numerical).max():.6e}')

# print('\nPer-atom, per-direction: PyFock vs numerical:')
# for iatom in range(mol.natoms):
#     for idir, dirstr in enumerate(['x', 'y', 'z']):
#         diff = abs(dV[iatom, idir] - dV_numerical[iatom, idir]).max()
#         print(f'  Atom {iatom}, dir {dirstr}: max diff = {diff:.6e}')

# print('\nPer-atom, per-direction: PySCF vs numerical:')
# for iatom in range(molPySCF.natm):
#     for idir, dirstr in enumerate(['x', 'y', 'z']):
#         diff = abs(dV_pyscf_full[iatom, idir] - dV_numerical[iatom, idir]).max()
#         print(f'  Atom {iatom}, dir {dirstr}: max diff = {diff:.6e}')

if bench_GPU:
    print('Difference b/w PyFock (GPU) and PySCF: ', abs(dV_pyscf_full - cp.asnumpy(dV_gpu)).max())