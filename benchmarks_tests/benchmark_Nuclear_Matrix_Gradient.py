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

# basis_set_name = 'sto-2g'
# basis_set_name = 'sto-3g'
basis_set_name = 'sto-6g'
# basis_set_name = 'def2-SVP'
# basis_set_name = 'def2-DZVP'
# basis_set_name = 'def2-TZVP'
# basis_set_name = 'def2-TZVPPD'
# basis_set_name = 'def2-QZVPPD'
# basis_set_name = 'ano-rcc'

# xyzFilename = 'Benzene-Fulvene_Dimer.xyz'
# xyzFilename = 'H2.xyz'
xyzFilename = 'H2O.xyz'
# xyzFilename = 'Ethane.xyz'
# xyzFilename = 'Cholesterol.xyz'
# xyzFilename = 'Serotonin.xyz'
# xyzFilename = 'Decane_C10H22.xyz'
# xyzFilename = 'Icosane_C20H42.xyz'
# xyzFilename = 'Tetracontane_C40H82.xyz'
# xyzFilename = 'Pentacontane_C50H102.xyz'
# xyzFilename = 'Octacontane_C80H162.xyz'
# xyzFilename = 'Hectane_C100H202.xyz'
# xyzFilename = 'Icosahectane_C120H242.xyz'

mol = Mol(coordfile=xyzFilename)
basis = Basis(mol, {'all': Basis.load(mol=mol, basis_name=basis_set_name)})

print('\n\n\n')
print('CrysX-PyFock')
print('NAO: ', basis.bfs_nao)
start = timer()
dV = Integrals.nuc_mat_grad_symm(basis, mol)
# print(dV)
duration = timer() - start
print('Matrix dimensions: ', dV.shape)
print('Duration for dV using PyFock: ', duration)

start = timer()
dV_r = Integrals.nuc_mat_grad_r_symm(basis, mol)
duration = timer() - start
print('Matrix dimensions (dV/dr): ', dV_r.shape)
print('Duration for dV/dr using PyFock: ', duration)

start = timer()
dV_from_r = Integrals.nuc_mat_grad_r_symm(basis, mol, wrt_atoms=True)
duration = timer() - start
print('Matrix dimensions (dV/dRA from dV/dr + operator terms): ', dV_from_r.shape)
print('Duration for dV/dRA from dV/dr + operator terms using PyFock: ', duration)
print('Difference b/w direct dV/dRA and dV/dRA from r-gradient path: ', abs(dV - dV_from_r).max())

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

    h1 = mol.intor('int1e_ipnuc', comp=3)
    for iatom in range(natm):
        _, _, p0, p1 = mol.aoslice_by_atom()[iatom]
        dV_full[iatom, :, p0:p1, :] += -h1[:, p0:p1, :]
        dV_full[iatom, :, :, p0:p1] += -h1[:, p0:p1, :].transpose(0, 2, 1)
    
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
dV_pyscf_r = -molPySCF.intor('int1e_ipnuc', comp=3)
dV_pyscf_full = build_nuc_grad_pyscf(molPySCF)
duration = timer() - start

print('\n\nPySCF')
print('Matrix dimensions (dV/dr): ', dV_pyscf_r.shape)
print('Difference b/w PyFock dV/dr and PySCF dV/dr: ', abs(dV_pyscf_r - dV_r).max())
print('Duration for dV (full) using PySCF: ', duration)
# print(dV_pyscf_full)
print('Matrix dimensions (full dV): ', dV_pyscf_full.shape)

print('\n--- Final comparison ---')
print('Difference b/w PyFock (CPU) and PySCF: ', abs(dV_pyscf_full - dV).max())

if bench_GPU:
    print('Difference b/w PyFock (GPU) and PySCF: ', abs(dV_pyscf_full - cp.asnumpy(dV_gpu)).max())
