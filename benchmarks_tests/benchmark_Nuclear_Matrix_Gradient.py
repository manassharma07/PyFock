from pyfock import Basis
from pyfock import Mol
from pyfock import Integrals
from timeit import default_timer as timer
import numpy as np
import os

from pyscf import gto, dft
import numba

ncores = 4
bench_GPU = False

numba.set_num_threads(ncores)
os.environ['OMP_NUM_THREADS'] = str(ncores)
os.environ["OPENBLAS_NUM_THREADS"] = str(ncores) # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = str(ncores) # export MKL_NUM_THREADS=4
os.environ["VECLIB_MAXIMUM_THREADS"] = str(ncores) # export VECLIB_MAXIMUM_THREADS=4
# os.environ["NUMEXPR_NUM_THREADS"] = str(ncores) # export NUMEXPR_NUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = str(ncores) # export NUMEXPR_NUM_THREADS=1


# NUCLEAR MATRIX GRADIENT BENCHMARK and comparison with PySCF
# Benchmarking and performance assessment and comparison using various techniques and different softwares

# basis_set_name = 'sto-2g'
basis_set_name = 'sto-3g'
# basis_set_name = 'sto-6g'
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

# First of all we need a mol object with some geometry
mol = Mol(coordfile = xyzFilename)

# Next we need to specify some basis
# The basis set can then be used to calculate things like Overlap, KE, integrals/matrices.
basis = Basis(mol, {'all':Basis.load(mol=mol, basis_name=basis_set_name)})


print('\n\n\n')
print('CrysX-PyFock')
print('NAO: ', basis.bfs_nao)
# NOTE: The matrices are calculated in CAO basis and not the SAO basis
# You should refer to the example that shows the transformation between the two if you need matrices in SAO basis.
start=timer()
dV = Integrals.nuc_mat_grad_symm(basis, mol)
print(dV)
duration = timer() - start
print('Matrix dimensions: ', dV.shape)
print('Duration for dV using PyFock: ',duration)

if bench_GPU:
    print('\n\n\n')
    print('CrsX-PyFock (GPU)')
    print('NAO: ', basis.bfs_nao)
    start=timer()
    dV_gpu = Integrals.nuc_mat_symm_cupy(basis, mol)
    print(dV_gpu)
    duration = timer() - start
    print('Matrix dimensions: ', dV_gpu.shape)
    print('Duration for dV using PyFock (GPU): ',duration)
    import cupy as cp
    print('Difference b/w CPU and GPU version: ', abs(dV - cp.asnumpy(dV_gpu)).max())


# Comparison with PySCF
molPySCF = gto.Mole()
molPySCF.atom = xyzFilename
molPySCF.basis = basis_set_name
molPySCF.cart = True
molPySCF.build()
# print(molPySCF.cart_labels())


# Nuclear matrix derivative wrt AO centers only
start=timer()
dV_pyscf = -molPySCF.intor_symmetric('int1e_ipnuc', comp=3)
duration = timer() - start
print('\n\nPySCF')
print('Matrix dimensions (partial dV): ', dV_pyscf.shape)
print('Duration for dV (partial) using PySCF: ', duration)


# This mirrors the style used in the kinetic-matrix-gradient benchmark,
# but also adds the derivative of the nuclear operator with respect to
# the position of the selected nucleus through int1e_iprinv.
def Vnuc_deriv(atom_id, V_xAB, mol):
    shl0, shl1, p0, p1 = mol.aoslice_by_atom()[atom_id]

    with mol.with_rinv_at_nucleus(atom_id):
        vrinv = mol.intor('int1e_iprinv', comp=3)
    vrinv *= -mol.atom_charge(atom_id)

    vrinv[:, p0:p1, :] += V_xAB[:, p0:p1, :]

    final = vrinv + vrinv.transpose(0, 2, 1)
    final[:, p0:p1, p0:p1] = 0.0
    if atom_id!=0:
        for i in range(0, atom_id):
            shl0, shl1, p0, p1 = mol.aoslice_by_atom()[i]
            final[:, p0:p1, :] = -final[:, p0:p1, :]
            final[:, :, p0:p1] = -final[:, :, p0:p1]
    return final


dV_pyscf_full = np.zeros(((len(molPySCF.aoslice_by_atom()),) + dV_pyscf.shape))
for iatom in range(len(molPySCF.aoslice_by_atom())):
    dV_pyscf_full[iatom] = Vnuc_deriv(iatom, dV_pyscf, molPySCF)

duration = timer() - start
print('Duration for dV (full) using PySCF: ',duration)
print(dV_pyscf_full)
print('Matrix dimensions (full dV): ', dV_pyscf_full.shape)
print('Difference b/w PyFock (CPU) and PySCF: ', abs(dV_pyscf_full - dV).max())
if bench_GPU:
    print('Difference b/w PyFock (GPU) and PySCF: ',abs(dV_pyscf_full - cp.asnumpy(dV_gpu)).max())
