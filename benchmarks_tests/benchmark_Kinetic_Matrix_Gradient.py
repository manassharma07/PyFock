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


#OVERLAP MATRIX BENCHMARK and comparison with PySCF
#Benchmarking and performance assessment and comparison using various techniques and different softwares

# basis_set_name = 'sto-2g'
# basis_set_name = 'sto-3g'
# basis_set_name = 'sto-6g'
# basis_set_name = 'def2-SVP'
# basis_set_name = 'def2-DZVP'
basis_set_name = 'def2-TZVP'
# basis_set_name = 'def2-TZVPPD'
# basis_set_name = 'def2-QZVPPD'
# basis_set_name = 'ano-rcc'

# xyzFilename = 'Benzene-Fulvene_Dimer.xyz'
# xyzFilename = 'H2.xyz'
# xyzFilename = 'H2O.xyz'
# xyzFilename = 'Ethane.xyz'
# xyzFilename = 'Cholesterol.xyz'
# xyzFilename = 'Serotonin.xyz'
# xyzFilename = 'Decane_C10H22.xyz'
# xyzFilename = 'Icosane_C20H42.xyz'
xyzFilename = 'Tetracontane_C40H82.xyz'
# xyzFilename = 'Pentacontane_C50H102.xyz'
# xyzFilename = 'Octacontane_C80H162.xyz'
# xyzFilename = 'Hectane_C100H202.xyz'
# xyzFilename = 'Icosahectane_C120H242.xyz'

#First of all we need a mol object with some geometry
mol = Mol(coordfile = xyzFilename)

# Next we need to specify some basis
# The basis set can then be used to calculate things like Overlap, KE, integrals/matrices.
basis = Basis(mol, {'all':Basis.load(mol=mol, basis_name=basis_set_name)})


print('\n\n\n')
print('CrysX-PyFock')
print('NAO: ', basis.bfs_nao)
#NOTE: The matrices are calculated in CAO basis and not the SAO basis
#You should refer to the example that shows the transformation between the two if you need matrices in SAO basis.
start=timer()
dT = Integrals.kin_mat_grad_symm(basis)
print(dT) 
duration = timer() - start
print('Matrix dimensions: ', dT.shape)
print('Duration for dT using PyFock: ',duration)

if bench_GPU:
    print('\n\n\n')
    print('CrsX-PyFock (GPU)')
    print('NAO: ', basis.bfs_nao)
    #NOTE: The matrices are calculated in CAO basis and not the SAO basis
    #You should refer to the example that shows the transformation between the two if you need matrices in SAO basis.
    start=timer()
    dT_gpu = Integrals.overlap_mat_symm_cupy(basis)
    print(dT_gpu) 
    duration = timer() - start
    print('Matrix dimensions: ', dT_gpu.shape)
    print('Duration for dS using PyFock (GPU): ',duration)
    import cupy as cp
    print('Difference b/w CPU and GPU version: ', abs(dT - cp.asnumpy(dT_gpu)).max())


#Comparison with PySCF
molPySCF = gto.Mole()
molPySCF.atom = xyzFilename
molPySCF.basis = basis_set_name
molPySCF.cart = True
molPySCF.build()
#print(molPySCF.cart_labels())


#Overlap mat
start=timer()
dT_pyscf = -molPySCF.intor_symmetric('int1e_ipkin', comp=3)
dT_pyscf = dT_pyscf #+ dS_pyscf.transpose(0,2,1)
duration = timer() - start
print('\n\nPySCF')
# print(dT_pyscf)
print('Matrix dimensions (partial dT): ', dT_pyscf.shape)
print('Duration for dT (partial) using PySCF: ', duration)

# https://github.com/pyscf/pyscf/issues/1067
# https://github.com/jcandane/HF_Gradient/blob/main/PySCF_Gradients.ipynb
def T_deriv(atom_id, T_xAB, mol):
    shl0, shl1, p0, p1 = mol.aoslice_by_atom()[atom_id]

    vrinv = np.zeros(T_xAB.shape)
    vrinv[:, p0:p1, :] += T_xAB[:, p0:p1, :]
    
    final = vrinv + vrinv.transpose(0, 2, 1)
    # Set derivatives to zero for basis functions on the same atom using numpy indexing
    final[:, p0:p1, p0:p1] = 0.0
    if atom_id!=0:
        for i in range(0, atom_id):
            shl0, shl1, p0, p1 = mol.aoslice_by_atom()[i]
            final[:, p0:p1, :] = -final[:, p0:p1, :]
            final[:, :, p0:p1] = -final[:, :, p0:p1]
    return final
dT_pyscf_full = np.zeros(((len(molPySCF.aoslice_by_atom()),) + dT_pyscf.shape))
for iatom in range(len(molPySCF.aoslice_by_atom())):
    dT_pyscf_full[iatom] = T_deriv(iatom, dT_pyscf, molPySCF)
    
duration = timer() - start
print('Duration for dS (full) using PySCF: ',duration)
print(dT_pyscf_full)
print('Matrix dimensions (full dS): ', dT_pyscf_full.shape)
print('Difference b/w PyFock (CPU) and PySCF: ', abs(dT_pyscf_full - dT).max())  #There will sometimes be a difference b/w PySCF and CrysX values because PySCF doesn't normalize d,f,g orbitals.
if bench_GPU:
    print('Difference b/w PyFock (GPU) and PySCF: ',abs(dT_pyscf_full - cp.asnumpy(dT_gpu)).max())  #There will sometimes be a difference b/w PySCF and CrysX values because PySCF doesn't normalize d,f,g orbitals.
    