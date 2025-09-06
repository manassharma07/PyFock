from pyfock import Basis
from pyfock import Mol
from pyfock import Integrals
from timeit import default_timer as timer
import numpy as np
import os

from pyscf import gto, dft
import numba 

ncores = 4
bench_GPU = True
numba.set_num_threads(ncores)
os.environ['OMP_NUM_THREADS'] = str(ncores)
os.environ["OPENBLAS_NUM_THREADS"] = str(ncores) # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = str(ncores) # export MKL_NUM_THREADS=4
os.environ["VECLIB_MAXIMUM_THREADS"] = str(ncores) # export VECLIB_MAXIMUM_THREADS=4
# os.environ["NUMEXPR_NUM_THREADS"] = str(ncores) # export NUMEXPR_NUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = str(ncores) # export NUMEXPR_NUM_THREADS=1


#IMPORTANT
start=timer()
#Since, it seems that in order for the Numba implementation to work efficiently we must remove the compile time from the calculation.
#So, a very simple calculation on a very small system must be run before, to have the numba functions compiled.
mol_temp = Mol(coordfile='H2O.xyz')
basis_temp = Basis(mol_temp, {'all':Basis.load(mol=mol_temp, basis_name='sto-2g')})
Vmat_temp = Integrals.dipole_moment_mat_symm(basis_temp)
duration = timer() - start
print('Compilation duration: ',duration)


#DIPOLE MOMENT MATRIX BENCHMARK and comparison with PySCF
#Benchmarking and performance assessment and comparison using various techniques and different softwares

# basis_set_name = 'sto-2g'
# basis_set_name = 'sto-3g'
# basis_set_name = 'sto-6g'
basis_set_name = 'def2-SVP'
# basis_set_name = 'def2-DZVP'
# basis_set_name = 'def2-TZVP'
# basis_set_name = 'def2-TZVPPD'
# basis_set_name = 'def2-QZVP'
# basis_set_name = 'def2-QZVPPD'
# basis_set_name = 'ano-rcc'

# xyzFilename = 'Benzene-Fulvene_Dimer.xyz'
# xyzFilename = 'Adenine-Thymine.xyz'
# xyzFilename = 'Zn.xyz'
# xyzFilename = 'Zn_dimer.xyz'
# xyzFilename = 'TPP.xyz'
# xyzFilename = 'Zn_TPP.xyz'
# xyzFilename = 'H2O.xyz'

# xyzFilename = 'Caffeine.xyz'
# xyzFilename = 'Serotonin.xyz'
# xyzFilename = 'Cholesterol.xyz'
# xyzFilename = 'C60.xyz'
# xyzFilename = 'Taxol.xyz'
# xyzFilename = 'Valinomycin.xyz'
# xyzFilename = 'Olestra.xyz'
# xyzFilename = 'Ubiquitin.xyz'

### 1D Carbon Alkanes
# xyzFilename = 'Ethane.xyz'
# xyzFilename = 'Decane_C10H22.xyz'
# xyzFilename = 'Icosane_C20H42.xyz'
xyzFilename = 'Tetracontane_C40H82.xyz'
# xyzFilename = 'Pentacontane_C50H102.xyz'
# xyzFilename = 'Octacontane_C80H162.xyz'
# xyzFilename = 'Hectane_C100H202.xyz'
# xyzFilename = 'Icosahectane_C120H242.xyz'

### 2D Carbon
# xyzFilename = 'Graphene_C16.xyz'
# xyzFilename = 'Graphene_C76.xyz'
# xyzFilename = 'Graphene_C102.xyz'
# xyzFilename = 'Graphene_C184.xyz'
# xyzFilename = 'Graphene_C210.xyz'
# xyzFilename = 'Graphene_C294.xyz'

### 3d Carbon Fullerenes
# xyzFilename = 'C60.xyz'
# xyzFilename = 'C70.xyz'
# xyzFilename = 'Graphene_C102.xyz'
# xyzFilename = 'Graphene_C184.xyz'
# xyzFilename = 'Graphene_C210.xyz'
# xyzFilename = 'Graphene_C294.xyz'

#First of all we need a mol object with some geometry
mol = Mol(coordfile = xyzFilename)
# origin = mol.get_center_of_charge()
origin = np.zeros((3))
print('Gauge origin: ', origin)

# Next we need to specify some basis
# The basis set can then be used to calculate things like Overlap, KE, integrals/matrices.
basis = Basis(mol, {'all':Basis.load(mol=mol, basis_name=basis_set_name)})
#basis = Basis(mol, {'all':Basis.load(mol=mol, basis_name='def2-svp')})


print('\n\n\n')
print('CrysX-PyFock MMD')
print('NAO: ', basis.bfs_nao)
#NOTE: The matrices are calculated in CAO basis and not the SAO basis
#You should refer to the example that shows the transformation between the two if you need matrices in SAO basis.
start=timer()
M_mmd = Integrals.dipole_moment_mat_symm(basis, origin=origin)
print(M_mmd[0]) 
duration = timer() - start
print('Matrix dimensions: ', M_mmd.shape)
print('Duration for M using PyFock (MMD): ',duration)

if bench_GPU:
    print('\n\n\n')
    print('CrsX-PyFock MMD (GPU)')
    print('NAO: ', basis.bfs_nao)
    #NOTE: The matrices are calculated in CAO basis and not the SAO basis
    #You should refer to the example that shows the transformation between the two if you need matrices in SAO basis.
    start=timer()
    M_mmd_gpu = Integrals.dipole_moment_mat_symm_cupy(basis)
    print(M_mmd_gpu[0]) 
    duration = timer() - start
    print('Matrix dimensions: ', M_mmd_gpu.shape)
    print('Duration for M using PyFock (GPU): ',duration)
    import cupy as cp
    print('Difference b/w CPU and GPU version: ', abs(M_mmd - cp.asnumpy(M_mmd_gpu)).max())


#Comparison with PySCF
molPySCF = gto.Mole()
molPySCF.atom = xyzFilename
molPySCF.basis = basis_set_name
molPySCF.cart = True
molPySCF.build()
#print(molPySCF.cart_labels())


# Dipole moment mat
start=timer()
with molPySCF.with_common_orig(origin):
    M = molPySCF.intor_symmetric('int1e_r', comp=3)
duration = timer() - start
print('\n\nPySCF')
print(M[0])
print('Matrix dimensions: ', M.shape)
print('Duration for M using PySCF: ',duration)
print('Difference b/w PyFock (CPU) and PySCF: ', abs(M - M_mmd).max())  #There will sometimes be a difference b/w PySCF and CrysX values because PySCF doesn't normalize d,f,g orbitals.
if bench_GPU:
    print('Difference b/w PyFock (GPU) and PySCF: ',abs(M - cp.asnumpy(M_mmd_gpu)).max())  #There will sometimes be a difference b/w PySCF and CrysX values because PySCF doesn't normalize d,f,g orbitals.
    

# Calculate dipole moment
mf = dft.rks.RKS(molPySCF)
mf.init_guess = 'minao'
dmat = mf.init_guess_by_minao(molPySCF)

print('Dipole moment PySCF')
print(mf.dip_moment(molPySCF, dmat, unit='AU')) 
el_dip = np.einsum('xij,ji->x', M, dmat).real
charges = molPySCF.atom_charges()
coords  = molPySCF.atom_coords() 
nucl_dip = np.einsum('i,ix->x', charges, coords)
mol_dip = nucl_dip - el_dip
print('Dipole moment(X, Y, Z, A.U.):', *mol_dip)

print('Dipole moment CrysX-PyFock')
# el_dip = np.einsum('xij,ji->x', M_mmd, dmat).real
# charges = mol.Zcharges
# coords  = mol.coordsBohrs 
# nucl_dip = np.einsum('i,ix->x', charges, coords)
# mol_dip = nucl_dip - el_dip
mol_dip = mol.get_dipole_moment(M_mmd, dmat)
print('Dipole moment(X, Y, Z, A.U.):', *mol_dip)