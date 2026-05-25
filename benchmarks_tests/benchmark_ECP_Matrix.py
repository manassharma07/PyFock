from pyfock import Basis
from pyfock import Mol
from pyfock import Integrals
from timeit import default_timer as timer
import numpy as np
import os

from pyscf import gto
import numba

ncores = 4

numba.set_num_threads(ncores)
os.environ['OMP_NUM_THREADS'] = str(ncores)
os.environ["OPENBLAS_NUM_THREADS"] = str(ncores) # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = str(ncores) # export MKL_NUM_THREADS=4
os.environ["VECLIB_MAXIMUM_THREADS"] = str(ncores) # export VECLIB_MAXIMUM_THREADS=4
# os.environ["NUMEXPR_NUM_THREADS"] = str(ncores) # export NUMEXPR_NUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = str(ncores) # export NUMEXPR_NUM_THREADS=1

#IMPORTANT
#Since, it seems that in order for the Numba implementation to work efficiently we must remove the compile time from the calculation.
#So, a very simple calculation on a very small system must be run before, to have the numba functions compiled.
mol_temp = Mol(atoms=[['Cd', 0.0, 0.0, 0.0]])
basis_temp = Basis(mol_temp, {'all':Basis.load(mol=mol_temp, basis_name='def2-SVP')})
Vecp_temp = Integrals.ecp_mat_symm(basis_temp)
Vecp_temp = Integrals.ecp_mat_symm_test(basis_temp, n_radial=32, n_theta=8, n_phi=16)

#ECP MATRIX BENCHMARK and comparison with PySCF
#Benchmarking and performance assessment and comparison using various techniques and different softwares

# basis_set_name = 'def2-SVP'
basis_set_name = 'def2-TZVP'
# basis_set_name = 'def2-QZVP'

### def2 ECP benchmark systems
# xyzFilename = 'AgCl.xyz'
# xyzFilename = 'AuCl.xyz'
# xyzFilename = 'BiH3.xyz'
# xyzFilename = 'I2.xyz'
# xyzFilename = 'PbH4.xyz'
# xyzFilename = 'RbCl.xyz'
# xyzFilename = 'SnCl4.xyz'
# xyzFilename = 'W_CO6.xyz'
# xyzFilename = 'XeF2.xyz'
xyzFilename = 'Cd_dimer.xyz'

#First of all we need a mol object with some geometry
mol = Mol(coordfile = xyzFilename)

# Next we need to specify some basis
# The basis set can then be used to calculate things like Overlap, KE, integrals/matrices.
basis = Basis(mol, {'all':Basis.load(mol=mol, basis_name=basis_set_name)})


print('\n\n\n')
print('CrysX-PyFock analytical ECP')
print('NAO: ', basis.bfs_nao)
print('Number of ECP centers: ', len(basis.ecps))
#NOTE: The matrices are calculated in CAO basis and not the SAO basis
#You should refer to the example that shows the transformation between the two if you need matrices in SAO basis.
start=timer()
Vecp_analytical = Integrals.ecp_mat_symm(basis)
print(Vecp_analytical)
duration = timer() - start
print('Matrix dimensions: ', Vecp_analytical.shape)
print('Duration for Vecp analytical using PyFock: ', duration)


print('\n\n\n')
print('CrysX-PyFock numerical ECP')
print('NAO: ', basis.bfs_nao)
#NOTE: The matrices are calculated in CAO basis and not the SAO basis
#You should refer to the example that shows the transformation between the two if you need matrices in SAO basis.
start=timer()
Vecp_numerical = Integrals.ecp_mat_symm_test(basis, n_radial=128, n_theta=18, n_phi=36)
print(Vecp_numerical)
duration = timer() - start
print('Matrix dimensions: ', Vecp_numerical.shape)
print('Duration for Vecp numerical using PyFock: ', duration)
print('Difference b/w PyFock analytical and numerical ECP: ', abs(Vecp_analytical - Vecp_numerical).max())


#Comparison with PySCF
molPySCF = gto.Mole()
molPySCF.atom = xyzFilename
molPySCF.basis = basis_set_name
molPySCF.ecp = basis_set_name
molPySCF.cart = True
molPySCF.build()
#print(molPySCF.cart_labels())


#ECP mat
start=timer()
Vecp_pyscf = molPySCF.intor_symmetric('ECPscalar_cart')
duration = timer() - start
print('\n\nPySCF')
print(Vecp_pyscf)
print('Matrix dimensions: ', Vecp_pyscf.shape)
print('Duration for Vecp using PySCF: ', duration)
print('Difference b/w PyFock analytical and PySCF: ', abs(Vecp_pyscf - Vecp_analytical).max())  #There will sometimes be a difference b/w PySCF and CrysX values because PySCF doesn't normalize d,f,g orbitals.
print('Difference b/w PyFock numerical and PySCF: ', abs(Vecp_pyscf - Vecp_numerical).max())  #There will sometimes be a difference b/w PySCF and CrysX values because PySCF doesn't normalize d,f,g orbitals.
