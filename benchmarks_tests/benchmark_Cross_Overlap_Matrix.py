from pyfock import Basis
from pyfock import Mol
from pyfock import Integrals
from timeit import default_timer as timer
import numpy as np
import os

from pyscf import gto, dft
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
mol_temp = Mol(coordfile='H2O.xyz')
basis_temp = Basis(mol_temp, {'all':Basis.load(mol=mol_temp, basis_name='sto-2g')})
Vmat_temp = Integrals.overlap_mat_symm(basis_temp)

#OVERLAP MATRIX BENCHMARK and comparison with PySCF
#Benchmarking and performance assessment and comparison using various techniques and different softwares

# basis_set_nameA = 'sto-2g'
basis_set_nameA = 'sto-3g'
# basis_set_nameA = 'sto-6g'
# basis_set_nameA = 'def2-SVP'
# basis_set_nameA = 'def2-DZVP'
# basis_set_nameA = 'def2-TZVP'
# basis_set_nameA = 'def2-TZVPPD'
# basis_set_nameA = 'def2-QZVPPD'
# basis_set_nameA = 'ano-rcc'

# basis_set_nameB = 'sto-2g'
basis_set_nameB = 'sto-3g'
# basis_set_nameB = 'sto-6g'
# basis_set_nameB = 'def2-SVP'
# basis_set_nameB = 'def2-DZVP'
# basis_set_nameB = 'def2-TZVP'
# basis_set_nameB = 'def2-TZVPPD'
# basis_set_nameB = 'def2-QZVPPD'
# basis_set_nameB = 'ano-rcc'

# xyzFilenameA = 'Benzene-Fulvene_Dimer.xyz'
# xyzFilenameA = 'H2O.xyz'
# xyzFilenameA = 'Ethane.xyz'
# xyzFilenameA = 'Cholesterol.xyz'
# xyzFilenameA = 'Serotonin.xyz'
xyzFilenameA = 'Decane_C10H22.xyz'
# xyzFilenameA = 'Icosane_C20H42.xyz'
# xyzFilenameA = 'Tetracontane_C40H82.xyz'
# xyzFilenameA = 'Pentacontane_C50H102.xyz'
# xyzFilenameA = 'Octacontane_C80H162.xyz'
# xyzFilenameA = 'Hectane_C100H202.xyz'
# xyzFilenameA = 'Icosahectane_C120H242.xyz'

# xyzFilenameB = 'Benzene-Fulvene_Dimer.xyz'
# xyzFilenameB = 'H2O.xyz'
# xyzFilenameB = 'Ethane.xyz'
# xyzFilenameB = 'Cholesterol.xyz'
# xyzFilenameB = 'Serotonin.xyz'
# xyzFilenameB = 'Decane_C10H22.xyz'
# xyzFilenameB = 'Icosane_C20H42.xyz'
# xyzFilenameB = 'Tetracontane_C40H82.xyz'
# xyzFilenameB = 'Pentacontane_C50H102.xyz'
# xyzFilenameB = 'Octacontane_C80H162.xyz'
# xyzFilenameB = 'Hectane_C100H202.xyz'
xyzFilenameB = 'Icosahectane_C120H242.xyz'

#First of all we need a mol object with some geometry
molA = Mol(coordfile = xyzFilenameA)
molB = Mol(coordfile = xyzFilenameB)

# Next we need to specify some basis
# The basis set can then be used to calculate things like Overlap, KE, integrals/matrices.
basisA = Basis(molA, {'all':Basis.load(mol=molA, basis_name=basis_set_nameA)})
basisB = Basis(molB, {'all':Basis.load(mol=molB, basis_name=basis_set_nameB)})



print('\n\n\n')
print('CrysX-PyFock')
print('NAO (A): ', basisA.bfs_nao)
print('NAO (B): ', basisB.bfs_nao)
#NOTE: The matrices are calculated in CAO basis and not the SAO basis
#You should refer to the example that shows the transformation between the two if you need matrices in SAO basis.
start=timer()
S_AB = Integrals.cross_overlap_mat_symm(basisA, basisB)
print(S_AB) 
duration = timer() - start
print('Matrix dimensions: ', S_AB.shape)
print('Duration for S using PyFock: ',duration)



#Comparison with PySCF
molPySCFA = gto.Mole()
molPySCFA.atom = xyzFilenameA
molPySCFA.basis = basis_set_nameA
molPySCFA.cart = True
molPySCFA.build()

molPySCFB = gto.Mole()
molPySCFB.atom = xyzFilenameB
molPySCFB.basis = basis_set_nameB
molPySCFB.cart = True
molPySCFB.build()


#Nuclear mat
start=timer()
# V = molPySCF.intor('int1e_nuc')
S_AB_pyscf = gto.intor_cross('int1e_ovlp', molPySCFA, molPySCFB)
duration = timer() - start
print('\n\nPySCF')
print(S_AB_pyscf)
print('Matrix dimensions: ', S_AB_pyscf.shape)
print('Duration for S using PySCF: ',duration)
print('Difference b/w PyFock (CPU) and PySCF: ', abs(S_AB_pyscf - S_AB).max())  #There will sometimes be a difference b/w PySCF and CrysX values because PySCF doesn't normalize d,f,g orbitals.
