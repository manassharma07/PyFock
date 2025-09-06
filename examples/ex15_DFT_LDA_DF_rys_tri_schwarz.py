import os

ncores = 4
os.environ['OMP_NUM_THREADS'] = str(ncores)
os.environ["OPENBLAS_NUM_THREADS"] = str(ncores) # or export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = str(ncores) #  or export MKL_NUM_THREADS=4
os.environ["VECLIB_MAXIMUM_THREADS"] = str(ncores)  # or  export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = str(ncores) # or  export NUMEXPR_NUM_THREADS=4


# Run your tasks here
from pyfock import Basis
from pyfock import Mol
from pyfock import Integrals
from pyfock import DFT

from timeit import default_timer as timer
import numpy as np
import scipy



# DFT SCF example using density fitting for Coulomb term and rys quadrature for 3c2e and 2c2e integrals.
# Furthermore, Schwarz screening and other symmetries are used to reduce the number of integral evaluations as well as storage.
# Please note: Here we are using the DFT grids provided by the numgrid library that is installed automatically while
# installing PyFock.
# However, it was found during testing that PySCF grids were much more efficient as they resulted in overall less
# number of grid points and also less no. of basis functions contributing to a particular batch/block of grid points.
# Furthermore, PySCF takes very little amount of time to generate grids as opposed to numgrid which is much slower.

#LDA
# funcx = 1
# funcc = 7
# funcidcrysx = [funcx, funcc]
# funcidpyscf = str(funcx)+','+str(funcc)

#GGA
funcx = 101
funcc = 130
funcidcrysx = [funcx, funcc]
funcidpyscf = str(funcx)+','+str(funcc)

# basis_set_name = 'sto-2g'
# basis_set_name = 'sto-3g'
# basis_set_name = 'sto-6g'
# basis_set_name = '6-31G'
basis_set_name = 'def2-SVP'
# basis_set_name = 'def2-DZVP'
# basis_set_name = 'def2-TZVP'
# basis_set_name = 'def2-TZVPP'
# basis_set_name = 'def2-TZVPPD'
# basis_set_name = 'def2-QZVPPD'
# basis_set_name = 'cc-pVDZ'
# basis_set_name = 'ano-rcc'

auxbasis_name = 'def2-universal-jfit'
# auxbasis_name = 'sto-3g'
# auxbasis_name = 'def2-SVP'

# xyzFilename = 'Benzene-Fulvene_Dimer.xyz'
# xyzFilename = 'Zn.xyz'
# xyzFilename = 'Zn_dimer.xyz'
# xyzFilename = 'TPP.xyz'
# xyzFilename = 'Zn_TPP.xyz'
# xyzFilename = 'h2o.xyz'
# xyzFilename = 'Ethane.xyz'
# xyzFilename = 'Cholesterol.xyz'
# xyzFilename = 'Serotonin.xyz'
# xyzFilename = 'Decane_C10H22.xyz'
xyzFilename = 'Icosane_C20H42.xyz'
# xyzFilename = 'Tetracontane_C40H82.xyz'
# xyzFilename = 'Pentacontane_C50H102.xyz'
# xyzFilename = 'Octacontane_C80H162.xyz'
# xyzFilename = 'Hectane_C100H202.xyz'
# xyzFilename = 'Icosahectane_C120H242.xyz'


#Initialize a Mol object with somewhat large geometry
mol = Mol(coordfile=xyzFilename)


#Initialize a Basis object with a very large basis set
basis = Basis(mol, {'all':Basis.load(mol=mol, basis_name=basis_set_name)})

auxbasis = Basis(mol, {'all':Basis.load(mol=mol, basis_name=auxbasis_name)})

dftObj = DFT(mol, basis, auxbasis, xc=funcidcrysx)

dftObj.conv_crit = 1e-7
dftObj.max_itr = 20
dftObj.ncores = ncores
dftObj.save_ao_values = True
# Using CrysX grids 
# To get the same energies as PySCF (level=5) upto 1e-7 au, use the following settings
# radial_precision=1.0e-13
# level=3
# pruning by density with threshold = 1e-011
# alpha_min and alpha_max corresponding to QZVP
energyCrysX, dmat = dftObj.scf()
