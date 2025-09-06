from pyfock import Basis
from pyfock import Mol
from pyfock import Integrals
from pyfock import DFT
from pyfock import Utils

from timeit import default_timer as timer
import numpy as np
import scipy

ncores = 4

#LDA
funcx = 1
funcc = 7
funcidcrysx = [funcx, funcc]
funcidpyscf = str(funcx)+','+str(funcc)


basis_set_name = '6-31G'

auxbasis_name = 'def2-universal-jfit'

xyzFilename = 'h2o.xyz'
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
energyCrysX, dmat = dftObj.scf()

Utils.write_density_cube(mol, basis, dftObj.dmat, 'water_dens.cube', nx=100, ny=100, nz=100, ncores=ncores)