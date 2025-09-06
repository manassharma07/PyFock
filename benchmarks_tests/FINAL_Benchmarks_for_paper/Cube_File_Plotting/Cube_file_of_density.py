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


basis_set_name = 'def2-SVP'

auxbasis_name = 'def2-universal-jfit'

xyzFilename = 'Benezene_UMA_OMOL_Optimized.xyz'


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

Utils.write_density_cube(mol, basis, dftObj.dmat, 'benzene_dens.cube', nx=100, ny=100, nz=100, ncores=ncores)