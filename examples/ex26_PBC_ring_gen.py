from pyfock import Basis
from pyfock import Mol
from pyfock import Integrals
from pyfock import DFT
from pyfock import Utils
from pyfock import PBC_ring

from timeit import default_timer as timer
import numpy as np
import scipy

ncores = 4

#LDA
funcx = 1
funcc = 7
funcidcrysx = [funcx, funcc]
funcidpyscf = str(funcx)+','+str(funcc)


basis_set_name = 'sto-3g'

auxbasis_name = 'def2-universal-jfit'

xyzFilename = 'LiH.xyz'


#Initialize a Mol object with somewhat large geometry
unit_mol = Mol(coordfile=xyzFilename)
ring_mol = PBC_ring.ring(unit_mol, N=10, periodicity=3.2, periodic_dir = 'x', 
        output_xyz = True, xyz_filename = 'pbc_ring_10')


#Initialize a Basis object with a very large basis set
basis = Basis(ring_mol, {'all':Basis.load(mol=ring_mol, basis_name=basis_set_name)})

auxbasis = Basis(ring_mol, {'all':Basis.load(mol=ring_mol, basis_name=auxbasis_name)})

dftObj = DFT(ring_mol, basis, auxbasis, xc=funcidcrysx)

dftObj.conv_crit = 1e-7
dftObj.max_itr = 20
dftObj.ncores = ncores
dftObj.save_ao_values = True
energyCrysX, dmat = dftObj.scf()
