import os

ncores = 4
os.environ['OMP_NUM_THREADS'] = str(ncores)
os.environ["OPENBLAS_NUM_THREADS"] = str(ncores)
os.environ["MKL_NUM_THREADS"] = str(ncores)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(ncores)
os.environ["NUMEXPR_NUM_THREADS"] = str(ncores)


from pyfock import Basis
from pyfock import Mol
from pyfock import DFT


# This example requires PySCF to be installed because the grids are generated
# internally through PySCF.

# LDA
funcidpyfock = 'LDA'

basis_set_name = 'def2-SVP'
auxbasis_name = 'def2-universal-jfit'
xyzFilename = 'h2o.xyz'


# Initialize a Mol object
mol = Mol(coordfile=xyzFilename)


# Initialize Basis objects
basis = Basis(mol, {'all': Basis.load(mol=mol, basis_name=basis_set_name)})
auxbasis = Basis(mol, {'all': Basis.load(mol=mol, basis_name=auxbasis_name)})


# Ask PyFock to generate and use PySCF grids internally
dftObj = DFT(mol, basis, auxbasis, xc=funcidpyfock, use_pyscf_grids=True, gridsLevel=3)

dftObj.conv_crit = 1e-7
dftObj.max_itr = 20
dftObj.ncores = ncores
dftObj.save_ao_values = True

energyCrysX, dmat = dftObj.scf()
