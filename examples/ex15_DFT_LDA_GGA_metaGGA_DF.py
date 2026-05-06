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
from pyfock import DFT

#LDA

# DFT functional can be specified either like [1, 7]
# funcx = 1
# funcc = 7
# funcidcrysx = [funcx, funcc]
#
# or using predefined strings
# funcidpyfock = 'LDA' # [1, 7]
# funcidpyfock = 'SPZ' # [1, 9]
# funcidpyfock = 'SPW' # [1, 12]
#
# Natively implemented GGA functionals
funcidpyfock = 'PBE' # [101, 130]
# funcidpyfock = 'PBESOL' # [116, 133]
# funcidpyfock = 'RPBE' # [117, 130]
# funcidpyfock = 'PW91' # [109, 134]
# funcidpyfock = 'BP86' # [106, 132]
# funcidpyfock = 'BLYP' # [106, 131]
#
# Natively implemented metaGGA functionals
# funcidpyfock = 'R2SCAN' # [497, 498]
# funcidpyfock = 'TPSS' # [202, 231]
# funcidpyfock = 'M06L' # [203, 233]
# funcidpyfock = 'TASK' # [707]



# basis_set_name = 'sto-3g'
basis_set_name = 'def2-SVP'
# basis_set_name = 'def2-TZVP'


auxbasis_name = 'def2-universal-jfit'


xyzFilename = 'h2o.xyz'


# Initialize a Mol object 
mol = Mol(coordfile=xyzFilename)


#Initialize a Basis object 
basis = Basis(mol, {'all':Basis.load(mol=mol, basis_name=basis_set_name)})
#Initialize an auxiliary basis object for density fitting
auxbasis = Basis(mol, {'all':Basis.load(mol=mol, basis_name=auxbasis_name)})

dftObj = DFT(mol, basis, auxbasis, xc=funcidpyfock)

dftObj.conv_crit = 1e-7
dftObj.max_itr = 20
dftObj.ncores = ncores
dftObj.save_ao_values = True # Requires more memory but is faster
energyCrysX, dmat = dftObj.scf()
