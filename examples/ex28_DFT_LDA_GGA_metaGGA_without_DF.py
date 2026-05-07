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



# DFT functional can be specified either like [1, 7]
# funcx = 1
# funcc = 7
# funcidcrysx = [funcx, funcc]
#
# or using predefined strings
# funcidpyfock = 'LDA' # [1, 7]
# funcidpyfock = 'SPZ' # [1, 9]
# funcidpyfock = 'SPZMOD' # [1, 10]
# funcidpyfock = 'SPW' # [1, 12]
# funcidpyfock = 'SPWMOD' # [1, 12]
#
# Natively implemented GGA functionals
# funcidpyfock = 'PBE' # [101, 130]
# funcidpyfock = 'PBESOL' # [116, 133]
# funcidpyfock = 'RPBE' # [117, 130]
# funcidpyfock = 'PW91' # [109, 134]
# funcidpyfock = 'BP86' # [106, 132]
# funcidpyfock = 'BLYP' # [106, 131]
#
# Natively implemented metaGGA functionals
funcidpyfock = 'R2SCAN' # [497, 498]
# funcidpyfock = 'TPSS' # [202, 231]
# funcidpyfock = 'M06L' # [203, 233]
# funcidpyfock = 'TASK' # [707]



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

auxbasis_name = 'def2-universal-jfit'

xyzFilename = 'H2O.xyz'


# Initialize a Mol object 
mol = Mol(coordfile=xyzFilename)


#Initialize a Basis object 
basis = Basis(mol, {'all':Basis.load(mol=mol, basis_name=basis_set_name)})


# APPROACH 1: Good for only small systems as it requires storing significant ERIs in memory
dftObj = DFT(mol, basis, xc=funcidpyfock)

dftObj.conv_crit = 1e-7
dftObj.max_itr = 20
dftObj.ncores = ncores
dftObj.save_ao_values = True # Requires more memory but is faster
dftObj.isDF = False # Disable density fitting
dftObj.coul_algo = 2 # When DF is disabled, PyFock uses the coul_algo=2 by default. This means that the singificant 4c2e ERIs are stored in memory.
energyCrysX, dmat = dftObj.scf()



# APPROACH 2: Direct SCF. Good for small to medium sized systems with extremely low memory footprint at the cost of speed.
# NOTE: The first iteration of direct_scf mode also includes just-in-time compilation time, so the timings may appear off.
dftObj = DFT(mol, basis, xc=funcidpyfock)

dftObj.conv_crit = 1e-7
dftObj.max_itr = 20
dftObj.ncores = ncores
dftObj.save_ao_values = True # Requires more memory but is faster
dftObj.isDF = False # Disable density fitting
dftObj.direct_scf = True # Coulomb matrix is computed on the fly without storing the large 4c2e ERI tensor.
dftObj.rys = False # By default PyFock uses Rys quadrature for the evaluation of the ERIs. But Obara-Saika (OS) scheme is faster for direct_scf.
energyCrysX, dmat = dftObj.scf()


# NOTE: The above examples are only for demonstration purposes. For production runs, one should always use density fitting
# as it gives significant acceleration without compromising accuracy.