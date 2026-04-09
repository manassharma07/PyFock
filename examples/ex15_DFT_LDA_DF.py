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
funcidpyfock = 'SPZ' # [1, 9]
# funcidpyfock = 'SPW' # [1, 12]


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

# xyzFilename = 'Benzene-Fulvene_Dimer.xyz'
# xyzFilename = 'Zn.xyz'
# xyzFilename = 'Zn_dimer.xyz'
# xyzFilename = 'TPP.xyz'
# xyzFilename = 'Zn_TPP.xyz'
xyzFilename = 'H2O.xyz'
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
dftObj.isDF = True # Use density fitting
energyCrysX, dmat = dftObj.scf()
