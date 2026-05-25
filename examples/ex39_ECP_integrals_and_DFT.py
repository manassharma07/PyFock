import os

ncores = 4
os.environ['OMP_NUM_THREADS'] = str(ncores)
os.environ["OPENBLAS_NUM_THREADS"] = str(ncores) # or export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = str(ncores) #  or export MKL_NUM_THREADS=4
os.environ["VECLIB_MAXIMUM_THREADS"] = str(ncores)  # or  export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = str(ncores) # or  export NUMEXPR_NUM_THREADS=4


from pyfock import Basis
from pyfock import Mol
from pyfock import Integrals
from pyfock import DFT


# ECP integral and DFT example

atoms = [['Cd', 0.0, 0.0, 0.0]]

basis_set_name = 'def2-SVP'
auxbasis_name = 'def2-universal-jfit'

#First of all we need a mol object with some geometry
mol = Mol(atoms=atoms)

# Next we need to specify some basis
# The def2 basis sets include ECP data for elements that need it.
basis = Basis(mol, {'all':Basis.load(mol=mol, basis_name=basis_set_name)})

print('\n\n\n')
print('Integrals')
print('ECP matrix\n')
print('Number of active electrons after ECP: ', mol.nelectrons)
print('Number of ECP centers: ', len(basis.ecps))
#NOTE: The matrices are calculated in CAO basis and not the SAO basis
#You should refer to the example that shows the transformation between the two if you need matrices in SAO basis.
Vecp = Integrals.ecp_mat_symm(basis)
print(Vecp)
print(Vecp.shape)


# If needed, convert the same matrix to the spherical AO basis
Vecp_sph = basis.cart2sph_operator_blockwise(Vecp)
print('\n\nECP matrix in spherical AO basis')
print(Vecp_sph)
print(Vecp_sph.shape)


# DFT with ECP
# PyFock automatically adds the ECP matrix to the core Hamiltonian when the basis contains ECP data.
auxbasis = Basis(mol, {'all':Basis.load(mol=mol, basis_name=auxbasis_name)})

dftObj = DFT(mol, basis, auxbasis, xc='LDA')
dftObj.conv_crit = 1e-7
dftObj.max_itr = 20
dftObj.ncores = ncores
dftObj.save_ao_values = True # Requires more memory but is faster

energyCrysX, dmat = dftObj.scf()
print('DFT total energy with ECP: ', energyCrysX)
