from pyfock import Basis
from pyfock import Mol
from pyfock import Integrals
from timeit import default_timer as timer
import numpy as np

from pyscf import gto, dft
import numba 

from opt_einsum import contract


#KINETIC POTENTIAL MATRIX BENCHMARK and comparison with PySCF
#Benchmarking and performance assessment and comparison using various techniques and different softwares

# basis_set_name = 'sto-2g'
basis_set_name = 'sto-3g'
# basis_set_name = 'sto-6g'
# basis_set_name = 'def2-SVP'
# basis_set_name = 'def2-DZVP'
# basis_set_name = 'def2-TZVP'
# basis_set_name = 'def2-TZVPPD'
# basis_set_name = 'def2-QZVPPD'
# basis_set_name = 'ano-rcc'

# xyzFilename = 'Benzene-Fulvene_Dimer.xyz'
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

#First of all we need a mol object with some geometry
mol = Mol(coordfile = xyzFilename)

# Next we need to specify some basis
# The basis set can then be used to calculate things like Overlap, KE, integrals/matrices.
basis = Basis(mol, {'all':Basis.load(mol=mol, basis_name=basis_set_name)})


print('\n\n\n')
print('PyFock')
#NOTE: The matrices are calculated in CAO basis and not the SAO basis
#You should refer to the example that shows the transformation between the two if you need matrices in SAO basis.
start = timer()
S = Integrals.overlap_mat_symm(basis)
print(S) 
duration = timer() - start
print('Matrix dimensions: ', S.shape)
print('Duration for S using PyFock: ', duration)


# run PySCF calculation
mol = gto.M(
    atom=xyzFilename,  # just removing the last line that was only for psi4
    basis=basis_set_name,
    symmetry=False,
    cart=True,
)


mf = dft.rks.RKS(mol, xc=[1, 7])
mf.grids.level = 5
mf.grids.build()

coords = mf.grids.coords
weights = mf.grids.weights

def get_ovlp_numint(mol, grids):
    print('Calculating ao values....')
    ao = dft.numint.eval_ao(mol, grids.coords, deriv=0)
    print('Done!')
    weights = grids.weights.reshape(grids.weights.shape[0], 1)
    print('Calculating S numerically ....')
    ao = ao*np.sqrt(weights)
    s = contract('xi,xj->ij', ao, ao)
    print('Done!')
    return s

S_numint = get_ovlp_numint(mol, mf.grids)
print(S_numint)


print(abs(S-S_numint).max())


# Coulomb matrix 

mf.init_guess = 'minao'
dmat_init = mf.init_guess_by_minao(mol)

ERI_pyscf = mol.intor('int2e')
J = contract('ijkl,ij', ERI_pyscf, dmat_init)

print('J Analytical', J)

def get_J_numint(mol, grids, dmat):
    print('Calculating ao values....')
    ao = dft.numint.eval_ao(mol, grids.coords, deriv=0)
    print('Done!')
    print('Calculating rho values....')
    rho = dft.numint.eval_rho(mol, ao, dmat, xctype='LDA')
    print('Done!')
    weights = grids.weights.reshape(grids.weights.shape[0], 1)
    rho = rho.reshape(rho.shape[0], 1)
    print('Calculating J numerically ....')
    ao = ao*np.sqrt(weights)*np.sqrt(rho)
    J = contract('xi,xj->ij', ao, ao)
    print('Done!')
    return J

J_numint = get_J_numint(mol, mf.grids, dmat_init)
print(J_numint)


print(abs(J-J_numint).max())
