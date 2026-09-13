# SANO initial guess: superposition of atomic natural-orbital densities
#
# The SANO guess builds the starting density of the SCF from spherically averaged free-atom
# densities. No atomic SCF is run: the contracted functions of the ANO-RCC-MB minimal basis
# (shipped with PyFock) are the natural orbitals of the free atoms, so each atomic density is a
# diagonal matrix in that basis (2 electrons per closed shell, the open-shell electrons spread
# evenly over the 2l+1 orbitals). The superposition is projected onto the calculation basis with
# the cross overlap between the two basis sets, exactly as PySCF's 'minao' guess does.
# Compared with the core-Hamiltonian guess this typically halves the number of SCF iterations.
# The SCF output lists the references to cite for the guess.
import os

ncores = 4
os.environ['OMP_NUM_THREADS'] = str(ncores)
os.environ['OPENBLAS_NUM_THREADS'] = str(ncores)
os.environ['MKL_NUM_THREADS'] = str(ncores)
os.environ['VECLIB_MAXIMUM_THREADS'] = str(ncores)
os.environ['NUMEXPR_NUM_THREADS'] = str(ncores)

import numpy as np
from pyfock import Basis, DFT, Guess, Integrals, Mol

mol = Mol(coordfile='Serotonin.xyz')
basis = Basis(mol, {'all': Basis.load(mol=mol, basis_name='def2-SVP')})
auxbasis = Basis(mol, {'all': Basis.load(mol=mol, basis_name='def2-universal-jfit')})

# 1. The guess density itself (CAO representation), e.g. to inspect it or to reuse it
dmat_guess, info = Guess.sano_dmat(mol, basis)
S = Integrals.overlap_mat_symm(basis)
print('Electrons in the SANO guess density:', round(float(np.einsum('ij,ji->', dmat_guess, S)), 4), 'of', mol.nelectrons)
print('Atoms described by ANO-RCC-MB natural orbitals:', info['species'])
print('Fraction of the atomic natural orbitals representable in def2-SVP: %.2f %%' % (100 * info['nelectrons_projected'] / info['nelectrons']))
print(Guess.sano_citation_text(info['nuclear_charges']))

# 2. SCF with the SANO guess (dmat_guess_method='sano'); compare with the core-Hamiltonian guess ('core')
for guess in ('core', 'sano'):
    dft = DFT(mol, basis, auxbasis, xc='PBE', dmat_guess_method=guess, ncores=ncores, save_ao_values=True)
    dft.conv_crit = 1e-7
    energy, dmat = dft.scf()
    print('\nGuess %-4s : E = %.8f Hartree, converged in %d SCF iterations\n' % (guess, energy, dft.niter))
