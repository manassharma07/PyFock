from pyfock import Basis
from pyfock import Mol
from pyfock import Integrals

import numpy as np

# Converstion from Cartesian AO basis potential matrices of PyFock to Soherical AO basis

# basis_set_name = 'sto-2g'
# basis_set_name = 'sto-3g'
# basis_set_name = 'sto-6g'
# basis_set_name = '6-31G'
# basis_set_name = 'def2-SVP'
# basis_set_name = 'def2-DZVP'
# basis_set_name = 'def2-TZVP'
# basis_set_name = 'def2-QZVP'
# basis_set_name = 'def2-TZVPP'
# basis_set_name = 'def2-QZVPP'
# basis_set_name = 'def2-TZVPD'
basis_set_name = 'def2-QZVPD'
# basis_set_name = 'def2-TZVPPD'
# basis_set_name = 'def2-QZVPPD'
# basis_set_name = 'cc-pVDZ'
# basis_set_name = 'ano-rcc'

# xyzFilename = 'Benzene-Fulvene_Dimer.xyz'
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

#First of all we need a mol object with some geometry
mol = Mol(coordfile = xyzFilename)

# Next we need to specify some basis
# The basis set can then be used to calculate things like Overlap, KE, integrals/matrices.
basis = Basis(mol, {'all':Basis.load(mol=mol, basis_name=basis_set_name)})

#Now we can calculate integrals.
# This example shows how to calculate the kinetic matrix using the basis set object created.

#Let's calculate the complete kinetic potential matrix
print('\n\n\n')
print('Integrals')
print('Kinetic energy matrix (CAO basis)\n')
#NOTE: The matrices are calculated in CAO basis and not the SAO basis
#You should refer to the example that shows the transformation between the two if you need matrices in SAO basis.
Vkin_cart = Integrals.kin_mat_symm(basis)
print(Vkin_cart) 

print('\n\nKinetic energy matrix (SAO basis)\n')
c2sph_mat = basis.cart2sph_basis()
Vkin_sph = np.dot(c2sph_mat, np.dot(Vkin_cart, c2sph_mat.T))
print(Vkin_sph) 

#Comparison with PySCF
from pyscf import gto
molPySCF = gto.Mole()
molPySCF.atom = xyzFilename
molPySCF.basis = basis_set_name
molPySCF.cart = False
molPySCF.build()
# print(molPySCF.sph_labels())
# print(molPySCF.cart_labels())
print(molPySCF.ao_labels())


#Kinetic pot mat
Vkin_sph_pyscf = molPySCF.intor_symmetric('int1e_kin_sph')
print('\n\nPySCF')
print(Vkin_sph_pyscf)
print('Matrix dimensions: ', Vkin_sph_pyscf.shape)
print(abs(Vkin_sph_pyscf - Vkin_sph).max())  #There will sometimes be a difference b/w PySCF and CrysX values because PySCF doesn't have the same ordering of g,h,.. orbitals.

