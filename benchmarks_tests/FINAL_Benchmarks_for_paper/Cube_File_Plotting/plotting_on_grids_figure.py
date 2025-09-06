from pyfock import Basis
from pyfock import Mol
from pyfock import DFT
from pyfock import Utils

mol = Mol(coordfile='Benzene.xyz')

basis = Basis(mol, {'all':Basis.load(mol=mol, basis_name='def2-SVP')})
auxbasis = Basis(mol, {'all':Basis.load(mol=mol, basis_name='def2-universal-jfit')})

dftObj = DFT(mol, basis, auxbasis, xc=[1, 7])
dftObj.scf()

Utils.write_density_cube(mol, basis, dftObj.dmat, 'benzene_dens_temp.cube')




