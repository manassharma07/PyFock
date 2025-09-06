from pyfock import Graphics as gfx
# from pyfock import Graphics_old as gfx_old
from pyfock import Mol

# mol = Mol(coordfile='Graphene_C20.xyz')
mol = Mol(coordfile='h2o.xyz')

gfx.visualize(mol)