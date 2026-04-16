import os
import numpy as np

ncores = 4

from pyfock import Basis
from pyfock import DFT
from pyfock import DFT_NumGrad
from pyfock import Mol


mol = Mol(coordfile="h2o.xyz")
basis = Basis(mol, {"all": Basis.load(mol=mol, basis_name="def2-SVP")})
auxbasis = Basis(mol, {"all": Basis.load(mol=mol, basis_name="def2-universal-jfit")})

dft_obj = DFT(mol, basis, auxbasis, xc="LDA")
dft_obj.conv_crit = 1e-7
dft_obj.max_itr = 20
dft_obj.ncores = ncores
dft_obj.save_ao_values = True

dft_obj.scf()

grad_obj = DFT_NumGrad(
    dft_obj, step_size=1.0e-3, step_unit="bohr", method="central"
)
results = grad_obj.calculate()

print("Total energy (Ha):", results["energy"])
print("Forces (Ha/Bohr):")
print(np.array2string(results["forces"], precision=8, suppress_small=False))
