"""Skala, the neural exchange-correlation functional, in PyFock.

Skala is a machine-learned functional from Microsoft Research AI for Science. Select it by name, like
any other functional, and everything else works as usual.

Skala is parametrised together with a DFT-D3 correction, which `dispersion=True` adds after the SCF.

It is a PyTorch model, so `skala_gpu=True` evaluates it on the GPU while the integrals, the Coulomb
term and the assembly of Vxc stay on the CPU. That is where nearly all of Skala's cost sits, so it is
worth setting whenever a CUDA build of PyTorch is installed; nothing else about the calculation changes.

Needs:  pip install "pyfock[skala]"
The 2.4 MB checkpoint is downloaded from Hugging Face on first use and cached afterwards.
"""

from pyfock import Basis, DFT, DFT_Grad, Mol

mol = Mol(coordfile='h2o.xyz')
basis = Basis(mol, {'all': Basis.load(mol=mol, basis_name='def2-SVP')})
auxbasis = Basis(mol, {'all': Basis.load(mol=mol, basis_name='def2-universal-jfit')})

dftObj = DFT(mol, basis, auxbasis, xc='skala-1.1', dispersion=True,
             skala_gpu=False)  # True evaluates the functional on the GPU (needs CUDA PyTorch)
dftObj.conv_crit = 1e-8
dftObj.ncores = 4

energy, dmat = dftObj.scf()
forces = DFT_Grad(dftObj).calculate()['forces']

print('\nEnergy (Skala + D3) = %.8f Ha' % energy)
print('forces (Ha/Bohr):')
print(forces)
