"""Skala, the neural exchange-correlation functional, in PyFock.

Skala is a machine-learned functional from Microsoft Research AI for Science. Select it by name, like
any other functional, and everything else works as usual.

Skala is parametrised together with a DFT-D3 correction, which `dispersion=True` adds after the SCF.

`use_gpu=True` runs the whole SCF on the GPU, Skala included, and is the fastest way to use it.
`skala_gpu=True` is the lighter alternative: it moves only the neural functional to the device and
leaves the rest of the SCF on the CPU. Both need a CUDA build of PyTorch, and `use_gpu` also needs CuPy.

Needs:  pip install "pyfock[skala]"
The 2.4 MB checkpoint is downloaded from Hugging Face on first use and cached afterwards.
"""

from pyfock import Basis, DFT, DFT_Grad, Mol

mol = Mol(coordfile='h2o.xyz')
basis = Basis(mol, {'all': Basis.load(mol=mol, basis_name='def2-SVP')})
auxbasis = Basis(mol, {'all': Basis.load(mol=mol, basis_name='def2-universal-jfit')})

dftObj = DFT(mol, basis, auxbasis, xc='skala-1.1', dispersion=True,
             use_gpu=False)  # True runs the whole SCF, Skala included, on the GPU
dftObj.conv_crit = 1e-8
dftObj.ncores = 4

energy, dmat = dftObj.scf()
forces = DFT_Grad(dftObj).calculate()['forces']

print('\nEnergy (Skala + D3) = %.8f Ha' % energy)
print('forces (Ha/Bohr):')
print(forces)
