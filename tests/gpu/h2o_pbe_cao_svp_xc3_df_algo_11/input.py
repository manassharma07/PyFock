"""Native-grid GPU GGA regression (GPU-native XC algorithm 3), referenced to the CPU (XC algorithm 2)."""
import os
from pyfock import Mol, Basis, DFT
ncores = int(os.environ.get('OMP_NUM_THREADS', '4'))
mol = Mol(coordfile='h2o.xyz')
basis = Basis(mol, {'all': Basis.load(mol=mol, basis_name='def2-SVP')})
aux = Basis(mol, {'all': Basis.load(mol=mol, basis_name='def2-universal-jfit')})
dft = DFT(mol, basis, aux, xc=[101, 130], conv_crit=1e-7, gridsLevel=3,
          use_pyscf_grids=False, blocksize=5000, save_ao_values=True, use_gpu=True, ncores=ncores)
# The reference output was generated with the core-Hamiltonian guess (PyFock's default is now 'sano')
dft.dmat_guess_method = "core"
dft.max_itr = 35
dft.XC_algo = 3
dft.DF_algo = 11
dft.sao = False
dft.strict_schwarz = True
dft.threshold_schwarz = 1e-9
dft.max_memory_ints3c2e = None
dft.use_libxc = False
dft.scf()
