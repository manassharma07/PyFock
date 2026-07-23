# Verify that DFT_Grad gives identical forces regardless of the DF algorithm
# used during the SCF (it only consumes D, the MOs and the grids).
import os
import numpy as np

ncores = 4
os.environ["OMP_NUM_THREADS"] = str(ncores)
os.environ["OPENBLAS_NUM_THREADS"] = str(ncores)
os.environ["MKL_NUM_THREADS"] = str(ncores)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(ncores)
os.environ["NUMEXPR_NUM_THREADS"] = str(ncores)

from pyfock import Basis, DFT, DFT_Grad, Mol
from pyscf import dft as pyscf_dft
from pyscf import gto

xyz = "H2O.xyz"
basis_name = "def2-SVP"
aux_name = "def2-universal-jfit"

mol_pyscf = gto.Mole()
mol_pyscf.atom = xyz
mol_pyscf.basis = basis_name
mol_pyscf.cart = False
mol_pyscf.verbose = 0
mol_pyscf.build()
mf = pyscf_dft.rks.RKS(mol_pyscf).density_fit(auxbasis=aux_name)
mf.xc = "101,130"
mf.init_guess = "atom"
dm0 = mf.init_guess_by_atom(mol_pyscf)
mf.conv_tol = 1e-11
mf.grids.level = 3
mf.kernel(dm0=dm0)
forces_pyscf = -mf.nuc_grad_method().kernel()
grids = mf.grids


def forces_with_algo(df_algo, cholesky):
    m = Mol(coordfile=xyz)
    b = Basis(m, {"all": Basis.load(mol=m, basis_name=basis_name)})
    ab = Basis(m, {"all": Basis.load(mol=m, basis_name=aux_name)})
    d = DFT(m, b, ab, xc="PBE", grids=grids)
    d.conv_crit = 1e-11
    d.max_itr = 60
    d.ncores = ncores
    d.isDF = True
    d.DF_algo = df_algo
    d.XC_algo = 2
    d.xc_bf_screen = True
    d.threshold_schwarz = 1e-9
    d.strict_schwarz = False
    d.cholesky = cholesky  # cholesky only supported for DF_algo 6/10
    d.orthogonalize = True
    d.sao = True
    d.use_gpu = False
    d.use_libxc = False
    d.dmat = dm0
    d.scf()
    res = DFT_Grad(d, verbose=False).calculate()
    return res["forces"]


f1 = forces_with_algo(1, cholesky=False)
f10 = forces_with_algo(10, cholesky=True)

print("\n\nMax |F(DF_algo=1) - F(DF_algo=10)|  :", np.abs(f1 - f10).max())
print("Max |F(DF_algo=1)  - F(PySCF)|      :", np.abs(f1 - forces_pyscf).max())
print("Max |F(DF_algo=10) - F(PySCF)|      :", np.abs(f10 - forces_pyscf).max())
