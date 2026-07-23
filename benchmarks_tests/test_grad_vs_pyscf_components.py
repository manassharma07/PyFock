# Compare PyFock analytical gradient components against PySCF's gradient
# decomposition term by term: hcore (T+V), Pulay, XC, DF-Coulomb.
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

basis_set_name = "def2-SVP"
auxbasis_name = "def2-universal-jfit"
xyz_filename = "H2O.xyz"

mol = gto.Mole()
mol.atom = xyz_filename
mol.basis = basis_set_name
mol.cart = False
mol.verbose = 0
mol.build()
mf = pyscf_dft.rks.RKS(mol).density_fit(auxbasis=auxbasis_name)
mf.xc = "101,130"
mf.init_guess = "atom"
dm0_guess = mf.init_guess_by_atom(mol)
mf.conv_tol = 1e-12
mf.grids.level = 3
mf.kernel(dm0=dm0_guess)
pyscf_grids = mf.grids
dm0 = mf.make_rdm1()

g = mf.nuc_grad_method()
aoslices = mol.aoslice_by_atom()
natm = mol.natm

# ---- hcore (T + V + HF operator term) ----
hcore_deriv = g.hcore_generator(mol)
de_hcore = np.zeros((natm, 3))
for ia in range(natm):
    de_hcore[ia] = np.einsum('xij,ij->x', hcore_deriv(ia), dm0)

# ---- Pulay ----
dme0 = g.make_rdm1e()
s1 = g.get_ovlp(mol)  # = -<d/dr mu | nu>
de_pulay = np.zeros((natm, 3))
for ia in range(natm):
    p0, p1 = aoslices[ia, 2], aoslices[ia, 3]
    de_pulay[ia] = -np.einsum('xij,ij->x', s1[:, p0:p1], dme0[p0:p1]) * 2

# ---- XC ----
from pyscf.grad import rks as rks_grad
ni = mf._numint
exc_grad, vmat_xc = rks_grad.get_vxc(ni, mol, mf.grids, mf.xc, dm0)
de_xc = np.zeros((natm, 3))
for ia in range(natm):
    p0, p1 = aoslices[ia, 2], aoslices[ia, 3]
    de_xc[ia] = np.einsum('xij,ij->x', vmat_xc[:, p0:p1], dm0[p0:p1]) * 2

# ---- DF Coulomb (3c + 2c + aux response) ----
vj = g.get_j(mol, dm0)
de_j = np.zeros((natm, 3))
for ia in range(natm):
    p0, p1 = aoslices[ia, 2], aoslices[ia, 3]
    de_j[ia] = np.einsum('xij,ij->x', vj[:, p0:p1], dm0[p0:p1]) * 2
if hasattr(vj, 'aux'):
    de_j += np.asarray(vj.aux).reshape(-1, natm, 3).sum(axis=0)

# ---- nuclear repulsion ----
de_nn = g.grad_nuc()

total_pyscf = de_hcore + de_pulay + de_xc + de_j + de_nn
print("PySCF reassembled total gradient:\n", np.array2string(total_pyscf, precision=10))
print("PySCF kernel gradient:\n", np.array2string(g.kernel(), precision=10))

# =========================== PyFock ===========================
mol_pf = Mol(coordfile=xyz_filename)
basis = Basis(mol_pf, {"all": Basis.load(mol=mol_pf, basis_name=basis_set_name)})
auxbasis = Basis(mol_pf, {"all": Basis.load(mol=mol_pf, basis_name=auxbasis_name)})
dft_obj = DFT(mol_pf, basis, auxbasis, xc="PBE", grids=pyscf_grids)
dft_obj.conv_crit = 1e-12
dft_obj.max_itr = 60
dft_obj.ncores = ncores
dft_obj.save_ao_values = True
dft_obj.isDF = True
dft_obj.DF_algo = 10
dft_obj.XC_algo = 2
dft_obj.sortGrids = False
dft_obj.xc_bf_screen = True
dft_obj.threshold_schwarz = 1e-9
dft_obj.strict_schwarz = False
dft_obj.cholesky = True
dft_obj.orthogonalize = True
dft_obj.sao = True
dft_obj.use_gpu = False
dft_obj.use_libxc = False
dft_obj.dmat = dm0_guess
dft_obj.scf()

grad_obj = DFT_Grad(dft_obj, verbose=False)
res = grad_obj.calculate()
comp = res["gradient_components"]

pf_hcore = comp["kinetic"] + comp["nuclear_attraction"]
print("\n--- component comparison (PyFock - PySCF), max abs ---")
print("hcore (T+V):", np.abs(pf_hcore - de_hcore).max())
print("pulay      :", np.abs(comp["overlap_pulay"] - de_pulay).max())
print("xc         :", np.abs(comp["xc"] - de_xc).max())
print("coulomb_df :", np.abs(comp["coulomb_df"] - de_j).max())
print("nuc_rep    :", np.abs(comp["nuclear_repulsion"] - de_nn).max())

print("\nPySCF de_xc:\n", np.array2string(de_xc, precision=10))
print("PyFock xc :\n", np.array2string(comp["xc"], precision=10))
print("\nPySCF de_j:\n", np.array2string(de_j, precision=10))
print("PyFock  j :\n", np.array2string(comp["coulomb_df"], precision=10))
print("\nPySCF hcore:\n", np.array2string(de_hcore, precision=10))
print("PyFock hcore:\n", np.array2string(pf_hcore, precision=10))
