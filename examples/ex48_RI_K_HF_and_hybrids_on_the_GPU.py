"""Hartree-Fock and global hybrid functionals (RI-K) on the GPU.

With ``use_gpu=True`` an ``xc='HF'`` or hybrid (B3LYP, B3LYP5, PBE0) SCF now runs entirely on the
device: the screened three-center integrals of ``DF_algo=11`` (the default 12 falls back to it, as
on the CPU) are orthonormalized in the fit metric on the GPU and both the Coulomb and the exact
exchange matrix are contracted from these rows with cuBLAS; the semilocal part of the hybrid is
evaluated by the GPU XC driver (``XC_algo=3``). Nothing moves between host and device during the
iterations.

Dynamic precision (``dft.dynamic_precision``, the GPU default, also used for the XC term) contracts
the exchange in single precision until the relative energy change falls below 5e-7 and in double
precision from there on; the stored rows are always double, so the converged energy is a
double-precision one. On consumer GPUs with a 1/32-1/64 fp64 rate this is where most of the
exchange time goes, so leave it on unless you want every iteration in fp64.

Design, validation and timings: docs/ri_k_exchange.md (section 7) and tests/test_rik_gpu.py.
On Windows set ``CUPY_ACCELERATORS=`` (empty) in the shell first if CuPy cannot find ``cl.exe``.
"""
from timeit import default_timer as timer

from pyfock import Basis, DFT, Mol

ncores = 4
xc = "B3LYP"            # or "HF", "PBE0", "B3LYP5", [402], ...

mol = Mol(coordfile="Decane_C10H22.xyz")
basis = Basis(mol, {"all": Basis.load(mol=mol, basis_name="def2-SVP")})
# a JK-fitting auxiliary basis is needed for exact exchange (the J-fitting sets are too small for K)
auxbasis = Basis(mol, {"all": Basis.load(mol=mol, basis_name="def2-universal-jkfit")})


def run(use_gpu, dynamic_precision=True):
    dft = DFT(mol, basis, auxbasis, xc=xc, conv_crit=1e-7, ncores=ncores, use_gpu=use_gpu)
    dft.sao = True                      # spherical orbital and fit spaces
    dft.max_itr = 50
    dft.dynamic_precision = dynamic_precision   # only acts on GPU runs
    start = timer()
    energy, dmat = dft.scf()
    return float(energy), float(dft.Exx_energy), timer() - start, dft.niter


e_gpu, exx_gpu, t_gpu, n_gpu = run(use_gpu=True)
e_cpu, exx_cpu, t_cpu, n_cpu = run(use_gpu=False)

print("\n%s / def2-SVP / def2-universal-jkfit, RI-J + RI-K" % xc)
print("GPU: E = %.10f Ha, exact-exchange energy %.10f Ha, %d iterations, %.1f s" % (e_gpu, exx_gpu, n_gpu, t_gpu))
print("CPU: E = %.10f Ha, exact-exchange energy %.10f Ha, %d iterations, %.1f s" % (e_cpu, exx_cpu, n_cpu, t_cpu))
print("|E(GPU) - E(CPU)| = %.2e Ha" % abs(e_gpu - e_cpu))
