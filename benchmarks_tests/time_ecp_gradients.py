# Timing harness for analytical DFT forces on the ECP example systems.
# Reports: PyFock SCF time, PyFock analytical-gradient time (warm) with the
# per-term breakdown (highlighting the finite-difference ECP term), and PySCF's
# analytical-gradient time for comparison. Numerical (finite-difference) force
# time is included only for the small systems (it needs 6*natoms SCFs).
import os
import sys
import numpy as np
from timeit import default_timer as timer

ncores = 4
os.environ["OMP_NUM_THREADS"] = str(ncores)
os.environ["OPENBLAS_NUM_THREADS"] = str(ncores)
os.environ["MKL_NUM_THREADS"] = str(ncores)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(ncores)
os.environ["NUMEXPR_NUM_THREADS"] = str(ncores)
os.environ["PYSCF_MAX_MEMORY"] = str(25000)

from pyfock import Basis, DFT, DFT_Grad, DFT_NumGrad, Mol
from pyscf import dft as pyscf_dft
from pyscf import gto

basis_set_name = "def2-SVP"
auxbasis_name = "def2-universal-jfit"
xc_pyfock = "PBE"
xc_pyscf = "PBE"

# (xyz, run_numerical?) -- numerical only for the small systems
systems = [
    ("AgCl.xyz", True),
    ("AuCl.xyz", True),
    ("Cd_dimer.xyz", True),
    ("XeF2.xyz", True),
    ("BiH3.xyz", False),
    ("SnCl4.xyz", False),
    ("W_CO6.xyz", False),
]
if len(sys.argv) > 1:
    systems = [(sys.argv[1], len(sys.argv) > 2 and sys.argv[2] == "numerical")]

rows = []
for xyz, do_num in systems:
    if not os.path.exists(xyz):
        print(f"skip {xyz} (not found)")
        continue
    try:
        # ----- PySCF (for grids, init guess, reference gradient timing) -----
        m = gto.Mole()
        m.atom = xyz
        m.basis = basis_set_name
        m.ecp = basis_set_name
        m.cart = False
        m.verbose = 0
        m.max_memory = 5000
        m.build()
        mf = pyscf_dft.rks.RKS(m).density_fit(auxbasis=auxbasis_name)
        mf.xc = xc_pyscf
        mf.init_guess = "minao"
        dm0 = mf.init_guess_by_minao(m)
        mf.conv_tol = 1e-9
        mf.max_cycle = 50
        mf.grids.level = 3
        mf.kernel(dm0=dm0)
        t = timer()
        mf.nuc_grad_method().kernel()
        t_pyscf_grad = timer() - t
        grids = mf.grids

        # ----- PyFock SCF -----
        mol = Mol(coordfile=xyz)
        basis = Basis(mol, {"all": Basis.load(mol=mol, basis_name=basis_set_name)})
        auxbasis = Basis(mol, {"all": Basis.load(mol=mol, basis_name=auxbasis_name)})
        d = DFT(mol, basis, auxbasis, xc=xc_pyfock, grids=grids)
        d.conv_crit = 1e-9
        d.max_itr = 50
        d.ncores = ncores
        d.save_ao_values = True
        d.isDF = True
        d.DF_algo = 10
        d.XC_algo = 2
        d.xc_bf_screen = True
        d.threshold_schwarz = 1e-9
        d.strict_schwarz = False
        d.cholesky = True
        d.orthogonalize = True
        d.sao = True
        d.use_gpu = False
        d.use_libxc = False
        d.dmat = dm0
        t = timer()
        d.scf()
        t_scf = timer() - t

        # ----- PyFock analytical gradient (warm) -----
        g = DFT_Grad(d, verbose=False)
        g.calculate()  # warm-up (JIT)
        t = timer()
        res = g.calculate()
        t_ana = timer() - t
        tm = res["timings"]
        t_ecp = tm.get("ecp", 0.0)
        t_1e = tm["kinetic"] + tm["nuclear_attraction"] + tm["overlap"]
        t_coul = tm["df_coefficients"] + tm["coulomb_3c2e_grad"] + tm["coulomb_2c2e_grad"]

        # ----- PyFock numerical (optional) -----
        t_num = None
        if do_num:
            ng = DFT_NumGrad(d, step_size=1e-3, step_unit="bohr", method="central",
                             use_fixed_grids=True, verbose=False)
            t = timer()
            ng.calculate()
            t_num = timer() - t

        rows.append((xyz, mol.natoms, basis.bfs_nao, basis.ecp_total_core_electrons,
                     t_scf, t_ana, t_1e, t_coul, tm["xc"], t_ecp, t_pyscf_grad, t_num))
        print(f"done {xyz}")
    except Exception as e:
        print(f"FAILED {xyz}: {e}")

print("\n\n" + "=" * 118)
print("Analytical DFT force timings for ECP systems (def2-SVP, PBE, DF_algo=10, 4 cores)  [seconds]")
print("=" * 118)
hdr = (f"{'system':<14}{'atoms':>6}{'nbf':>5}{'ncore':>6}{'SCF':>8}"
       f"{'GRAD(ana)':>10}{'  1e':>8}{'Coulomb':>9}{'XC':>8}{'ECP':>9}{'PySCF grad':>12}{'GRAD(num)':>11}")
print(hdr)
print("-" * 118)
for (xyz, nat, nbf, ncore, t_scf, t_ana, t_1e, t_coul, t_xc, t_ecp, t_py, t_num) in rows:
    num_str = f"{t_num:>11.2f}" if t_num is not None else f"{'-':>11}"
    print(f"{xyz:<14}{nat:>6}{nbf:>5}{ncore:>6}{t_scf:>8.2f}{t_ana:>10.3f}{t_1e:>8.3f}"
          f"{t_coul:>9.3f}{t_xc:>8.3f}{t_ecp:>9.3f}{t_py:>12.3f}{num_str}")
print("-" * 118)
print("GRAD(ana) = total PyFock analytical gradient (warm). ECP(fd) = finite-difference ECP term (subset of GRAD(ana)).")
print("GRAD(num) = PyFock numerical (finite-difference) forces = 6*natoms SCFs (small systems only).")
