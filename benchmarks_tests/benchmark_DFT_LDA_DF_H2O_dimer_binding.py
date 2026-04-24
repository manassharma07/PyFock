import os
from timeit import default_timer as timer

import numpy as np
from pyscf import dft, gto

from pyfock import Basis, DFT, Mol, Utils


benchmark_ncores = 4
os.environ["OMP_NUM_THREADS"] = str(benchmark_ncores)
os.environ["OPENBLAS_NUM_THREADS"] = str(benchmark_ncores)
os.environ["MKL_NUM_THREADS"] = str(benchmark_ncores)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(benchmark_ncores)
os.environ["NUMEXPR_NUM_THREADS"] = str(benchmark_ncores)
os.environ["PYSCF_MAX_MEMORY"] = str(25000)


Utils.print_sys_info()
print("Number of cores being actually used/requested for the benchmark:", benchmark_ncores)
print("OMP_NUM_THREADS =", os.environ.get("OMP_NUM_THREADS"))
print("OPENBLAS_NUM_THREADS =", os.environ.get("OPENBLAS_NUM_THREADS"))
print("MKL_NUM_THREADS =", os.environ.get("MKL_NUM_THREADS"))
print("VECLIB_MAXIMUM_THREADS =", os.environ.get("VECLIB_MAXIMUM_THREADS"))
print("NUMEXPR_NUM_THREADS =", os.environ.get("NUMEXPR_NUM_THREADS"))
print("PYSCF_MAX_MEMORY =", os.environ.get("PYSCF_MAX_MEMORY"))


# LDA_X + LDA_C_VWN (SVWN / LDA)
funcx = 1
funcc = 7

funcidcrysx = [funcx, funcc]
funcidpyscf = f"{funcx},{funcc}"

basis_set_name = "def2-SVP"
auxbasis_name = "def2-universal-jfit"
grids_level = 3

xyzFilename = os.path.join(os.path.dirname(__file__), "H2O_dimer_stable.xyz")

HARTREE_TO_KCAL_MOL = 627.5094740631


def read_xyz_atoms(filename):
    with open(filename, "r", encoding="utf-8") as handle:
        lines = handle.readlines()

    natoms = int(lines[0].strip())
    atoms = []
    for line in lines[2 : 2 + natoms]:
        symbol, x, y, z = line.split()[:4]
        atoms.append([symbol, float(x), float(y), float(z)])
    return atoms


def make_ghost_atoms(atoms):
    return [[f"Ghost-{symbol}", x, y, z] for symbol, x, y, z in atoms]


def atoms_to_pyscf(atoms):
    return [(symbol, (x, y, z)) for symbol, x, y, z in atoms]


def run_pyscf(label, atoms):
    mol = gto.Mole()
    mol.atom = atoms_to_pyscf(atoms)
    mol.basis = basis_set_name
    mol.cart = False
    mol.verbose = 0
    mol.max_memory = 5000
    mol.build()

    mf = dft.rks.RKS(mol).density_fit(auxbasis=auxbasis_name)
    mf.xc = funcidpyscf
    mf.direct_scf = False
    mf.init_guess = "minao"
    dmat_init = mf.init_guess_by_minao(mol)
    mf.max_cycle = 50
    mf.conv_tol = 1e-8
    mf.grids.level = grids_level

    start = timer()
    energy = mf.kernel(dm0=dmat_init)
    duration = timer() - start

    print(f"\nPySCF {label} energy: {energy:.12f} Ha")
    print(f"PySCF {label} time: {duration:.3f} s")

    return {
        "energy": energy,
        "time": duration,
        "grids": mf.grids,
        "dmat_init": dmat_init,
    }


def run_pyfock(label, atoms, pyscf_ref):
    mol = Mol(atoms=atoms)
    if not mol.success:
        raise RuntimeError(f"PyFock Mol construction failed for {label}")

    basis = Basis(mol, {"all": Basis.load(mol=mol, basis_name=basis_set_name)})
    auxbasis = Basis(mol, {"all": Basis.load(mol=mol, basis_name=auxbasis_name)})

    dft_obj = DFT(mol, basis, auxbasis, xc=funcidcrysx, grids=pyscf_ref["grids"])
    dft_obj.dmat = pyscf_ref["dmat_init"]
    dft_obj.conv_crit = 1e-8
    dft_obj.max_itr = 50
    dft_obj.ncores = benchmark_ncores
    dft_obj.save_ao_values = True
    dft_obj.rys = True
    dft_obj.isDF = True
    dft_obj.DF_algo = 10
    dft_obj.blocksize = 5000
    dft_obj.XC_algo = 2
    dft_obj.debug = False
    dft_obj.sortGrids = False
    dft_obj.xc_bf_screen = True
    dft_obj.threshold_schwarz = 1e-9
    dft_obj.strict_schwarz = False
    dft_obj.cholesky = True
    dft_obj.orthogonalize = True
    dft_obj.sao = True
    dft_obj.use_gpu = False
    dft_obj.keep_ao_in_gpu = False
    dft_obj.use_libxc = False

    start = timer()
    energy, _ = dft_obj.scf()
    duration = timer() - start

    print(f"\nPyFock {label} energy: {energy:.12f} Ha")
    print(f"PyFock {label} time: {duration:.3f} s")

    return {"energy": energy, "time": duration}


def print_binding_summary(engine_name, energies):
    binding_no_cp = energies["dimer"] - energies["monomer_a"] - energies["monomer_b"]
    binding_cp = energies["dimer"] - energies["monomer_a_cp"] - energies["monomer_b_cp"]
    bsse = binding_no_cp - binding_cp

    print(f"\n{engine_name} binding energies")
    print(f"Without supersystem basis : {binding_no_cp:.12f} Ha  ({binding_no_cp * HARTREE_TO_KCAL_MOL:.6f} kcal/mol)")
    print(f"With supersystem basis    : {binding_cp:.12f} Ha  ({binding_cp * HARTREE_TO_KCAL_MOL:.6f} kcal/mol)")
    print(f"BSSE correction           : {bsse:.12f} Ha  ({bsse * HARTREE_TO_KCAL_MOL:.6f} kcal/mol)")


if __name__ == "__main__":
    dimer_atoms = read_xyz_atoms(xyzFilename)
    monomer_a = dimer_atoms[:3]
    monomer_b = dimer_atoms[3:]

    systems = {
        "dimer": dimer_atoms,
        "monomer_a": monomer_a,
        "monomer_b": monomer_b,
        "monomer_a_cp": monomer_a + make_ghost_atoms(monomer_b),
        "monomer_b_cp": make_ghost_atoms(monomer_a) + monomer_b,
    }

    print("\nRunning H2O dimer counterpoise benchmark")
    print("Geometry file:", xyzFilename)
    print("Basis:", basis_set_name)
    print("Aux basis:", auxbasis_name)
    print("XC:", funcidpyscf)

    pyscf_results = {}
    pyfock_results = {}

    for label, atoms in systems.items():
        print("\n" + "=" * 80)
        print("System:", label)
        print("=" * 80)
        pyscf_results[label] = run_pyscf(label, atoms)
        pyfock_results[label] = run_pyfock(label, atoms, pyscf_results[label])

    pyscf_energies = {label: result["energy"] for label, result in pyscf_results.items()}
    pyfock_energies = {label: result["energy"] for label, result in pyfock_results.items()}

    print_binding_summary("PySCF", pyscf_energies)
    print_binding_summary("PyFock", pyfock_energies)

    print("\nEnergy differences (PySCF - PyFock)")
    for label in systems:
        diff = pyscf_energies[label] - pyfock_energies[label]
        print(f"{label:14s}: {diff:.12e} Ha")
