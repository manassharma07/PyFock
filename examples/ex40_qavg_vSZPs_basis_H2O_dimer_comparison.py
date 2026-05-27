import os
from contextlib import redirect_stderr
from contextlib import redirect_stdout

from pyfock import Basis
from pyfock import DFT
from pyfock import Mol


ncores = 4


HARTREE_TO_KCAL_MOL = 627.5094740631
xyz_files = {
    "dimer": "h2o_dimer.xyz",
    "water1": "h2o_dimer_water1.xyz",
    "water2": "h2o_dimer_water2.xyz",
}


def energy(xyz_file, basis_name):
    mol = Mol(coordfile=xyz_file)
    basis = Basis(mol, {"all": Basis.load(mol=mol, basis_name=basis_name)})

    with open(os.devnull, "w", encoding="utf-8") as devnull:
        with redirect_stdout(devnull), redirect_stderr(devnull):
            dft = DFT(mol, basis, xc="R2SCAN", use_pyscf_grids=True)
            dft.conv_crit = 1e-7
            dft.max_itr = 20
            dft.ncores = ncores
            dft.save_ao_values = True
            dft.sao = True
            total_energy, _ = dft.scf()
            return total_energy


for basis_name in ["qavg-vSZPs", "def2-SVP", "def2-TZVP", "def2-QZVP"]:
    dimer = energy(xyz_files["dimer"], basis_name)
    water1 = energy(xyz_files["water1"], basis_name)
    water2 = energy(xyz_files["water2"], basis_name)
    binding = dimer - water1 - water2

    print("\nBasis:", basis_name)
    print(f"Dimer energy   : {dimer:.12f} Ha")
    print(f"Water 1 energy : {water1:.12f} Ha")
    print(f"Water 2 energy : {water2:.12f} Ha")
    print(f"Binding energy : {binding:.12f} Ha ({binding * HARTREE_TO_KCAL_MOL:.6f} kcal/mol)")
