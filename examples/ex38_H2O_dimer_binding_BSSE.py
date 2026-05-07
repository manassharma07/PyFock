import os


ncores = 4
os.environ["OMP_NUM_THREADS"] = str(ncores)
os.environ["OPENBLAS_NUM_THREADS"] = str(ncores)
os.environ["MKL_NUM_THREADS"] = str(ncores)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(ncores)
os.environ["NUMEXPR_NUM_THREADS"] = str(ncores)


from pyfock import Basis
from pyfock import DFT
from pyfock import Mol


os.chdir(os.path.abspath(os.path.dirname(__file__)))

HARTREE_TO_KCAL_MOL = 627.5094740631
basis_set_name = "def2-SVP"
auxbasis_name = "def2-universal-jfit"


dimer_xyz_filename = "h2o_dimer.xyz"
dimer_xyz = """6
H2O dimer
O   -1.551007   0.114520   0.000000
H   -1.934259   0.988994   0.000000
H   -0.599677   0.040712   0.000000
O    1.350625  -0.111469   0.000000
H    1.680398  -0.373741  -0.758561
H    1.680398  -0.373741   0.758561
"""
with open(dimer_xyz_filename, "w", encoding="utf-8") as handle:
    handle.write(dimer_xyz)


water1_xyz_filename = "h2o_dimer_water1.xyz"
water1_xyz = """3
Water 1
O   -1.551007   0.114520   0.000000
H   -1.934259   0.988994   0.000000
H   -0.599677   0.040712   0.000000
"""
with open(water1_xyz_filename, "w", encoding="utf-8") as handle:
    handle.write(water1_xyz)


water2_xyz_filename = "h2o_dimer_water2.xyz"
water2_xyz = """3
Water 2
O    1.350625  -0.111469   0.000000
H    1.680398  -0.373741  -0.758561
H    1.680398  -0.373741   0.758561
"""
with open(water2_xyz_filename, "w", encoding="utf-8") as handle:
    handle.write(water2_xyz)


water1_cp_xyz_filename = "h2o_dimer_water1_ghost_water2.xyz"
water1_cp_xyz = """6
Water 1 plus ghost water 2
O        -1.551007   0.114520   0.000000
H        -1.934259   0.988994   0.000000
H        -0.599677   0.040712   0.000000
Ghost-O   1.350625  -0.111469   0.000000
Ghost-H   1.680398  -0.373741  -0.758561
Ghost-H   1.680398  -0.373741   0.758561
"""
with open(water1_cp_xyz_filename, "w", encoding="utf-8") as handle:
    handle.write(water1_cp_xyz)


water2_cp_xyz_filename = "h2o_dimer_water2_ghost_water1.xyz"
water2_cp_xyz = """6
Water 2 plus ghost water 1
Ghost-O  -1.551007   0.114520   0.000000
Ghost-H  -1.934259   0.988994   0.000000
Ghost-H  -0.599677   0.040712   0.000000
O         1.350625  -0.111469   0.000000
H         1.680398  -0.373741  -0.758561
H         1.680398  -0.373741   0.758561
"""
with open(water2_cp_xyz_filename, "w", encoding="utf-8") as handle:
    handle.write(water2_cp_xyz)


def pyfock_energy(xyz_filename):
    mol = Mol(coordfile=xyz_filename)
    basis = Basis(mol, {"all": Basis.load(mol=mol, basis_name=basis_set_name)})
    auxbasis = Basis(mol, {"all": Basis.load(mol=mol, basis_name=auxbasis_name)})

    dft_obj = DFT(mol, basis, auxbasis, xc="r2SCAN", use_pyscf_grids=True, gridsLevel=3)
    dft_obj.conv_crit = 1e-7
    dft_obj.max_itr = 20
    dft_obj.ncores = ncores
    dft_obj.save_ao_values = True
    dft_obj.sao = True
    dft_obj.strict_schwarz = False

    energy, _ = dft_obj.scf()
    return energy


dimer_energy = pyfock_energy(dimer_xyz_filename)
water1_energy = pyfock_energy(water1_xyz_filename)
water2_energy = pyfock_energy(water2_xyz_filename)
water1_cp_energy = pyfock_energy(water1_cp_xyz_filename)
water2_cp_energy = pyfock_energy(water2_cp_xyz_filename)

binding_energy = dimer_energy - water1_energy - water2_energy
binding_energy_cp = dimer_energy - water1_cp_energy - water2_cp_energy
bsse_correction = binding_energy_cp - binding_energy

print("\nH2O dimer binding energy")
print(f"Without CP correction : {binding_energy:.12f} Ha ({binding_energy * HARTREE_TO_KCAL_MOL:.6f} kcal/mol)")
print(f"With CP correction    : {binding_energy_cp:.12f} Ha ({binding_energy_cp * HARTREE_TO_KCAL_MOL:.6f} kcal/mol)")
print(f"BSSE correction       : {bsse_correction:.12f} Ha ({bsse_correction * HARTREE_TO_KCAL_MOL:.6f} kcal/mol)")
