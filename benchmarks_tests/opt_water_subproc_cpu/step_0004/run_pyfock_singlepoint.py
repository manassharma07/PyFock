import json
import os
import numpy as np

from pyfock import Basis
from pyfock import DFT
from pyfock import DFT_Grad
from pyfock import DFT_NumGrad
from pyfock import Integrals
from pyfock import Mol
from pyfock import Data


def compute_homo_lumo_gap(dft_obj):
    eigvalues = getattr(dft_obj, "mo_energies", None)
    occupations = getattr(dft_obj, "mo_occupations", None)
    if eigvalues is None or occupations is None:
        return None, None
    eigvalues = np.asarray(eigvalues)
    occupations = np.asarray(occupations)
    occupied = np.where(occupations > 1e-8)[0]
    if len(occupied) == 0 or occupied[-1] + 1 >= len(eigvalues):
        return None, None
    homo_idx = occupied[-1]
    lumo_idx = homo_idx + 1
    gap_au = float(eigvalues[lumo_idx] - eigvalues[homo_idx])
    gap_ev = float(gap_au * Data.au2eVFactor)
    return gap_au, gap_ev


ncores = 4
if ncores is not None:
    os.environ["OMP_NUM_THREADS"] = str(ncores)
    os.environ["OPENBLAS_NUM_THREADS"] = str(ncores)
    os.environ["MKL_NUM_THREADS"] = str(ncores)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(ncores)
    os.environ["NUMEXPR_NUM_THREADS"] = str(ncores)

mol = Mol(coordfile='structure.xyz', charge=0)
basis = Basis(mol, {"all": Basis.load(mol=mol, basis_name='def2-SVP')})

use_df = True
if use_df:
    auxbasis = Basis(mol, {"all": Basis.load(mol=mol, basis_name='def2-universal-jfit')})
else:
    auxbasis = None

dft_obj = DFT(mol, basis, auxbasis)
dft_obj.conv_crit = 1e-08
dft_obj.ncores = 4
dft_obj.save_ao_values = True
dft_obj.xc = 'skala-1.1'

density_guess_path = 'D:\\pyfock\\benchmarks_tests\\opt_water_subproc_cpu\\step_0003\\converged_dmat.npy'
density_guess_used = False
if density_guess_path is not None:
    try:
        density_guess = np.load(density_guess_path, allow_pickle=False)
        expected_shape = (basis.bfs_nao, basis.bfs_nao)
        if density_guess.shape != expected_shape:
            print(
                "WARNING: Previous density matrix has shape "
                + str(density_guess.shape)
                + "; expected "
                + str(expected_shape)
                + ". Using the configured initial guess instead."
            )
        elif not np.all(np.isfinite(density_guess)):
            print(
                "WARNING: Previous density matrix contains non-finite values. "
                "Using the configured initial guess instead."
            )
        else:
            dft_obj.dmat = np.asarray(density_guess, dtype=np.float64)
            dft_obj.grid_pruning_use_core_guess = True
            density_guess_used = True
            print("Using converged density matrix from the previous ASE step.")
    except (EOFError, OSError, ValueError) as exc:
        print(
            "WARNING: Could not load the previous density matrix: "
            + str(exc)
            + ". Using the configured initial guess instead."
        )

energy_au, dmat = dft_obj.scf()
gap_au, gap_ev = compute_homo_lumo_gap(dft_obj)

if bool(getattr(dft_obj, "converged", False)):
    np.save("converged_dmat.npy", np.asarray(dmat, dtype=np.float64))

result = {
    "converged": bool(getattr(dft_obj, "converged", False)),
    "niter": int(getattr(dft_obj, "niter", 0)),
    "total_energy_au": float(energy_au),
    "total_energy_ev": float(energy_au * Data.au2eVFactor),
    "xc_energy_au": None if getattr(dft_obj, "XC_energy", None) is None else float(dft_obj.XC_energy),
    "coulomb_energy_au": None if getattr(dft_obj, "J_energy", None) is None else float(dft_obj.J_energy),
    "kinetic_energy_au": None if getattr(dft_obj, "Kinetic_energy", None) is None else float(dft_obj.Kinetic_energy),
    "electron_nuclear_energy_au": None if getattr(dft_obj, "Nuc_energy", None) is None else float(dft_obj.Nuc_energy),
    "nuclear_repulsion_energy_au": None if getattr(dft_obj, "Nuclear_repulsion_energy", None) is None else float(dft_obj.Nuclear_repulsion_energy),
    "homo_lumo_gap_au": gap_au,
    "homo_lumo_gap_ev": gap_ev,
    "density_guess_used": density_guess_used,
    "density_guess_source": density_guess_path if density_guess_used else None,
}

if True:
    force_mode = 'analytical'
    force_results = None
    if force_mode == "analytical":
        try:
            grad_obj = DFT_Grad(dft_obj, grid_response=None)
            force_results = grad_obj.calculate()
            result["force_method_used"] = "analytical"
        except (NotImplementedError, ValueError) as exc:
            print("WARNING: Analytical gradients are not available for this "
                  "configuration: " + str(exc))
            print("Falling back to numerical finite-difference forces.")
    if force_results is None and False:
        grad_obj = DFT_NumGrad(
            dft_obj,
            step_size=0.001,
            step_unit='bohr',
            method='central',
            use_fixed_grids=True,
            verbose=False,
        )
        force_results = grad_obj.calculate()
        result["force_method_used"] = "numerical"
    if force_results is not None:
        result["forces_au_bohr"] = np.asarray(force_results["forces"]).tolist()

if False:
    dipole_matrix = Integrals.dipole_moment_mat_symm(basis)
    dipole_au = mol.get_dipole_moment(dipole_matrix, dmat)
    result["dipole_au"] = np.asarray(dipole_au).tolist()

print("PYFOCK_RESULT_JSON=" + json.dumps(result, sort_keys=True))
