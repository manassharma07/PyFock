"""
ASE calculator for PyFock.

This module provides a subprocess-backed ASE calculator that writes a
standard PyFock API script into a working directory, executes it in the
background, and parses results from the saved output file.
"""

import hashlib
import json
import os
import re
import subprocess
import sys
import warnings
from pathlib import Path
from pprint import pformat

import numpy as np
from ase.calculators.calculator import Calculator, all_changes

from .DFT import DFT
from .Data import Data
from .Mol import Mol


class PyFockConvergenceError(RuntimeError):
    """Raised when a PyFock SCF calculation does not converge."""


class PyFockConvergenceWarning(UserWarning):
    """Warning issued when a PyFock SCF calculation does not converge."""


def check_convergence(output_file):
    """
    Check whether the PyFock calculation converged from its output file.

    Returns
    -------
    tuple
        ``(converged, message)`` where ``message`` is a short matching line if
        convergence failed.
    """

    failure_patterns = [
        "SCF NOT Converged",
        "NOT CONVERGED",
        "did not converge",
        "convergence not achieved",
        "ERROR:",
    ]
    success_patterns = [
        "SCF Converged after",
        "SCF CONVERGED",
    ]

    try:
        with open(output_file, "r", encoding="utf-8") as handle:
            content = handle.read()
    except FileNotFoundError:
        return False, f"Output file '{output_file}' not found"

    content_lower = content.lower()
    for pattern in failure_patterns:
        if pattern.lower() in content_lower:
            for line in content.splitlines():
                if pattern.lower() in line.lower():
                    return False, line.strip()
            return False, pattern

    for pattern in success_patterns:
        if pattern.lower() in content_lower:
            return True, None

    return False, "Could not determine convergence from PyFock output"


def _parse_result_marker(output_file):
    marker = "PYFOCK_RESULT_JSON="
    with open(output_file, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith(marker):
                return json.loads(line[len(marker) :].strip())
    raise RuntimeError(
        f"Could not find '{marker}' marker in '{output_file}'. "
        "Check the output file for PyFock errors."
    )


class PyFockCalculator(Calculator):
    """
    ASE calculator wrapper for PyFock DFT/HF calculations.

    The calculator writes a ``run_pyfock.py`` script using PyFock's normal
    Python API, runs it as a subprocess, saves the full PyFock stdout to a
    text file, and parses a final JSON marker from that output.

    Forces are computed analytically by default (``force_mode="analytical"``,
    using :class:`pyfock.DFT_Grad`), which supports LDA, GGA and meta-GGA
    functionals (native or pylibxc) with density fitting, including ECPs. If
    the analytical gradients do not support the requested configuration (e.g.
    HF, no density fitting, or GPU), the calculation automatically falls back
    to finite-difference forces and notes this in ``pyfock_results``. Pass
    ``force_mode="numerical"`` to explicitly request finite-difference
    forces; the ``force_step_size``/``force_step_unit``/``force_method``/
    ``force_use_fixed_grids`` parameters apply to the numerical path only.
    """

    implemented_properties = ["energy", "forces"]
    default_parameters = {
        "basis": None,
        "auxbasis": None,
        "charge": 0,
        "convergence_check": "error",
        "dispersion": False,
        "dispersion_kwargs": None,
        "force_mode": "analytical",
        "force_step_size": 1.0e-3,
        "force_step_unit": "bohr",
        "force_method": "central",
        "force_use_fixed_grids": True,
    }
    _cached_dft_attr_names = None

    def __init__(
        self,
        basis=None,
        auxbasis=None,
        charge=0,
        directory="pyfock_calc",
        convergence_check="error",
        dispersion=False,
        dispersion_kwargs=None,
        force_mode="analytical",
        force_step_size=1.0e-3,
        force_step_unit="bohr",
        force_method="central",
        force_use_fixed_grids=True,
        **kwargs,
    ):
        super().__init__()

        if convergence_check not in ("error", "warning", "ignore"):
            raise ValueError(
                "convergence_check must be 'error', 'warning', or 'ignore'."
            )
        if force_mode not in ("analytical", "numerical"):
            raise ValueError("force_mode must be 'analytical' or 'numerical'.")

        canonical_options = self._canonicalize_options(kwargs)
        self._validate_option_names(canonical_options)

        self.parameters.update(self.default_parameters)
        self.parameters["basis"] = basis
        self.parameters["auxbasis"] = auxbasis
        self.parameters["charge"] = charge
        self.parameters["convergence_check"] = convergence_check
        self.parameters["dispersion"] = dispersion
        self.parameters["dispersion_kwargs"] = (
            None if dispersion_kwargs is None else dict(dispersion_kwargs)
        )
        self.parameters["force_mode"] = force_mode
        self.parameters["force_step_size"] = force_step_size
        self.parameters["force_step_unit"] = force_step_unit
        self.parameters["force_method"] = force_method
        self.parameters["force_use_fixed_grids"] = force_use_fixed_grids

        self.directory = os.path.abspath(directory)
        self.pyfock_options = canonical_options
        self.pyfock_results = {}
        self.converged = None
        self._iteration = 0
        self._last_energy_token = None
        self._last_dipole_token = None
        self._last_homo_lumo_gap_ev = None
        self._last_homo_lumo_gap_au = None
        self._last_dipole_eang = None
        self._last_step_dir = None
        self._last_step_token = None

    @classmethod
    def _dft_attribute_names(cls):
        if cls._cached_dft_attr_names is None:
            mol = Mol(atoms=[["H", 0.0, 0.0, 0.0], ["H", 0.0, 0.0, 0.74]])
            dft_obj = DFT(mol, mol.basis, xc="PBE")
            cls._cached_dft_attr_names = set(dft_obj.__dict__.keys())
        return cls._cached_dft_attr_names

    def _canonicalize_options(self, kwargs):
        options = dict(kwargs)
        if "functional" in options:
            if "xc" in options:
                raise TypeError("Use either 'functional' or 'xc', not both.")
            options["xc"] = options.pop("functional")
        if "DF" in options:
            if "isDF" in options:
                raise TypeError("Use either 'DF' or 'isDF', not both.")
            options["isDF"] = options.pop("DF")
        return options

    def _validate_option_names(self, options):
        allowed = self._dft_attribute_names()
        unknown = sorted(set(options) - allowed)
        if unknown:
            raise TypeError(
                "Unknown PyFock calculator option(s): " + ", ".join(unknown)
            )

    def _default_basis_name(self, atoms):
        if not np.any(atoms.pbc):
            return "def2-SVP"
        return "def2-SVP"

    def _state_token(self, atoms):
        payload = {
            "symbols": atoms.get_chemical_symbols(),
            "positions": np.asarray(atoms.get_positions(), dtype=np.float64).round(12).tolist(),
            "cell": np.asarray(atoms.get_cell().array, dtype=np.float64).round(12).tolist(),
            "pbc": [bool(x) for x in atoms.pbc],
            "charge": self.parameters["charge"],
            "basis": self.parameters["basis"],
            "auxbasis": self.parameters["auxbasis"],
            "options": self.pyfock_options,
        }
        encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _to_ev_forces(self, forces_au_bohr):
        factor = Data.au2eVFactor / Data.Bohr2AngsFactor
        return np.asarray(forces_au_bohr, dtype=np.float64) * factor

    def _to_eang_dipole(self, dipole_au):
        return np.asarray(dipole_au, dtype=np.float64) * Data.Bohr2AngsFactor

    def _compute_dispersion_correction(self, atoms, compute_forces):
        try:
            from torch_dftd.torch_dftd3_calculator import TorchDFTD3Calculator
        except ImportError as exc:
            raise ImportError(
                "Dispersion correction requires the optional 'torch-dftd' package. "
                "Install it with: pip install torch-dftd"
            ) from exc

        dispersion_kwargs = self.parameters.get("dispersion_kwargs")
        if dispersion_kwargs is None:
            dispersion_kwargs = {}
        else:
            dispersion_kwargs = dict(dispersion_kwargs)

        dispersion_kwargs.setdefault("atoms", atoms.copy())
        disp_atoms = dispersion_kwargs["atoms"]
        disp_calc = TorchDFTD3Calculator(**dispersion_kwargs)
        disp_atoms.calc = disp_calc

        disp_energy = float(disp_atoms.get_potential_energy())
        disp_forces = None
        if compute_forces:
            disp_forces = np.asarray(disp_atoms.get_forces(), dtype=np.float64)
        return disp_energy, disp_forces

    def _next_step_dir(self):
        self._iteration += 1
        step_dir = os.path.join(self.directory, f"step_{self._iteration:04d}")
        os.makedirs(step_dir, exist_ok=True)
        self._last_step_dir = step_dir
        return step_dir

    def _get_workdir_for_state(self, atoms):
        state_token = self._state_token(atoms)
        if self._last_step_token == state_token and self._last_step_dir is not None:
            os.makedirs(self._last_step_dir, exist_ok=True)
            return state_token, self._last_step_dir

        step_dir = self._next_step_dir()
        self._last_step_token = state_token
        return state_token, step_dir

    def _write_xyz(self, atoms, filepath):
        positions = atoms.get_positions()
        with open(filepath, "w", encoding="utf-8") as handle:
            handle.write(f"{len(atoms)}\n")
            handle.write("Generated by PyFockCalculator\n")
            for symbol, coord in zip(atoms.get_chemical_symbols(), positions):
                handle.write(
                    f"{symbol} {coord[0]:.16f} {coord[1]:.16f} {coord[2]:.16f}\n"
                )

    def _render_value(self, value):
        return pformat(value, sort_dicts=False)

    def _prepare_runtime_options(self):
        options = dict(self.pyfock_options)
        xc_value = options.get("xc")
        user_set_df = "isDF" in options
        user_set_rys = "rys" in options
        user_set_direct_scf = "direct_scf" in options

        if xc_value == "HF":
            if user_set_df and options.get("isDF", True):
                raise ValueError("PyFock HF through the DFT module requires DF=False.")
            if not user_set_df:
                options["isDF"] = False
            if user_set_rys and options.get("rys", True):
                raise ValueError("PyFock HF currently requires rys=False.")
            if not user_set_rys:
                options["rys"] = False
            if user_set_direct_scf and not options.get("direct_scf", False):
                raise ValueError("PyFock HF currently requires direct_scf=True.")
            if not user_set_direct_scf:
                options["direct_scf"] = True

        return options

    def _write_run_script(self, atoms, workdir, task_name, compute_forces=False, compute_dipole=False):
        options = self._prepare_runtime_options()
        basis_name = self.parameters["basis"] or self._default_basis_name(atoms)
        auxbasis_name = self.parameters["auxbasis"] or "def2-universal-jfit"
        xyz_filename = "structure.xyz"
        output_filename = f"output_pyfock_{task_name}.txt"
        script_path = os.path.join(workdir, f"run_pyfock_{task_name}.py")

        option_lines = []
        for key, value in sorted(options.items()):
            option_lines.append(f"dft_obj.{key} = {self._render_value(value)}")
        option_block = "\n".join(option_lines)

        script = f"""import json
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


ncores = {self._render_value(options.get("ncores", 1))}
if ncores is not None:
    os.environ["OMP_NUM_THREADS"] = str(ncores)
    os.environ["OPENBLAS_NUM_THREADS"] = str(ncores)
    os.environ["MKL_NUM_THREADS"] = str(ncores)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(ncores)
    os.environ["NUMEXPR_NUM_THREADS"] = str(ncores)

mol = Mol(coordfile={self._render_value(xyz_filename)}, charge={self._render_value(self.parameters["charge"])})
basis = Basis(mol, {{"all": Basis.load(mol=mol, basis_name={self._render_value(basis_name)})}})

use_df = {self._render_value(options.get("isDF", True))}
if use_df:
    auxbasis = Basis(mol, {{"all": Basis.load(mol=mol, basis_name={self._render_value(auxbasis_name)})}})
else:
    auxbasis = None

dft_obj = DFT(mol, basis, auxbasis)
{option_block}

energy_au, dmat = dft_obj.scf()
gap_au, gap_ev = compute_homo_lumo_gap(dft_obj)

result = {{
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
}}

if {self._render_value(compute_forces)}:
    force_mode = {self._render_value(self.parameters["force_mode"])}
    force_results = None
    if force_mode == "analytical":
        try:
            grad_obj = DFT_Grad(dft_obj)
            force_results = grad_obj.calculate()
            result["force_method_used"] = "analytical"
        except (NotImplementedError, ValueError) as exc:
            print("WARNING: Analytical gradients are not available for this "
                  "configuration: " + str(exc))
            print("Falling back to numerical finite-difference forces.")
    if force_results is None:
        grad_obj = DFT_NumGrad(
            dft_obj,
            step_size={self._render_value(self.parameters["force_step_size"])},
            step_unit={self._render_value(self.parameters["force_step_unit"])},
            method={self._render_value(self.parameters["force_method"])},
            use_fixed_grids={self._render_value(self.parameters["force_use_fixed_grids"])},
            verbose=False,
        )
        force_results = grad_obj.calculate()
        result["force_method_used"] = "numerical"
    result["forces_au_bohr"] = np.asarray(force_results["forces"]).tolist()

if {self._render_value(compute_dipole)}:
    dipole_matrix = Integrals.dipole_moment_mat_symm(basis)
    dipole_au = mol.get_dipole_moment(dipole_matrix, dmat)
    result["dipole_au"] = np.asarray(dipole_au).tolist()

print("PYFOCK_RESULT_JSON=" + json.dumps(result, sort_keys=True))
"""

        with open(script_path, "w", encoding="utf-8") as handle:
            handle.write(script)

        return script_path, os.path.join(workdir, output_filename)

    def _run_pyfock_script(self, workdir, script_path, output_path):
        with open(output_path, "w", encoding="utf-8") as output_handle:
            result = subprocess.run(
                [sys.executable, os.path.basename(script_path)],
                cwd=workdir,
                stdout=output_handle,
                stderr=subprocess.STDOUT,
            )

        if result.returncode != 0:
            raise RuntimeError(
                f"PyFock subprocess failed with exit code {result.returncode}. "
                f"Check '{output_path}' for details."
            )

        convergence_mode = self.parameters["convergence_check"]
        converged, message = check_convergence(output_path)
        self.converged = converged
        if convergence_mode != "ignore" and not converged:
            error_message = (
                f"PyFock calculation did not converge. Details: {message}. "
                f"Check '{output_path}' for more information."
            )
            if convergence_mode == "error":
                raise PyFockConvergenceError(error_message)
            warnings.warn(error_message, PyFockConvergenceWarning)

        summary = _parse_result_marker(output_path)
        if convergence_mode == "ignore":
            self.converged = bool(summary.get("converged", True))
        return summary

    def _populate_common_results(self, summary):
        self.results["energy"] = float(summary["total_energy_ev"])
        self.results["free_energy"] = self.results["energy"]

        self.pyfock_results = {
            "converged": bool(summary.get("converged", False)),
            "niter": int(summary.get("niter", 0)),
            "total_energy_au": summary.get("total_energy_au"),
            "total_energy_ev": summary.get("total_energy_ev"),
            "xc_energy_au": summary.get("xc_energy_au"),
            "coulomb_energy_au": summary.get("coulomb_energy_au"),
            "kinetic_energy_au": summary.get("kinetic_energy_au"),
            "electron_nuclear_energy_au": summary.get("electron_nuclear_energy_au"),
            "nuclear_repulsion_energy_au": summary.get("nuclear_repulsion_energy_au"),
            "homo_lumo_gap_au": summary.get("homo_lumo_gap_au"),
            "homo_lumo_gap_ev": summary.get("homo_lumo_gap_ev"),
            "dispersion_enabled": bool(self.parameters.get("dispersion", False)),
        }
        self._last_homo_lumo_gap_au = summary.get("homo_lumo_gap_au")
        self._last_homo_lumo_gap_ev = summary.get("homo_lumo_gap_ev")

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        if properties is None:
            properties = self.implemented_properties

        super().calculate(atoms, properties, system_changes)

        os.makedirs(self.directory, exist_ok=True)
        state_token, step_dir = self._get_workdir_for_state(self.atoms)
        xyz_path = os.path.join(step_dir, "structure.xyz")
        self._write_xyz(self.atoms, xyz_path)

        compute_forces = "forces" in properties
        script_path, output_path = self._write_run_script(
            self.atoms,
            step_dir,
            task_name="singlepoint",
            compute_forces=compute_forces,
            compute_dipole=False,
        )
        summary = self._run_pyfock_script(step_dir, script_path, output_path)
        self._populate_common_results(summary)

        if compute_forces:
            if "forces_au_bohr" not in summary:
                raise RuntimeError(
                    f"Forces were requested but not found in '{output_path}'."
                )
            self.results["forces"] = self._to_ev_forces(summary["forces_au_bohr"])
            self.pyfock_results["force_method_used"] = summary.get("force_method_used")

        self.pyfock_results["base_energy_ev"] = float(self.results["energy"])
        self.pyfock_results["base_free_energy_ev"] = float(self.results["free_energy"])
        if compute_forces:
            self.pyfock_results["base_forces_ev_ang"] = np.asarray(
                self.results["forces"], dtype=np.float64
            ).tolist()

        if self.parameters["dispersion"]:
            disp_energy, disp_forces = self._compute_dispersion_correction(
                self.atoms, compute_forces
            )
            self.results["energy"] += disp_energy
            self.results["free_energy"] = self.results["energy"]
            self.pyfock_results["dispersion_energy_ev"] = disp_energy
            self.pyfock_results["total_energy_ev"] = float(self.results["energy"])
            if compute_forces:
                self.results["forces"] = self.results["forces"] + disp_forces
                self.pyfock_results["dispersion_forces_ev_ang"] = disp_forces.tolist()
                self.pyfock_results["total_forces_ev_ang"] = np.asarray(
                    self.results["forces"], dtype=np.float64
                ).tolist()

        self._last_energy_token = state_token
        self._last_dipole_token = None
        self._last_dipole_eang = None

    def get_potential_energy(self, atoms=None, force_consistent=False):
        return self.get_property("energy", atoms)

    def get_forces(self, atoms=None):
        return self.get_property("forces", atoms)

    def get_dipole_moment(self, atoms=None):
        if atoms is None:
            atoms = self.atoms

        state_token = self._state_token(atoms)
        if self._last_energy_token != state_token or self._last_step_dir is None:
            raise RuntimeError(
                "Dipole moment is available only after get_potential_energy() "
                "has been called for the current structure."
            )

        if self._last_dipole_token == state_token and self._last_dipole_eang is not None:
            return self._last_dipole_eang.copy()

        xyz_path = os.path.join(self._last_step_dir, "structure.xyz")
        if not os.path.exists(xyz_path):
            self._write_xyz(atoms, xyz_path)

        script_path, output_path = self._write_run_script(
            atoms,
            self._last_step_dir,
            task_name="dipole",
            compute_forces=False,
            compute_dipole=True,
        )
        summary = self._run_pyfock_script(self._last_step_dir, script_path, output_path)
        dipole_au = summary.get("dipole_au")
        if dipole_au is None:
            raise RuntimeError(
                f"Dipole moment was requested but not found in '{output_path}'."
            )

        dipole_eang = self._to_eang_dipole(dipole_au)
        self._last_dipole_eang = dipole_eang
        self._last_dipole_token = state_token
        self.pyfock_results["dipole_au"] = dipole_au
        self.pyfock_results["dipole_eang"] = dipole_eang.tolist()
        return dipole_eang.copy()

    def get_homo_lumo_gap(self, atoms=None, unit="eV"):
        if atoms is None:
            atoms = self.atoms

        state_token = self._state_token(atoms)
        if self._last_energy_token != state_token:
            raise RuntimeError(
                "HOMO-LUMO gap is available only after get_potential_energy() "
                "has been called for the current structure."
            )

        if unit.lower() == "ev":
            return self._last_homo_lumo_gap_ev
        if unit.lower() == "au":
            return self._last_homo_lumo_gap_au
        raise ValueError("unit must be 'eV' or 'au'.")
