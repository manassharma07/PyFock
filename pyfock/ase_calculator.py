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

from . import guess_projection
from . import XC
from .Basis import Basis
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

    ``grid_response`` controls whether the analytical XC gradient differentiates the quadrature grid's
    own dependence on the nuclear positions -- the points of an atom translating with it, and the Becke
    weights depending on every nucleus. ``None`` (default) keeps PyFock's convention: off for semilocal
    functionals, on for Skala, where it is mandatory. ``True`` turns it on for any functional, which
    makes the forces exactly translationally invariant (net force ~1e-13 instead of ~1e-4 Ha/Bohr) at
    the cost of one extra pass over the partitioning. It requires the native ('treutler') grids.

    ``dispersion=True`` adds a DFT-D3 correction to the energy and the forces, evaluated here after each
    SCF; ``dispersion_kwargs`` pick the backend and the parametrisation (see
    :meth:`_compute_dispersion_correction`). When they name no parametrisation, the functional's own is
    used, as with ``DFT(..., dispersion=True)``: Skala 1.1 declares D3(BJ) with B3LYP5 parameters, the
    correction it was fitted with, so ``PyFockCalculator(functional='skala-1.1', dispersion=True)`` is
    complete. Other functionals declare none and must name one, e.g. ``dispersion_kwargs={'xc': 'pbe'}``.
    The SCF itself runs without D3, so the correction is never counted twice.

    The converged AO density matrix is checkpointed after each successful
    calculation and used for the initial guess of the next compatible ASE
    geometry. Pass ``reuse_density=False`` to disable this behavior.
    ``density_guess`` selects how the checkpoints become the starting density
    (see :mod:`pyfock.guess_projection`, whose projection and extrapolation
    code was contributed by Prof. Vincenzo Barone):

    - ``'previous'`` (default): the previous converged matrix, unchanged.
    - ``'transfer'``: the previous density carried with the atoms, i.e. its
      occupied natural orbitals re-orthonormalized in the new overlap metric.
    - ``'project'``: the previous density projected into the new basis.
    - ``'extrapolate'``: the densities of the last five geometries carried to the
      new one and extrapolated with weights fitted to the geometries, then made
      N-representable (about 15-18 % fewer SCF iterations than ``'previous'``
      along the same optimization path in our tests).

    ``density_guess_options`` (a dict) is passed on to
    :class:`pyfock.guess_projection.DensityHistory`, e.g. ``{'max_points': 5}``;
    ``{'implementation': 'original'}`` runs Prof. Barone's code exactly as
    contributed (projection and equally-spaced extrapolation) for comparison.

    ASE asks for the energy and the forces in two separate calls, so by default the forces are computed
    together with the energy and the second call is served from the cache; without that every geometry
    would run the SCF twice. Pass ``forces_with_energy=False`` if you only ever want energies.

    Pass ``run_in_process=True`` to run each step in the calling process instead of a subprocess. The
    subprocess is the safer default -- a crash or a non-converged step cannot take the optimizer with it,
    and every step leaves a complete PyFock output file on disk -- but it starts cold each time,
    paying the imports again, re-reading the Skala checkpoint, warming TorchScript up and rebuilding the
    CUDA context. None of that depends on the geometry, so for a geometry optimization it is pure
    repetition: one water step costs about 10 s in a fresh process against 2 s in a warm one. Staying in-process is what makes
    ``skala_gpu=True`` pay off during an optimization, since the model is then loaded once rather than
    once per step. Nothing large is carried between steps either way -- only the converged density matrix
    on disk, as the next SCF's starting guess. Two things to know: BLAS thread counts come from the
    environment at import time, so set ``OMP_NUM_THREADS`` and friends before importing numpy rather than
    relying on ``ncores``, and no per-step output file is written.
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
        "grid_response": None,
        "force_step_size": 1.0e-3,
        "force_step_unit": "bohr",
        "force_method": "central",
        "force_use_fixed_grids": True,
        "reuse_density": True,
        "density_guess": "previous",
        "density_guess_options": None,
        "run_in_process": False,
        "forces_with_energy": True,
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
        grid_response=None,
        force_step_size=1.0e-3,
        force_step_unit="bohr",
        force_method="central",
        force_use_fixed_grids=True,
        reuse_density=True,
        density_guess="previous",
        density_guess_options=None,
        run_in_process=False,
        forces_with_energy=True,
        **kwargs,
    ):
        super().__init__()

        if convergence_check not in ("error", "warning", "ignore"):
            raise ValueError(
                "convergence_check must be 'error', 'warning', or 'ignore'."
            )
        if force_mode not in ("analytical", "numerical"):
            raise ValueError("force_mode must be 'analytical' or 'numerical'.")
        density_guess_options = dict(density_guess_options or {})
        if "method" in density_guess_options:
            raise TypeError("Pass the method as density_guess, not in density_guess_options.")
        # Fail now on unknown methods or options rather than at the second geometry.
        guess_projection.DensityHistory(method=density_guess, **density_guess_options)

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
        # Fail now rather than after the first SCF: the default backend needs a parametrisation, named
        # here or declared by the functional -- which only Skala does, in the checkpoint the SCF loads.
        if dispersion and not XC.is_skala(canonical_options.get("xc")):
            disp_kwargs = self.parameters["dispersion_kwargs"] or {}
            if (self._dispersion_backend(disp_kwargs) == "dftd3"
                    and not self._DISPERSION_NAMES & set(disp_kwargs)):
                raise ValueError(
                    "dispersion=True needs the functional whose D3 parameters to use, e.g. "
                    "dispersion_kwargs={'xc': 'pbe'}. Only Skala declares its own, which is then "
                    "used automatically.")
        self.parameters["force_mode"] = force_mode
        self.parameters["grid_response"] = grid_response
        self.parameters["force_step_size"] = force_step_size
        self.parameters["force_step_unit"] = force_step_unit
        self.parameters["force_method"] = force_method
        self.parameters["force_use_fixed_grids"] = force_use_fixed_grids
        self.parameters["reuse_density"] = bool(reuse_density)
        self.parameters["density_guess"] = density_guess
        self.parameters["density_guess_options"] = density_guess_options
        self.parameters["run_in_process"] = bool(run_in_process)
        self.parameters["forces_with_energy"] = bool(forces_with_energy)

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
        # Converged (structure.xyz, converged_dmat.npy) checkpoints of the latest compatible
        # geometries, oldest first, and the compatibility token they belong to.
        self._density_checkpoints = []
        self._density_checkpoints_token = None

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

    def _density_compatibility_token(self, atoms):
        """Identify geometries whose AO density matrices are compatible."""
        payload = {
            "symbols": atoms.get_chemical_symbols(),
            "pbc": [bool(x) for x in atoms.pbc],
            "charge": self.parameters["charge"],
            "basis": self.parameters["basis"],
            "auxbasis": self.parameters["auxbasis"],
            "options": self.pyfock_options,
        }
        encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _density_history_options(self):
        """Keyword arguments of :class:`pyfock.guess_projection.DensityHistory` for this calculator."""
        options = dict(self.parameters.get("density_guess_options") or {})
        options["method"] = self.parameters.get("density_guess", "previous")
        return options

    def _density_guess_sources(self, atoms):
        """Checkpoints ``[(xyz_path, dmat_path), ...]`` (oldest first) usable for ``atoms``."""
        if not self.parameters.get("reuse_density", True):
            return []
        if self._density_checkpoints_token != self._density_compatibility_token(atoms):
            return []
        return [
            (xyz, dmat)
            for xyz, dmat in self._density_checkpoints
            if os.path.isfile(xyz) and os.path.isfile(dmat)
        ]

    def _density_guess_path(self, atoms):
        """The latest usable density checkpoint, or None."""
        sources = self._density_guess_sources(atoms)
        return sources[-1][1] if sources else None

    def _remember_density_checkpoint(self, atoms, xyz_path, dmat_path):
        token = self._density_compatibility_token(atoms)
        if token != self._density_checkpoints_token:
            self._density_checkpoints = []
            self._density_checkpoints_token = token
        entry = (xyz_path, dmat_path)
        # the same geometry again (same step directory) replaces its old entry
        self._density_checkpoints = [e for e in self._density_checkpoints if e != entry] + [entry]
        keep = guess_projection.DensityHistory(**self._density_history_options()).max_points
        del self._density_checkpoints[:-keep]

    def _to_ev_forces(self, forces_au_bohr):
        factor = Data.au2eVFactor / Data.Bohr2AngsFactor
        return np.asarray(forces_au_bohr, dtype=np.float64) * factor

    def _to_eang_dipole(self, dipole_au):
        return np.asarray(dipole_au, dtype=np.float64) * Data.Bohr2AngsFactor

    # torch-dftd spells the damping function differently from simple-dftd3.
    _DAMPING_ALIASES = {"bj": "d3bj", "zero": "d3zero", "bjm": "d3bjm", "zerom": "d3zerom"}
    # ... and some functionals. simple-dftd3 gives 'b3lyp5' (what Skala 1.1 declares) the B3LYP
    # parameters, which torch-dftd calls 'b3-lyp'; in float64 they agree to 1e-10 Ha on a water dimer.
    _TORCH_DFTD_XC = {"b3lyp5": "b3-lyp"}
    # The dispersion_kwargs that name a parametrisation; without any, the functional's own is used.
    _DISPERSION_NAMES = frozenset(("xc", "method", "param"))

    @staticmethod
    def _dispersion_backend(kwargs):
        """The D3 backend ``dispersion_kwargs`` select: explicitly, or 'torch-dftd' for a non-CPU device."""
        backend = kwargs.get("backend")
        if backend is None:
            device = kwargs.get("device")
            backend = "dftd3" if device is None or str(device) == "cpu" else "torch-dftd"
        return backend

    def _compute_dispersion_correction(self, atoms, compute_forces, declared=None):
        """Dispersion energy (eV) and forces (eV/Angstrom) for ``atoms``.

        Two backends are available. The default, ``'dftd3'``, is simple-dftd3, the Grimme group's
        reference implementation, reached through :mod:`pyfock.Dispersion`; it runs on the CPU, is what
        the rest of PyFock uses, and needs no PyTorch. ``'torch-dftd'`` is the original backend here and
        is worth keeping for GPU runs, where it evaluates the correction on the device alongside a GPU
        SCF. It is selected explicitly with ``backend='torch-dftd'`` or implicitly by asking for a
        non-CPU ``device``.

        ``declared`` is the parametrisation the functional declares, as the SCF reports it -- Skala 1.1
        declares ``'b3lyp5'``, meaning D3(BJ) with those parameters and no three-body term. It is used
        when ``dispersion_kwargs`` name none, translated to torch-dftd's names for that backend.
        """
        kwargs = dict(self.parameters.get("dispersion_kwargs") or {})
        backend = self._dispersion_backend(kwargs)
        kwargs.pop("backend", None)
        if declared is not None and not self._DISPERSION_NAMES & set(kwargs):
            if backend == "torch-dftd":
                # torch-dftd defaults to PBE with zero damping, which is not what Skala was fitted with
                kwargs["xc"] = self._TORCH_DFTD_XC.get(declared, declared)
                kwargs.setdefault("damping", "bj")
            else:
                kwargs["xc"] = declared

        if backend == "dftd3":
            return self._dispersion_dftd3(atoms, compute_forces, kwargs)
        if backend == "torch-dftd":
            return self._dispersion_torch_dftd(atoms, compute_forces, kwargs)
        raise ValueError("Unknown dispersion backend '" + str(backend)
                         + "'. Available: 'dftd3' (default, CPU) and 'torch-dftd' (GPU).")

    def _dispersion_dftd3(self, atoms, compute_forces, kwargs):
        """simple-dftd3 through :mod:`pyfock.Dispersion` (the default backend)."""
        from . import Dispersion

        method = kwargs.pop("xc", None) or kwargs.pop("method", None)
        version = kwargs.pop("version", None) or kwargs.pop("damping", None) or "d3bj"
        version = self._DAMPING_ALIASES.get(str(version).lower(), str(version).lower())
        atm = bool(kwargs.pop("atm", False))
        param = kwargs.pop("param", None)
        kwargs.pop("device", None)  # meaningful only for the torch backend
        if kwargs:
            raise TypeError("Unexpected dispersion_kwargs for the 'dftd3' backend: "
                            + ', '.join(sorted(kwargs)) + ". Supported: xc (or method), damping (or "
                            "version), atm, param, backend.")
        if method is None and param is None:
            raise ValueError("The 'dftd3' dispersion backend needs the functional whose D3 parameters "
                             "to use, e.g. dispersion_kwargs={'xc': 'pbe'}, and this functional does "
                             "not declare one.")

        # Multiply by Angs2BohrFactor rather than dividing by Bohr2AngsFactor: the two constants are
        # not exact reciprocals (they differ in the 12th digit), and Mol uses the former, so this keeps
        # the geometry bit-identical to the one the rest of PyFock would build.
        geometry = (atoms.get_atomic_numbers(),
                    np.asarray(atoms.get_positions(), dtype=np.float64) * Data.Angs2BohrFactor)
        if compute_forces:
            energy_au, gradient_au = Dispersion.d3_energy_and_gradient(
                geometry, method, version=version, atm=atm, param=param)
            # ASE wants forces, which are minus the gradient.
            return energy_au * Data.au2eVFactor, self._to_ev_forces(-gradient_au)
        energy_au = Dispersion.d3_energy(geometry, method, version=version, atm=atm, param=param)
        return energy_au * Data.au2eVFactor, None

    def _dispersion_torch_dftd(self, atoms, compute_forces, kwargs):
        """torch-dftd, kept for GPU runs; ``kwargs`` are passed straight to its calculator."""
        try:
            from torch_dftd.torch_dftd3_calculator import TorchDFTD3Calculator
        except ImportError as exc:
            raise ImportError(
                "The 'torch-dftd' dispersion backend requires the optional 'torch-dftd' package. "
                "Install it with: pip install torch-dftd, or use the default CPU backend "
                "(backend='dftd3', pip install dftd3)."
            ) from exc

        kwargs.setdefault("atoms", atoms.copy())
        disp_atoms = kwargs["atoms"]
        disp_atoms.calc = TorchDFTD3Calculator(**kwargs)

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

    def _write_run_script(
        self,
        atoms,
        workdir,
        task_name,
        compute_forces=False,
        allow_numerical_forces=True,
        compute_dipole=False,
        density_sources=None,
    ):
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

density_sources = {self._render_value([list(source) for source in (density_sources or [])])}
density_guess_used = False
density_guess_info = None
if density_sources:
    from pyfock import guess_projection
    try:
        density_guess, density_history = guess_projection.guess_from_checkpoints(
            mol, basis, density_sources, ncores=ncores, **{self._render_value(self._density_history_options())})
        dft_obj.dmat = density_guess
        dft_obj.grid_pruning_use_core_guess = True
        density_guess_used = True
        density_guess_info = density_history.last_info
        print("Initial guess from earlier ASE steps: "
              + guess_projection.describe_guess(density_guess_info) + ".")
        if density_guess_info["method"] != "previous":
            print(guess_projection.CREDITS)
    except (EOFError, OSError, ValueError, np.linalg.LinAlgError) as exc:
        print(
            "WARNING: Could not build the initial guess from earlier ASE steps: "
            + str(exc)
            + ". Using the configured initial guess instead."
        )

energy_au, dmat = dft_obj.scf()
gap_au, gap_ev = compute_homo_lumo_gap(dft_obj)

if bool(getattr(dft_obj, "converged", False)):
    np.save("converged_dmat.npy", np.asarray(dmat, dtype=np.float64))

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
    "d3_settings": None if getattr(dft_obj, "skala", None) is None else dft_obj.skala.d3_settings(),
    "density_guess_used": density_guess_used,
    "density_guess_source": density_sources[-1][1] if density_guess_used else None,
    "density_guess_info": density_guess_info,
}}

if {self._render_value(compute_forces)}:
    force_mode = {self._render_value(self.parameters["force_mode"])}
    force_results = None
    if force_mode == "analytical":
        try:
            grad_obj = DFT_Grad(dft_obj, grid_response={self._render_value(self.parameters["grid_response"])})
            force_results = grad_obj.calculate()
            result["force_method_used"] = "analytical"
        except (NotImplementedError, ValueError) as exc:
            print("WARNING: Analytical gradients are not available for this "
                  "configuration: " + str(exc))
            print("Falling back to numerical finite-difference forces.")
    if force_results is None and {self._render_value(allow_numerical_forces)}:
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
    if force_results is not None:
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

    def _run_in_process(self, workdir, xyz_path, compute_forces, allow_numerical_forces,
                        density_sources):
        """Run one geometry in this process and return the same summary dict the subprocess returns.

        A fresh subprocess starts cold every step: it repeats the imports, re-reads the Skala checkpoint,
        warms TorchScript up again and builds a new CUDA context, none of which depends on the geometry.
        (PyFock's own Numba kernels are compiled with ``cache=True``, so they reload from disk cheaply.)
        Staying in one process keeps all of it alive across steps -- the Skala model in particular lives
        in a module-level cache, so ``skala_gpu=True`` loads it once instead of once per step. The grids,
        the AO values and the integrals are rebuilt either way; those move with the atoms. Nothing is
        kept alive between geometries: the DFT object goes out of scope when this returns.
        """
        from .DFT_Grad import DFT_Grad
        from .DFT_NumGrad import DFT_NumGrad

        options = self._prepare_runtime_options()
        basis_name = self.parameters["basis"] or self._default_basis_name(self.atoms)

        mol = Mol(coordfile=xyz_path, charge=self.parameters["charge"])
        basis = Basis(mol, {"all": Basis.load(mol=mol, basis_name=basis_name)})
        auxbasis = None
        if options.get("isDF", True):
            auxbasis_name = self.parameters["auxbasis"] or "def2-universal-jfit"
            auxbasis = Basis(mol, {"all": Basis.load(mol=mol, basis_name=auxbasis_name)})

        dft_obj = DFT(mol, basis, auxbasis)
        for key, value in sorted(options.items()):
            setattr(dft_obj, key, value)

        summary = {"density_guess_used": False, "density_guess_source": None, "density_guess_info": None}
        if density_sources:
            try:
                guess, history = guess_projection.guess_from_checkpoints(
                    mol, basis, density_sources, ncores=options.get("ncores", 1),
                    **self._density_history_options())
            except (EOFError, OSError, ValueError, np.linalg.LinAlgError) as exc:
                warnings.warn("Could not build the initial guess from earlier steps (" + str(exc)
                              + "); using the configured initial guess instead.")
            else:
                dft_obj.dmat = guess
                dft_obj.grid_pruning_use_core_guess = True
                summary["density_guess_used"] = True
                summary["density_guess_source"] = density_sources[-1][1]
                summary["density_guess_info"] = history.last_info

        energy_au, dmat = dft_obj.scf()
        converged = bool(getattr(dft_obj, "converged", False))
        if converged:
            np.save(os.path.join(workdir, "converged_dmat.npy"), np.asarray(dmat, dtype=np.float64))

        gap_au = None
        energies = getattr(dft_obj, "mo_energies", None)
        occupations = getattr(dft_obj, "mo_occupations", None)
        if energies is not None and occupations is not None:
            occupied = np.where(np.asarray(occupations) > 1e-8)[0]
            if len(occupied) and occupied[-1] + 1 < len(energies):
                gap_au = float(energies[occupied[-1] + 1] - energies[occupied[-1]])

        summary.update({
            "converged": converged,
            "niter": int(getattr(dft_obj, "niter", 0)),
            "total_energy_au": float(energy_au),
            "total_energy_ev": float(energy_au * Data.au2eVFactor),
            "homo_lumo_gap_au": gap_au,
            "homo_lumo_gap_ev": None if gap_au is None else gap_au * Data.au2eVFactor,
            # the D3 parametrisation the functional declares; the model is only loaded here
            "d3_settings": (None if getattr(dft_obj, "skala", None) is None
                            else dft_obj.skala.d3_settings()),
        })
        for key, attribute in (("xc_energy_au", "XC_energy"),
                               ("coulomb_energy_au", "J_energy"),
                               ("kinetic_energy_au", "Kinetic_energy"),
                               ("electron_nuclear_energy_au", "Nuc_energy"),
                               ("nuclear_repulsion_energy_au", "Nuclear_repulsion_energy")):
            value = getattr(dft_obj, attribute, None)
            summary[key] = None if value is None else float(value)

        self.converged = converged
        mode = self.parameters["convergence_check"]
        if mode != "ignore" and not converged:
            message = "PyFock calculation did not converge (in-process run)."
            if mode == "error":
                raise PyFockConvergenceError(message)
            warnings.warn(message, PyFockConvergenceWarning)

        if compute_forces:
            force_results = None
            if self.parameters["force_mode"] == "analytical":
                try:
                    force_results = DFT_Grad(
                        dft_obj, grid_response=self.parameters["grid_response"]).calculate()
                    summary["force_method_used"] = "analytical"
                except (NotImplementedError, ValueError) as exc:
                    warnings.warn("Analytical gradients are not available for this configuration: "
                                  + str(exc) + ". Falling back to finite differences.")
            if force_results is None and allow_numerical_forces:
                force_results = DFT_NumGrad(
                    dft_obj,
                    step_size=self.parameters["force_step_size"],
                    step_unit=self.parameters["force_step_unit"],
                    method=self.parameters["force_method"],
                    use_fixed_grids=self.parameters["force_use_fixed_grids"],
                    verbose=False,
                ).calculate()
                summary["force_method_used"] = "numerical"
            if force_results is not None:
                summary["forces_au_bohr"] = np.asarray(force_results["forces"]).tolist()

        return summary

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
            "density_guess_used": bool(summary.get("density_guess_used", False)),
            "density_guess_source": summary.get("density_guess_source"),
            "density_guess_info": summary.get("density_guess_info"),
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

        # ASE asks for the energy and the forces in two separate calls. Computing only what was asked
        # for would run the whole SCF twice per geometry, and the forces cost a small fraction of it, so
        # they come along with the energy by default and the second call is served from the cache.
        forces_requested = "forces" in properties
        compute_forces = forces_requested or self.parameters["forces_with_energy"]
        # ... but never let that turn into a numerical gradient behind the user's back: that would be
        # 6N extra SCFs for an energy nobody asked forces for.
        allow_numerical_forces = forces_requested

        density_sources = self._density_guess_sources(self.atoms)
        if self.parameters["run_in_process"]:
            output_path = None
            summary = self._run_in_process(
                step_dir, xyz_path, compute_forces, allow_numerical_forces, density_sources
            )
        else:
            script_path, output_path = self._write_run_script(
                self.atoms,
                step_dir,
                task_name="singlepoint",
                compute_forces=compute_forces,
                allow_numerical_forces=allow_numerical_forces,
                compute_dipole=False,
                density_sources=density_sources,
            )
            summary = self._run_pyfock_script(step_dir, script_path, output_path)
        density_checkpoint_path = os.path.join(step_dir, "converged_dmat.npy")
        if summary.get("converged", False) and os.path.isfile(density_checkpoint_path):
            self._remember_density_checkpoint(self.atoms, xyz_path, density_checkpoint_path)
        self._populate_common_results(summary)

        have_forces = "forces_au_bohr" in summary
        if forces_requested and not have_forces:
            raise RuntimeError(
                "Forces were requested but not produced"
                + ("." if output_path is None else f"; see '{output_path}'.")
            )
        if have_forces:
            self.results["forces"] = self._to_ev_forces(summary["forces_au_bohr"])
            self.pyfock_results["force_method_used"] = summary.get("force_method_used")

        self.pyfock_results["base_energy_ev"] = float(self.results["energy"])
        self.pyfock_results["base_free_energy_ev"] = float(self.results["free_energy"])
        if have_forces:
            self.pyfock_results["base_forces_ev_ang"] = np.asarray(
                self.results["forces"], dtype=np.float64
            ).tolist()

        if self.parameters["dispersion"]:
            disp_energy, disp_forces = self._compute_dispersion_correction(
                self.atoms, have_forces, declared=summary.get("d3_settings")
            )
            self.results["energy"] += disp_energy
            self.results["free_energy"] = self.results["energy"]
            self.pyfock_results["dispersion_energy_ev"] = disp_energy
            self.pyfock_results["total_energy_ev"] = float(self.results["energy"])
            if have_forces:
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
