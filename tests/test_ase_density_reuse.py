from pathlib import Path

import numpy as np
import pytest


pytest.importorskip("ase")

from ase import Atoms

from pyfock import PyFockCalculator


def make_h2():
    return Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])


def test_density_checkpoint_is_reused_after_geometry_change(tmp_path, monkeypatch):
    calc = PyFockCalculator(
        functional="PBE",
        basis="sto-3g",
        directory=str(tmp_path / "calc"),
    )
    atoms = make_h2()
    atoms.calc = calc
    density_guesses = []

    def fake_write_run_script(
        atoms,
        workdir,
        task_name,
        compute_forces=False,
        compute_dipole=False,
        density_guess_path=None,
    ):
        density_guesses.append(density_guess_path)
        return str(Path(workdir) / "run.py"), str(Path(workdir) / "output.txt")

    def fake_run_pyfock_script(workdir, script_path, output_path):
        np.save(Path(workdir) / "converged_dmat.npy", np.eye(2))
        return {
            "converged": True,
            "niter": 1,
            "total_energy_au": -1.0,
            "total_energy_ev": -27.2114,
            "forces_au_bohr": np.zeros((2, 3)).tolist(),
            "force_method_used": "analytical",
            "density_guess_used": density_guesses[-1] is not None,
            "density_guess_source": density_guesses[-1],
        }

    monkeypatch.setattr(calc, "_write_run_script", fake_write_run_script)
    monkeypatch.setattr(calc, "_run_pyfock_script", fake_run_pyfock_script)

    atoms.get_forces()
    first_checkpoint = tmp_path / "calc" / "step_0001" / "converged_dmat.npy"
    atoms.positions[1, 2] += 0.01
    atoms.get_forces()

    assert density_guesses == [None, str(first_checkpoint)]
    assert calc.pyfock_results["density_guess_used"] is True
    assert calc.pyfock_results["density_guess_source"] == str(first_checkpoint)


def test_density_checkpoint_can_be_disabled_or_rejected_as_incompatible(tmp_path):
    calc = PyFockCalculator(
        functional="PBE",
        basis="sto-3g",
        directory=str(tmp_path / "calc"),
    )
    atoms = make_h2()
    checkpoint = tmp_path / "converged_dmat.npy"
    np.save(checkpoint, np.eye(2))
    calc._last_density_path = str(checkpoint)
    calc._last_density_compatibility_token = calc._density_compatibility_token(atoms)

    displaced = atoms.copy()
    displaced.positions[1, 2] += 0.01
    assert calc._density_guess_path(displaced) == str(checkpoint)

    incompatible = Atoms(
        "HeH", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.75]]
    )
    assert calc._density_guess_path(incompatible) is None

    calc.parameters["reuse_density"] = False
    assert calc._density_guess_path(displaced) is None
