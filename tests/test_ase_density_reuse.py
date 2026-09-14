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
        allow_numerical_forces=True,
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


def test_run_in_process_matches_the_subprocess(tmp_path):
    """The in-process path must reproduce what the subprocess path computes.

    Not bit for bit: the subprocess sets the BLAS thread counts from ``ncores`` before numpy is
    imported and the in-process run cannot, so summation orders differ and the two converged densities
    sit about ``conv_crit`` apart. The tolerances below are far tighter than anything that matters
    physically but loose enough not to measure that noise.
    """
    results = {}
    for in_process in (False, True):
        atoms = make_h2()
        atoms.calc = PyFockCalculator(
            functional="PBE",
            basis="def2-SVP",
            conv_crit=1e-8,
            run_in_process=in_process,
            directory=str(tmp_path / ("inproc" if in_process else "subproc")),
        )
        results[in_process] = (atoms.get_potential_energy(), atoms.get_forces())

    assert results[True][0] == pytest.approx(results[False][0], abs=1e-7)
    np.testing.assert_allclose(results[True][1], results[False][1], atol=1e-5)


def test_one_scf_per_geometry_in_process(tmp_path, monkeypatch):
    """Energy and forces at the same geometry must not repeat the SCF."""
    from pyfock import DFT

    scf_calls = []
    original_scf = DFT.scf
    monkeypatch.setattr(
        DFT, "scf", lambda self, *a, **k: (scf_calls.append(1), original_scf(self, *a, **k))[1]
    )

    atoms = make_h2()
    atoms.calc = PyFockCalculator(
        functional="PBE",
        basis="def2-SVP",
        run_in_process=True,
        directory=str(tmp_path / "calc"),
    )
    atoms.get_potential_energy()
    atoms.get_forces()
    assert len(scf_calls) == 1

    atoms.positions[1, 2] += 0.01
    atoms.get_forces()
    assert len(scf_calls) == 2


def test_one_subprocess_per_geometry(tmp_path, monkeypatch):
    """The same must hold for the default subprocess path, which cannot cache anything in memory."""
    runs = []
    original = PyFockCalculator._run_pyfock_script
    monkeypatch.setattr(
        PyFockCalculator,
        "_run_pyfock_script",
        lambda self, *a, **k: (runs.append(1), original(self, *a, **k))[1],
    )

    atoms = make_h2()
    atoms.calc = PyFockCalculator(
        functional="PBE", basis="def2-SVP", directory=str(tmp_path / "calc")
    )
    atoms.get_potential_energy()
    atoms.get_forces()
    assert len(runs) == 1


def test_forces_with_energy_can_be_turned_off(tmp_path, monkeypatch):
    """Opting out restores the old behaviour: an energy-only call computes no forces."""
    atoms = make_h2()
    atoms.calc = PyFockCalculator(
        functional="PBE",
        basis="def2-SVP",
        run_in_process=True,
        forces_with_energy=False,
        directory=str(tmp_path / "calc"),
    )
    atoms.get_potential_energy()
    assert "forces" not in atoms.calc.results
