from pathlib import Path

import numpy as np
import pytest


pytest.importorskip("ase")

from ase import Atoms

from pyfock import PyFockCalculator


def make_h2():
    return Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])


def _fake_runs(calc, monkeypatch, sources_seen):
    """Replace the subprocess by a stub that records the checkpoints offered to each step."""

    def fake_write_run_script(
        atoms,
        workdir,
        task_name,
        compute_forces=False,
        allow_numerical_forces=True,
        compute_dipole=False,
        density_sources=None,
    ):
        sources_seen.append(list(density_sources or []))
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
            "density_guess_used": bool(sources_seen[-1]),
            "density_guess_source": sources_seen[-1][-1][1] if sources_seen[-1] else None,
        }

    monkeypatch.setattr(calc, "_write_run_script", fake_write_run_script)
    monkeypatch.setattr(calc, "_run_pyfock_script", fake_run_pyfock_script)


def test_density_checkpoint_is_reused_after_geometry_change(tmp_path, monkeypatch):
    calc = PyFockCalculator(
        functional="PBE",
        basis="sto-3g",
        directory=str(tmp_path / "calc"),
    )
    atoms = make_h2()
    atoms.calc = calc
    sources_seen = []
    _fake_runs(calc, monkeypatch, sources_seen)

    atoms.get_forces()
    first = tmp_path / "calc" / "step_0001"
    atoms.positions[1, 2] += 0.01
    atoms.get_forces()

    assert sources_seen == [[], [(str(first / "structure.xyz"), str(first / "converged_dmat.npy"))]]
    assert calc.pyfock_results["density_guess_used"] is True
    assert calc.pyfock_results["density_guess_source"] == str(first / "converged_dmat.npy")


def test_extrapolation_keeps_the_latest_checkpoints(tmp_path, monkeypatch):
    calc = PyFockCalculator(
        functional="PBE",
        basis="sto-3g",
        density_guess="extrapolate",
        density_guess_options={"max_points": 3},
        directory=str(tmp_path / "calc"),
    )
    atoms = make_h2()
    atoms.calc = calc
    sources_seen = []
    _fake_runs(calc, monkeypatch, sources_seen)
    for _ in range(5):
        atoms.get_forces()
        atoms.positions[1, 2] += 0.01

    assert [len(s) for s in sources_seen] == [0, 1, 2, 3, 3]
    steps = [Path(xyz).parent.name for xyz, _ in calc._density_checkpoints]
    assert steps == ["step_0003", "step_0004", "step_0005"]
    assert [Path(xyz).parent.name for xyz, _ in sources_seen[-1]] == ["step_0002", "step_0003", "step_0004"]
    # a geometry computed again (same step directory) replaces its entry instead of adding one
    atoms.positions[1, 2] -= 0.01
    calc._remember_density_checkpoint(atoms, *calc._density_checkpoints[-1])
    assert [Path(xyz).parent.name for xyz, _ in calc._density_checkpoints] == steps


def test_density_checkpoint_can_be_disabled_or_rejected_as_incompatible(tmp_path):
    calc = PyFockCalculator(
        functional="PBE",
        basis="sto-3g",
        directory=str(tmp_path / "calc"),
    )
    atoms = make_h2()
    checkpoint = tmp_path / "converged_dmat.npy"
    structure = tmp_path / "structure.xyz"
    np.save(checkpoint, np.eye(2))
    calc._write_xyz(atoms, str(structure))
    calc._remember_density_checkpoint(atoms, str(structure), str(checkpoint))

    displaced = atoms.copy()
    displaced.positions[1, 2] += 0.01
    assert calc._density_guess_path(displaced) == str(checkpoint)

    incompatible = Atoms(
        "HeH", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.75]]
    )
    assert calc._density_guess_path(incompatible) is None

    calc.parameters["reuse_density"] = False
    assert calc._density_guess_path(displaced) is None


def test_invalid_density_guess_settings_fail_early(tmp_path):
    with pytest.raises(ValueError):
        PyFockCalculator(functional="PBE", density_guess="bogus", directory=str(tmp_path))
    with pytest.raises(TypeError):
        PyFockCalculator(functional="PBE", density_guess="extrapolate",
                         density_guess_options={"bogus": 1}, directory=str(tmp_path))
    with pytest.raises(ValueError):
        PyFockCalculator(functional="PBE", density_guess="transfer",
                         density_guess_options={"implementation": "original"}, directory=str(tmp_path))


@pytest.mark.parametrize("in_process", [True, False])
@pytest.mark.parametrize("options", [None, {"implementation": "original"}])
def test_extrapolated_guess_along_a_path(tmp_path, in_process, options):
    """Three geometries: the third SCF starts from the extrapolation of the first two."""
    atoms = make_h2()
    atoms.calc = PyFockCalculator(
        functional="PBE",
        basis="def2-SVP",
        conv_crit=1e-8,
        density_guess="extrapolate",
        density_guess_options=options,
        run_in_process=in_process,
        directory=str(tmp_path / "calc"),
    )
    energies, infos = [], []
    for _ in range(3):
        energies.append(atoms.get_potential_energy())
        infos.append(atoms.calc.pyfock_results["density_guess_info"])
        atoms.positions[1, 2] += 0.02
    assert infos[0] is None and infos[1]["npoints"] == 1 and infos[2]["npoints"] == 2
    expected_weights = [-1.0, 2.0]
    np.testing.assert_allclose(infos[2]["weights"], expected_weights, atol=1e-8)
    assert infos[2]["implementation"] == ("original" if options else "pyfock")

    # same energy as a calculation that starts from scratch
    fresh = make_h2()
    fresh.positions[1, 2] += 0.04
    fresh.calc = PyFockCalculator(functional="PBE", basis="def2-SVP", conv_crit=1e-8,
                                  run_in_process=True, directory=str(tmp_path / "fresh"))
    assert energies[2] == pytest.approx(fresh.get_potential_energy(), abs=1e-5)


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
