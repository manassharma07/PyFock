"""Tests of the DFT-D3 dispersion correction (:mod:`pyfock.Dispersion`).

D3 is a purely geometric, additive term, which makes it cheap to test but also easy to get subtly wrong:
the three-body ATM contribution is controlled by two different mechanisms depending on whether the
damping parameters are looked up by functional name or given explicitly, and the ASE calculator wants
forces rather than a gradient. Both are pinned here.
"""
from __future__ import annotations

import contextlib
import io

import numpy as np
import pytest

from pyfock import Data, Dispersion, Mol


pytest.importorskip('dftd3', reason='the D3 correction needs simple-dftd3 (pip install dftd3)')

WATER_DIMER = [
    ['O', -1.551007, 0.114520, 0.000000], ['H', -1.934259, 0.988994, 0.000000],
    ['H', -0.599677, 0.040712, 0.000000], ['O', 1.350625, -0.111469, 0.000000],
    ['H', 1.680398, -0.373741, -0.758561], ['H', 1.680398, -0.373741, 0.758561],
]

# PBE's published D3(BJ) damping parameters.
PBE_D3BJ = {'s6': 1.0, 's8': 0.7875, 'a1': 0.4289, 'a2': 4.4407}


@pytest.fixture(scope='module')
def dimer():
    return Mol(atoms=[list(atom) for atom in WATER_DIMER])


def test_energy_is_attractive(dimer):
    assert Dispersion.d3_energy(dimer, 'pbe') < 0.0


def test_damping_and_functional_change_the_answer(dimer):
    """The parametrisation actually reaches the library rather than being silently ignored."""
    pbe_bj = Dispersion.d3_energy(dimer, 'pbe')
    assert Dispersion.d3_energy(dimer, 'b3lyp') != pytest.approx(pbe_bj, rel=1e-3)
    assert Dispersion.d3_energy(dimer, 'pbe', version='d3zero') != pytest.approx(pbe_bj, rel=1e-3)


def test_explicit_parameters_reproduce_the_lookup(dimer):
    """Explicit damping parameters must go down the same path as looking them up by name.

    ``simple-dftd3`` uses two different constructors here: a name lookup takes an ``atm`` flag, while
    explicit parameters take ``s9`` and default it to 1.0 -- that is, ATM *on*. Without translating
    between the two, ``atm=False`` would be silently ignored whenever parameters are passed by hand.
    """
    for atm in (False, True):
        by_name = Dispersion.d3_energy(dimer, 'pbe', atm=atm)
        explicit = Dispersion.d3_energy(dimer, None, param=PBE_D3BJ, atm=atm)
        assert explicit == pytest.approx(by_name, rel=1e-12, abs=1e-14)


def test_atm_changes_the_energy(dimer):
    assert (Dispersion.d3_energy(dimer, 'pbe', atm=True)
            != pytest.approx(Dispersion.d3_energy(dimer, 'pbe', atm=False), rel=1e-6))


def test_gradient_matches_finite_differences(dimer):
    """dE_disp/dR from the library against a central difference of the energy."""
    energy, gradient = Dispersion.d3_energy_and_gradient(dimer, 'pbe')
    assert energy == pytest.approx(Dispersion.d3_energy(dimer, 'pbe'), rel=1e-12)
    assert gradient.shape == (len(WATER_DIMER), 3)

    numbers = np.asarray(dimer.Zcharges)
    positions = np.asarray(dimer.coordsBohrs, dtype=np.float64).reshape(-1, 3)
    step = 1e-5
    for atom, axis in ((0, 2), (3, 0), (5, 1)):
        shifted = positions.copy()
        shifted[atom, axis] += step
        plus = Dispersion.d3_energy((numbers, shifted), 'pbe')
        shifted[atom, axis] -= 2 * step
        minus = Dispersion.d3_energy((numbers, shifted), 'pbe')
        assert (plus - minus) / (2 * step) == pytest.approx(gradient[atom, axis], rel=1e-5, abs=1e-12)


def test_bare_geometry_matches_mol(dimer):
    """The ``(numbers, positions)`` entry point the ASE calculator uses is the same calculation."""
    geometry = (np.asarray(dimer.Zcharges), np.asarray(dimer.coordsBohrs))
    assert Dispersion.d3_energy(geometry, 'pbe') == Dispersion.d3_energy(dimer, 'pbe')


def test_unknown_damping_is_rejected(dimer):
    with pytest.raises(ValueError, match='Unknown D3 damping'):
        Dispersion.d3_energy(dimer, 'pbe', version='d3-not-a-thing')


def test_scf_total_energy_includes_the_correction(dimer):
    """``DFT(dispersion=...)`` must add exactly the standalone correction and nothing else."""
    from pyfock import Basis, DFT
    basis = Basis(dimer, {'all': Basis.load(mol=dimer, basis_name='sto-3g')})
    auxbasis = Basis(dimer, {'all': Basis.load(mol=dimer, basis_name='def2-universal-jfit')})

    energies = {}
    for dispersion in (None, 'pbe'):
        dft = DFT(dimer, basis, auxbasis, xc='PBE', dispersion=dispersion)
        dft.conv_crit = 1e-8
        with contextlib.redirect_stdout(io.StringIO()):
            energy, _ = dft.scf()
        assert dft.converged
        energies[dispersion] = (float(energy), dft.Edisp)

    assert energies[None][1] == 0.0
    assert energies['pbe'][1] == pytest.approx(Dispersion.d3_energy(dimer, 'pbe'), rel=1e-12)
    assert energies['pbe'][0] - energies[None][0] == pytest.approx(energies['pbe'][1], abs=1e-9)


def test_ase_calculator_defaults_to_dftd3(dimer):
    """The ASE path uses the same backend and returns forces (eV/Angstrom), not a gradient."""
    pytest.importorskip('ase', reason='the ASE calculator needs ASE')
    from ase import Atoms
    from pyfock.ase_calculator import PyFockCalculator

    atoms = Atoms(symbols=[a[0] for a in WATER_DIMER],
                  positions=[a[1:] for a in WATER_DIMER])
    calc = PyFockCalculator(functional='PBE', basis='sto-3g', dispersion=True,
                            dispersion_kwargs={'xc': 'pbe'})
    energy_ev, forces_ev = calc._compute_dispersion_correction(atoms, compute_forces=True)

    energy_au, gradient_au = Dispersion.d3_energy_and_gradient(dimer, 'pbe')
    assert energy_ev == pytest.approx(energy_au * Data.au2eVFactor, rel=1e-12)
    factor = Data.au2eVFactor / Data.Bohr2AngsFactor
    assert np.allclose(forces_ev, -gradient_au * factor, rtol=1e-12, atol=1e-14)


def test_ase_calculator_rejects_unknown_backend(dimer):
    pytest.importorskip('ase', reason='the ASE calculator needs ASE')
    from ase import Atoms
    from pyfock.ase_calculator import PyFockCalculator

    atoms = Atoms(symbols=[a[0] for a in WATER_DIMER], positions=[a[1:] for a in WATER_DIMER])
    calc = PyFockCalculator(functional='PBE', basis='sto-3g', dispersion=True,
                            dispersion_kwargs={'xc': 'pbe', 'backend': 'nope'})
    with pytest.raises(ValueError, match='Unknown dispersion backend'):
        calc._compute_dispersion_correction(atoms, compute_forces=False)
