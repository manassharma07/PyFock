"""Tests of the Skala neural exchange-correlation functional (``pyfock.XC.skala_iface`` and
``pyfock.Integrals.eval_xc_skala``).

The load-bearing test here is :func:`test_potential_is_the_energy_derivative`. Skala's potential comes
out of automatic differentiation rather than from closed-form derivatives, and the conversion between
the model's spin-resolved convention and PyFock's total-density one involves several factors of two that
would silently produce a wrong-but-plausible SCF. A directional finite difference of the energy pins all
of them at once.
"""
from __future__ import annotations

import contextlib
import io

from pathlib import Path

import numpy as np
import pytest

from pyfock import (Basis, Data, DFT, DFT_Grad, DFT_NumGrad, Grids, Integrals,
                    Mol, XC)


torch = pytest.importorskip('torch', reason='Skala needs PyTorch')

pytestmark = pytest.mark.filterwarnings('ignore::UserWarning')

H2_XYZ = Path(__file__).resolve().parents[1] / 'examples' / 'H2.xyz'

H2O = [['O', 0.0, 0.0, 0.1173], ['H', 0.0, 0.7572, -0.4692], ['H', 0.0, -0.7572, -0.4692]]


@pytest.fixture(scope='module')
def system():
    """A small closed-shell system with a coarse native grid and a physical-ish density matrix."""
    mol = Mol(atoms=[list(atom) for atom in H2O])
    basis = Basis(mol, {'all': Basis.load(mol=mol, basis_name='def2-SVP')})
    grids = Grids(mol, level=1, verbose=False)

    # A symmetric, positive-semidefinite density matrix normalised to the right electron count; the
    # derivative tests do not need it to be the SCF solution, only smooth and physically scaled.
    rng = np.random.default_rng(0)
    a = rng.standard_normal((basis.bfs_nao, basis.bfs_nao))
    dmat = a @ a.T
    overlap = Integrals.overlap_mat_symm(basis)
    dmat *= mol.nelectrons / np.einsum('ij,ji->', dmat, overlap)
    return mol, basis, grids, dmat


@pytest.fixture(scope='module')
def skala():
    try:
        return XC.load_skala('skala-1.1')
    except Exception as error:  # no network and no cached checkpoint
        pytest.skip('the Skala checkpoint is not available (' + str(error) + ')')


def test_grid_exposes_unpartitioned_weights(system):
    """The native grid must carry the raw single-atom weights Skala needs alongside the Becke ones."""
    _, _, grids, _ = system
    assert grids.atomic_weights is not None
    assert grids.atomic_weights.shape == grids.weights.shape
    assert np.all(grids.atomic_weights > 0.0), 'raw quadrature weights are positive'
    # weights = atomic_weights * P_A(r), a Becke partitioning factor in [0, 1]: it reaches 0 at points
    # that the partitioning assigns entirely to another atom.
    ratio = grids.weights / grids.atomic_weights
    assert np.all(ratio >= 0.0)
    assert np.all(ratio <= 1.0 + 1e-12)
    assert ratio.max() > 0.99, 'points close to their own nucleus keep essentially all of their weight'


def test_numgrid_scheme_is_rejected(system, skala):
    """The 'numgrid' scheme cannot supply unpartitioned weights, so it must fail loudly, not silently."""
    mol, basis, _, dmat = system
    grids = Grids(mol, level=3, preset='compact', verbose=False)
    assert grids.atomic_weights is None
    with pytest.raises(ValueError, match='unpartitioned'):
        Integrals.eval_xc_skala(basis, dmat, grids, skala, print_nelec=False)


def test_metadata(skala):
    """The checkpoint declares the features the driver feeds it and the dispersion it expects."""
    assert skala.name == 'skala-1.1'
    assert {'density', 'grad', 'kin', 'grid_weights', 'atomic_grid_weights',
            'atomic_grid_sizes'} <= set(skala.features)
    # Skala 1.1 is parametrised together with a D3 correction; DFT(..., dispersion=True) picks this up.
    assert skala.d3_settings() == 'b3lyp5'


def test_potential_is_the_energy_derivative(system, skala):
    """V_xc must be dE_xc/dP: compare a directional finite difference against trace(V @ D).

    This is what validates the spin convention (the model takes rho/2 in each of two channels), the fact
    that the quadrature weights are already inside the cotangents, and the AO contraction in the third
    pass -- any one of them being wrong shows up here as a factor of two or four.
    """
    _, basis, grids, dmat = system
    energy, potential = Integrals.eval_xc_skala(basis, dmat, grids, skala, print_nelec=False)
    assert np.isfinite(energy)
    assert np.allclose(potential, potential.T, atol=1e-10), 'V_xc must be symmetric'

    rng = np.random.default_rng(1)
    direction = rng.standard_normal(dmat.shape)
    direction = 0.5 * (direction + direction.T)      # stay in the symmetric-matrix subspace

    # The step is chosen at the bottom of the usual V-shaped error curve. The model's energy carries
    # about 1e-9 Ha of its own numerical noise, which a central difference amplifies by 1/(2h), so
    # smaller steps get worse rather than better; around h = 3e-4 the agreement bottoms out near 1e-4
    # relative. The tolerance below is loose compared with that floor but still two to three orders of
    # magnitude tighter than any misplaced factor of two would be.
    step = 3e-4
    plus, _ = Integrals.eval_xc_skala(basis, dmat + step * direction, grids, skala, print_nelec=False)
    minus, _ = Integrals.eval_xc_skala(basis, dmat - step * direction, grids, skala, print_nelec=False)
    numerical = (plus - minus) / (2.0 * step)
    analytical = float(np.einsum('ij,ij->', potential, direction))

    assert numerical == pytest.approx(analytical, rel=2e-3, abs=1e-7)


def test_chunking_does_not_change_the_result(system, skala):
    """The model is additive over atoms, so splitting the grid into chunks must be exact.

    "Exact" here means to the model's own reproducibility rather than to machine precision: a different
    chunking presents the network with differently shaped batches, which changes summation orders and
    moves the energy by around 1e-9 Ha. That is well below any SCF convergence threshold, and within a
    single calculation the chunking is fixed, so the shift is systematic rather than random. A wrongly
    mapped chunk, by contrast, scrambles whole atoms' worth of grid points and shows up immediately.
    """
    _, basis, grids, dmat = system
    whole = Integrals.eval_xc_skala(basis, dmat, grids, skala, max_points_per_chunk=10 ** 9,
                                    print_nelec=False)
    chunked = Integrals.eval_xc_skala(basis, dmat, grids, skala, max_points_per_chunk=1,
                                      print_nelec=False)
    assert chunked[0] == pytest.approx(whole[0], rel=1e-8, abs=1e-8)
    assert np.allclose(chunked[1], whole[1], rtol=1e-6, atol=1e-8)


def test_blocksize_does_not_change_the_result(system, skala):
    """The grid blocking is an implementation detail of the AO passes, not of the functional."""
    _, basis, grids, dmat = system
    coarse = Integrals.eval_xc_skala(basis, dmat, grids, skala, blocksize=10 ** 9, print_nelec=False)
    fine = Integrals.eval_xc_skala(basis, dmat, grids, skala, blocksize=777, print_nelec=False)
    assert fine[0] == pytest.approx(coarse[0], rel=1e-10, abs=1e-12)
    assert np.allclose(fine[1], coarse[1], rtol=1e-9, atol=1e-11)


def test_matches_published_reference_energy(skala):
    """Reproduce one of Microsoft's published Skala total energies end to end.

    The value is H2 at def2-SVP from ``benchmark/reference/measurements.json`` in the Skala repository,
    with the geometry taken from GMTKN55 ``W4-11/h2/coord`` (in Bohr). Their protocol is spherical
    orbitals, density fitting with def2-universal-jkfit, grid level 3 and ``conv_tol = 5e-6`` -- and the
    published Skala numbers **include** the D3 correction, because ``SkalaKS`` attaches it by default.
    This is the strongest single check in this file: it exercises the grid, the model, the potential and
    the dispersion term together against a number PyFock had no part in producing.
    """
    pytest.importorskip('dftd3', reason='the reference energies include the D3 correction')

    # The coord file is in Bohr and Mol takes Angstrom. Divide by Angs2BohrFactor rather than
    # multiplying by Bohr2AngsFactor (the convention in DFT_Grad and DFT_NumGrad): Mol multiplies by
    # that same constant, and the two are not exact reciprocals, so only this round trips faithfully.
    scale = 1.0 / Data.Angs2BohrFactor
    mol = Mol(atoms=[['H', 0.700986297 * scale, 0.0, -2e-9 * scale],
                     ['H', -0.700986297 * scale, 0.0, 2e-9 * scale]])
    basis = Basis(mol, {'all': Basis.load(mol=mol, basis_name='def2-SVP')})
    auxbasis = Basis(mol, {'all': Basis.load(mol=mol, basis_name='def2-universal-jkfit')})
    dft = DFT(mol, basis, auxbasis, xc='skala-1.1',
              grids=Grids(mol, level=3, verbose=False), dispersion=True)
    dft.sao = True
    dft.conv_crit = 5e-6
    with contextlib.redirect_stdout(io.StringIO()):
        energy, _ = dft.scf()

    assert dft.converged
    assert float(energy) == pytest.approx(-1.1683906705, abs=5e-7)


def test_gradient_is_translationally_invariant(skala):
    """The net force must vanish, which no reference calculation is needed to check.

    This is the sharpest test of the grid response. Skala's features are integrals over each atomic
    grid, so evaluating the gradient on a frozen grid -- the approximation PyFock makes for semilocal
    functionals -- breaks translational invariance by ~1e-2 Ha/Bohr, the size of the forces themselves.
    With the Becke weight derivatives and the grid-translation term included it returns to ~1e-14.
    """
    mol = Mol(atoms=[list(atom) for atom in H2O])
    basis = Basis(mol, {'all': Basis.load(mol=mol, basis_name='def2-SVP')})
    auxbasis = Basis(mol, {'all': Basis.load(mol=mol, basis_name='def2-universal-jfit')})
    dft = DFT(mol, basis, auxbasis, xc='skala-1.1', grids=Grids(mol, level=1, verbose=False))
    dft.conv_crit = 1e-8
    with contextlib.redirect_stdout(io.StringIO()):
        dft.scf()
        assert dft.converged
        forces = DFT_Grad(dft, verbose=False).calculate()['forces']

    scale = float(np.abs(forces).max())
    assert np.abs(forces.sum(axis=0)).max() < 1e-8 * max(scale, 1.0)


def test_grid_response_matters(system, skala):
    """Dropping the grid response must visibly change the gradient, and break translational invariance.

    Guards against the terms being silently inert -- a wrong but plausible gradient is the failure mode
    that cost the most time to find here.
    """
    mol, basis, grids, dmat = system
    full = Integrals.eval_xc_grad_skala(basis, dmat, grids, skala, grid_response=True)
    fixed = Integrals.eval_xc_grad_skala(basis, dmat, grids, skala, grid_response=False)
    assert np.abs(full[1]).max() > 1e-4, 'the grid-response terms are not contributing'
    assert np.abs(full[1] - fixed[1]).max() > 1e-4
    assert np.allclose(full[0], fixed[0]), 'the Pulay term must not depend on the switch'


def test_explicit_nuclear_term_is_present(system, skala):
    """Skala reads the grid geometry, so part of dE/dR bypasses the density entirely.

    If the extra cotangents were silently dropped -- an easy way for the gradient to look plausible but
    be wrong -- this term would come back as exactly zero.
    """
    mol, _, grids, dmat = system
    rho = np.zeros(grids.size)
    rho_grad = np.zeros((3, grids.size))
    tau = np.zeros(grids.size)
    from pyfock.Integrals.eval_xc_skala import _bfs_arrays, _block_aos
    from opt_einsum import contract
    basis = system[1]
    bfs = _bfs_arrays(basis)
    ao, ao_grad = _block_aos(bfs, grids.coords, None, None, None)
    Fmj = ao @ dmat
    rho[:] = contract('mj,mj->m', Fmj, ao)
    rho_grad[:] = 2 * contract('mj,kmj->km', Fmj, ao_grad)
    tau[:] = 0.5 * contract('ij,kmi,kmj->m', dmat, ao_grad, ao_grad)

    atom_coords = np.asarray(mol.coordsBohrs, dtype=np.float64).reshape(-1, 3)
    out = skala.exc_and_potential(rho, rho_grad, tau, grids.coords, grids.weights,
                                  grids.atomic_weights, grids.atom_idx, atom_coords,
                                  nuclear_terms=True)
    assert len(out) == 6
    explicit, dweights = out[4], out[5]
    assert explicit.shape == atom_coords.shape
    assert np.all(np.isfinite(explicit))
    # dE/dw is what the Becke weight derivatives are contracted with; it must be real and non-trivial.
    assert dweights.shape == (grids.size,)
    assert np.all(np.isfinite(dweights))
    assert np.abs(dweights).max() > 0.0, 'dE/dw must not be identically zero'


def test_grid_response_works_for_semilocal_functionals():
    """The grid response is functional-agnostic, so it must fix translational invariance for any of them.

    Off by default for semilocal functionals (PyFock's published references were generated without it,
    as is PySCF's default); this checks the opt-in path rather than a change of default.
    """
    mol = Mol(atoms=[list(atom) for atom in H2O])
    basis = Basis(mol, {'all': Basis.load(mol=mol, basis_name='def2-SVP')})
    auxbasis = Basis(mol, {'all': Basis.load(mol=mol, basis_name='def2-universal-jfit')})

    for xc in ('LDA', 'PBE', 'R2SCAN'):          # one per semilocal family
        dft = DFT(mol, basis, auxbasis, xc=xc, grids=Grids(mol, level=1, verbose=False))
        dft.conv_crit = 1e-9
        with contextlib.redirect_stdout(io.StringIO()):
            dft.scf()
            fixed = DFT_Grad(dft, verbose=False, grid_response=False).calculate()['forces']
            responsive = DFT_Grad(dft, verbose=False, grid_response=True).calculate()['forces']

        scale = max(float(np.abs(responsive).max()), 1e-6)
        assert np.abs(responsive.sum(axis=0)).max() < 1e-9 * scale, xc
        assert np.abs(fixed.sum(axis=0)).max() > np.abs(responsive.sum(axis=0)).max(), xc
        # It is a correction, not a different gradient.
        assert np.abs(responsive - fixed).max() < 0.05 * scale, xc


def test_skala_refuses_a_frozen_grid(skala):
    """Skala without the grid response is wrong by the size of the forces, so it must not be offered."""
    mol = Mol(atoms=[list(atom) for atom in H2O])
    basis = Basis(mol, {'all': Basis.load(mol=mol, basis_name='def2-SVP')})
    auxbasis = Basis(mol, {'all': Basis.load(mol=mol, basis_name='def2-universal-jfit')})
    dft = DFT(mol, basis, auxbasis, xc='skala-1.1', grids=Grids(mol, level=1, verbose=False))
    dft.conv_crit = 1e-8
    with contextlib.redirect_stdout(io.StringIO()):
        dft.scf()
    with pytest.raises(ValueError, match='not usable with Skala'):
        DFT_Grad(dft, verbose=False, grid_response=False)


def test_unknown_functional_name():
    assert not XC.is_skala('PBE')
    assert XC.is_skala('Skala-1.1')
    assert XC.canonical_skala_name('SKALA-1.1') == 'skala-1.1'
    with pytest.raises(ValueError, match='Unknown Skala functional'):
        XC.canonical_skala_name('skala-9.9')


def test_gpu_scf_matches_the_cpu_scf():
    """The whole SCF on the device must reproduce the CPU energy.

    This is the regression test for stream ordering in ``Integrals.eval_xc_skala_cupy``: the AO kernel
    and the CuPy contractions that read its output have to be on one stream. When they were not, the
    SCF still ran, but it wandered ~2e-2 Ha away to a slightly different place on every attempt.
    """
    cp = pytest.importorskip('cupy')
    from numba import cuda
    try:
        if not (cuda.is_available() and cp.cuda.runtime.getDeviceCount() > 0):
            pytest.skip('no CUDA device')
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip('no CUDA device')

    mol = Mol(atoms=[list(atom) for atom in H2O])
    energies = {}
    for use_gpu in (False, True):
        basis = Basis(mol, {'all': Basis.load(mol=mol, basis_name='def2-SVP')})
        auxbasis = Basis(mol, {'all': Basis.load(mol=mol, basis_name='def2-universal-jfit')})
        dft = DFT(mol, basis, auxbasis, xc='skala-1.1', use_gpu=use_gpu)
        dft.conv_crit = 1e-8
        with contextlib.redirect_stdout(io.StringIO()):
            energies[use_gpu], _ = dft.scf()

    # Float64 reductions associate differently on the device, which leaves ~1e-7 Ha; the bug this
    # guards against was five orders of magnitude larger.
    assert abs(energies[True] - energies[False]) < 1e-5
