"""
SCF-level checks of RI exact exchange for global hybrid functionals and of the threaded
exchange contraction of ``DF_algo=11``.

H2O / def2-SVP / def2-universal-jkfit B3LYP and PBE0 energies from the RI-K path
(``DF_algo=11``: orthonormalized rows, threaded contraction) are compared with the dense
reference path (``DF_algo=3``: full three-center tensor, exchange through an explicit
metric solve) in CAO and SAO mode; no external package is needed.  The exchange
contraction itself is checked to be independent of the number of threads, of whether the
partner rows are stored or gathered, and of the auxiliary block size.
"""
from __future__ import annotations

import numba
import numpy as np
import pytest

from pyfock import Basis, DFT, Integrals, Mol
from pyfock.Integrals import df_algo11_exchange as algo11x
from pyfock.Integrals import df_algo11_helpers as algo11
from pyfock.Integrals.schwarz_helpers import eri_4c2e_diag


WATER = [["O", 0.0, 0.0, 0.117], ["H", 0.0, 0.757, -0.467], ["H", 0.0, -0.757, -0.467]]
BASIS, AUX = "def2-SVP", "def2-universal-jkfit"
EXX = {"B3LYP": 0.2, "PBE0": 0.25}


def _run(xc, df_algo, sao):
    mol = Mol(atoms=[list(a) for a in WATER])
    basis = Basis(mol, {"all": Basis.load(mol=mol, basis_name=BASIS)})
    aux = Basis(mol, {"all": Basis.load(mol=mol, basis_name=AUX)})
    dft = DFT(mol, basis, aux, xc=xc, conv_crit=1e-9, use_gpu=False, ncores=2)
    dft.max_itr = 60
    dft.DF_algo = df_algo
    dft.sao = sao
    dft.XC_algo = 2
    dft.strict_schwarz = False
    dft.threshold_schwarz = 1e-9
    energy, dmat = dft.scf()
    assert dft.converged
    return float(energy), dmat, dft


@pytest.mark.parametrize("xc,sao", [("B3LYP", False), ("B3LYP", True), ("PBE0", True)])
def test_hybrid_rik_algo11_matches_dense_algo3(xc, sao):
    e11, d11, dft11 = _run(xc, 11, sao)
    e3, d3, dft3 = _run(xc, 3, sao)
    assert dft11.exx_coef == pytest.approx(EXX[xc])
    # same DF approximation and grid, different storage/contraction path for J and K;
    # only the 1e-9 Schwarz screening of the three-center integrals differs
    assert abs(e11 - e3) < 5e-8, (e11, e3)
    assert np.abs(d11 - d3).max() < 1e-6
    assert dft11.Exx_energy < 0 and abs(dft11.Exx_energy - dft3.Exx_energy) < 1e-7
    assert abs(dft11.Total_energy - e11) < 1e-12


def test_exchange_contraction_independent_of_threads_storage_and_blocking():
    atoms = list(WATER) + [[s, x + 6.0, y, z] for s, x, y, z in WATER]
    mol = Mol(atoms=atoms)
    basis = Basis(mol, {"all": Basis.load(mol=mol, basis_name=BASIS)})
    aux = Basis(mol, {"all": Basis.load(mol=mol, basis_name=AUX)})
    from pyfock.DFT_Helper_Coulomb import _pseudo_cartesian_metric_diagonal
    metric_sph = aux.cart2sph_operator_blockwise(Integrals.rys_2c2e_symm(aux))
    sqrt2 = np.sqrt(np.abs(_pseudo_cartesian_metric_diagonal(aux, metric_sph, aux.sph2cart_basis()) + 1e-12))
    sqrt4 = np.sqrt(np.abs(eri_4c2e_diag(basis)))
    plan = algo11.build_plan(basis, aux, sqrt4, sqrt2, 1e-9, False, sao=True)
    ex_stored = algo11x.build_exchange(plan, basis, aux, metric_sph, sao=True, store_partner_slabs=True)
    ex_gather = algo11x.build_exchange(plan, basis, aux, metric_sph, sao=True, store_partner_slabs=False)
    assert ex_stored.P is not None and ex_gather.P is None
    assert ex_stored.memory_gb > ex_gather.memory_gb
    rng = np.random.default_rng(3)
    factor = rng.standard_normal((basis.bfs_nao, 9))
    nmax = numba.get_num_threads()
    K_ref = algo11x.K_from_exchange(ex_stored, factor)
    assert np.allclose(K_ref, K_ref.T, atol=1e-13)
    for n in sorted({1, 2, nmax}):
        numba.set_num_threads(n)
        try:
            for ex in (ex_stored, ex_gather):
                np.testing.assert_allclose(algo11x.K_from_exchange(ex, factor), K_ref, atol=1e-12, rtol=1e-12)
                # a tiny budget forces one auxiliary function per block
                np.testing.assert_allclose(algo11x.K_from_exchange(ex, factor, block_memory_bytes=1), K_ref, atol=1e-12, rtol=1e-12)
        finally:
            numba.set_num_threads(nmax)
    # Coulomb side is unaffected by the storage choice
    dmat = factor @ factor.T
    g = algo11x.gamma_from_exchange(ex_stored, dmat)
    np.testing.assert_allclose(algo11x.gamma_from_exchange(ex_gather, dmat), g, atol=1e-13, rtol=1e-13)
    np.testing.assert_allclose(algo11x.J_from_exchange(ex_gather, g), algo11x.J_from_exchange(ex_stored, g), atol=1e-13, rtol=1e-13)


def test_default_auxiliary_basis_follows_the_functional():
    """Without an auxbasis, HF and hybrids fit with def2-universal-jkfit, other functionals with jfit."""
    import contextlib
    import io
    from pyfock.DFT import default_auxbasis_name

    mol = Mol(atoms=[list(a) for a in WATER])
    basis = Basis(mol, {"all": Basis.load(mol=mol, basis_name=BASIS)})
    naux = {name: Basis(mol, {"all": Basis.load(mol=mol, basis_name=name)}).bfs_nao
            for name in ("def2-universal-jkfit", "def2-universal-jfit")}
    assert naux["def2-universal-jkfit"] != naux["def2-universal-jfit"]
    for xc, name in (("HF", "def2-universal-jkfit"), ("B3LYP", "def2-universal-jkfit"), ([402], "def2-universal-jkfit"),
                     ("PBE0", "def2-universal-jkfit"), ("PBE", "def2-universal-jfit"), (None, "def2-universal-jfit")):
        assert default_auxbasis_name(xc) == name
        assert DFT(mol, basis, xc=xc).auxbasis.bfs_nao == naux[name]
    # a functional set after construction moves the default, a supplied basis is kept
    late = DFT(mol, basis, conv_crit=1e-9, ncores=2)
    late.xc = "B3LYP"
    jfit = Basis(mol, {"all": Basis.load(mol=mol, basis_name="def2-universal-jfit")})
    kept = DFT(mol, basis, jfit, xc="B3LYP", conv_crit=1e-9, ncores=2)
    explicit = DFT(mol, basis, Basis(mol, {"all": Basis.load(mol=mol, basis_name=AUX)}), xc="B3LYP",
                   conv_crit=1e-9, ncores=2)
    energies = []
    for dft in (late, kept, explicit):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            energies.append(float(dft.scf()[0]))
        assert dft.converged
    assert late.auxbasis.bfs_nao == naux["def2-universal-jkfit"] and kept.auxbasis is jfit
    assert abs(energies[0] - energies[2]) < 1e-9 and abs(energies[1] - energies[2]) > 1e-6
