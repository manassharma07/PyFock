"""Hardware tests for the multipole-accelerated CUDA DF-J (``DF_algo=12``).

The CPU plan of :mod:`pyfock.Integrals.df_algo12_helpers` is the value oracle: the GPU
driver reuses its metadata, so the near-field blocks must agree to rounding and ``gamma``
and ``J`` must agree with the CPU passes, not merely with algorithm 11.  A chain of water
molecules has a substantial far field; a single water molecule has none and must therefore
reproduce algorithm 11 exactly.
"""
import numpy as np
import pytest

cp = pytest.importorskip('cupy')
from numba import cuda

try:
    GPU_AVAILABLE = cuda.is_available() and cp.cuda.runtime.getDeviceCount() > 0
except cp.cuda.runtime.CUDARuntimeError:
    GPU_AVAILABLE = False
pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not GPU_AVAILABLE, reason='CUDA device unavailable')]

from pyfock import Basis, Mol
from pyfock.Integrals import df_algo11_helpers as cpu11
from pyfock.Integrals import df_algo11_helpers_cupy as gpu11
from pyfock.Integrals import df_algo12_helpers as cpu12
from pyfock.Integrals import df_algo12_helpers_cupy as gpu12
from tests.test_df_algo12 import _system


@pytest.fixture(scope='module', params=[False, True], ids=['cao', 'sao'])
def chain(request):
    return request.param, _system('water_chain', request.param)


def _coeff(aux, seed=11):
    return np.random.default_rng(seed).normal(size=aux.bfs_nao)


@pytest.mark.parametrize('strict', [False, True])
@pytest.mark.parametrize('threshold', [1e-9, 1e-12])
def test_values_and_contractions(chain, strict, threshold):
    sao, (basis, aux, sqrt4, sqrt2, _, dmat) = chain
    ref = cpu12.build_plan(basis, aux, sqrt4, sqrt2, threshold, strict, sao=sao)
    plan = gpu12.build_plan_cupy(basis, aux, cp.asarray(sqrt4), cp.asarray(sqrt2),
                                 threshold, strict, sao=sao, debug=True)
    assert plan.n_branches == ref.n_branches
    assert plan.n_entries == ref.n_entries
    np.testing.assert_array_equal(plan.pair_ncols, ref.pair_ncols)
    np.testing.assert_array_equal(plan.pair_offset, ref.pair_offset)
    np.testing.assert_allclose(cp.asnumpy(plan.values), ref.values, atol=1e-12, rtol=0)

    coeff = _coeff(aux)
    g = cp.asnumpy(gpu12.gamma_from_plan_cupy(plan, cp.asarray(dmat)))
    j = cp.asnumpy(gpu12.J_from_plan_cupy(plan, cp.asarray(coeff)))
    np.testing.assert_allclose(g, cpu12.gamma_from_plan(ref, dmat), atol=1e-10, rtol=0)
    np.testing.assert_allclose(j, cpu12.J_from_plan(ref, coeff), atol=1e-10, rtol=0)
    np.testing.assert_array_equal(j, j.T)

    # the far field is an approximation of algorithm 11, not a different quantity
    r11 = cpu11.build_plan(basis, aux, sqrt4, sqrt2, threshold, strict, sao=sao)
    g11 = cpu11.gamma_from_plan(r11, dmat)
    j11 = cpu11.J_from_plan(r11, coeff)
    assert np.linalg.norm(g - g11) < 1e-8 * np.linalg.norm(g11)
    assert np.linalg.norm(j - j11) < 1e-8 * np.linalg.norm(j11)


def test_budgets_batches_and_stream(chain):
    """Direct (uncached) near-field batches and a caller-supplied stream change nothing."""
    sao, (basis, aux, sqrt4, sqrt2, _, dmat) = chain
    coeff = _coeff(aux, 5)
    full = gpu12.build_plan_cupy(basis, aux, sqrt4, sqrt2, 1e-9, True, sao=sao)
    g = cp.asnumpy(gpu12.gamma_from_plan_cupy(full, dmat))
    j = cp.asnumpy(gpu12.J_from_plan_cupy(full, coeff))
    old_stream = cp.cuda.get_current_stream()
    stream = cp.cuda.Stream(non_blocking=True)
    for budget in (0.5 * full.memory_gb, 0.0):
        ref = cpu12.build_plan(basis, aux, sqrt4, sqrt2, 1e-9, True, sao=sao, max_memory_gb=budget)
        plan = gpu12.build_plan_cupy(basis, aux, sqrt4, sqrt2, 1e-9, True, sao=sao,
                                     max_memory_gb=budget, cp_stream=stream,
                                     batch_memory_gb=0.0001, debug=True)
        assert cp.cuda.get_current_stream() == old_stream
        assert plan.memory_gb <= budget
        assert plan.fraction_cached == ref.fraction_cached
        assert len(plan.batches) > 1
        np.testing.assert_array_equal(plan.pair_offset, ref.pair_offset)
        np.testing.assert_allclose(cp.asnumpy(plan.values), ref.values, atol=1e-12, rtol=0)
        np.testing.assert_allclose(cp.asnumpy(gpu12.gamma_from_plan_cupy(plan, dmat)), g,
                                   atol=1e-12, rtol=0)
        np.testing.assert_allclose(cp.asnumpy(gpu12.J_from_plan_cupy(plan, coeff)), j,
                                   atol=1e-12, rtol=0)


def test_without_far_field_equals_algo11():
    """One water molecule has no profitable far field: the GPU plans must coincide exactly."""
    basis, aux, sqrt4, sqrt2, _, dmat = _system('h2o', True)
    coeff = _coeff(aux, 7)
    p12 = gpu12.build_plan_cupy(basis, aux, sqrt4, sqrt2, 1e-9, True, sao=True, debug=True)
    p11 = gpu11.build_plan_cupy(basis, aux, sqrt4, sqrt2, 1e-9, True, sao=True)
    assert p12.n_entries == 0
    np.testing.assert_array_equal(p12.pair_ncols, p11.pair_ncols)
    np.testing.assert_allclose(cp.asnumpy(p12.values), cp.asnumpy(p11.values), atol=0, rtol=0)
    np.testing.assert_allclose(cp.asnumpy(gpu12.gamma_from_plan_cupy(p12, dmat)),
                               cp.asnumpy(gpu11.gamma_from_plan_cupy(p11, dmat)), atol=1e-12, rtol=0)
    np.testing.assert_allclose(cp.asnumpy(gpu12.J_from_plan_cupy(p12, coeff)),
                               cp.asnumpy(gpu11.J_from_plan_cupy(p11, coeff)), atol=1e-12, rtol=0)


@pytest.mark.parametrize('basis_name', ['def2-TZVP', 'def2-QZVP'])
def test_f_g_functions(basis_name):
    """High angular momentum in the orbital basis (f, g) on a chain with a real far field."""
    atoms = [[s, x + 5.0 * k, y, z] for k in range(3)
             for s, x, y, z in [['O', 0.0, 0.0, 0.117], ['H', 0.0, 0.757, -0.467],
                                ['H', 0.0, -0.757, -0.467]]]
    mol = Mol(atoms=atoms)
    basis = Basis(mol, {'all': Basis.load(mol=mol, basis_name=basis_name)})
    aux = Basis(mol, {'all': Basis.load(mol=mol, basis_name='def2-universal-jfit')})
    from pyfock import Integrals
    from pyfock.Integrals.df_algo10_helpers import aux_shell_max_bounds
    from pyfock.Integrals.schwarz_helpers import eri_4c2e_diag
    proj = aux.sph2cart_basis() @ aux.cart2sph_basis()
    metric = proj @ Integrals.rys_2c2e_symm(aux) @ proj.T + 1e-12 * np.eye(aux.bfs_nao)
    sqrt4 = np.sqrt(np.abs(eri_4c2e_diag(basis)))
    sqrt2 = aux_shell_max_bounds(np.sqrt(np.abs(np.diag(metric))), aux)
    S = Integrals.overlap_mat_symm(basis)
    dmat = S + 0.3 * S @ S
    ref = cpu12.build_plan(basis, aux, sqrt4, sqrt2, 1e-9, False, sao=True)
    plan = gpu12.build_plan_cupy(basis, aux, sqrt4, sqrt2, 1e-9, False, sao=True, debug=True)
    np.testing.assert_allclose(cp.asnumpy(plan.values), ref.values, atol=1e-12, rtol=0)
    coeff = _coeff(aux, 13)
    np.testing.assert_allclose(cp.asnumpy(gpu12.gamma_from_plan_cupy(plan, dmat)),
                               cpu12.gamma_from_plan(ref, dmat), atol=1e-9, rtol=0)
    np.testing.assert_allclose(cp.asnumpy(gpu12.J_from_plan_cupy(plan, coeff)),
                               cpu12.J_from_plan(ref, coeff), atol=1e-9, rtol=0)


@pytest.mark.parametrize('options', [dict(lmax=6), dict(box_size=4.0, separation=3.0),
                                     dict(break_even=0.25), dict(precision=1e-8)])
def test_multipole_options_follow_the_cpu_plan(options):
    basis, aux, sqrt4, sqrt2, _, dmat = _system('water_chain', True)
    coeff = _coeff(aux, 17)
    ref = cpu12.build_plan(basis, aux, sqrt4, sqrt2, 1e-9, True, sao=True, options=options)
    plan = gpu12.build_plan_cupy(basis, aux, sqrt4, sqrt2, 1e-9, True, sao=True,
                                 options=options, debug=True)
    assert plan.n_entries == ref.n_entries
    np.testing.assert_array_equal(plan.pair_ncols, ref.pair_ncols)
    np.testing.assert_allclose(cp.asnumpy(gpu12.gamma_from_plan_cupy(plan, dmat)),
                               cpu12.gamma_from_plan(ref, dmat), atol=1e-10, rtol=0)
    np.testing.assert_allclose(cp.asnumpy(gpu12.J_from_plan_cupy(plan, coeff)),
                               cpu12.J_from_plan(ref, coeff), atol=1e-10, rtol=0)


def test_low_memory_is_rejected():
    basis, aux, sqrt4, sqrt2, _, _ = _system('h2o', True)
    with pytest.raises(NotImplementedError):
        gpu12.build_plan_cupy(basis, aux, sqrt4, sqrt2, 1e-9, True, sao=True,
                              options=dict(low_memory=True))


def test_no_significant_pairs():
    mol = Mol(atoms=[['He', 0., 0., 0.]])
    basis = Basis(mol, {'all': Basis.load(mol=mol, basis_name='def2-SVP')})
    plan = gpu12.build_plan_cupy(basis, basis, np.zeros((basis.bfs_nao,) * 2),
                                 np.ones(basis.bfs_nao), 1e-9, True, debug=True)
    assert plan.values.size == 0
    assert plan.fraction_cached == 1
    assert not np.any(cp.asnumpy(gpu12.gamma_from_plan_cupy(plan, np.eye(basis.bfs_nao))))
    assert not np.any(cp.asnumpy(gpu12.J_from_plan_cupy(plan, np.ones(basis.bfs_nao))))


def test_driver_shares_the_callers_stream():
    """The DFT driver's J build must stay on the stream the caller opened."""
    from pyfock import Integrals
    from pyfock.DFT_Helper_Coulomb import Jmat_from_density_fitting
    basis, aux, sqrt4, sqrt2, metric, dmat = _system('water_chain', False)
    ref = cpu12.build_plan(basis, aux, sqrt4, sqrt2, 1e-9, True)
    gamma = cpu12.gamma_from_plan(ref, dmat)
    coeff = np.linalg.solve(metric, gamma)
    expected_j = cpu12.J_from_plan(ref, coeff)
    stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        plan = gpu12.build_plan_cupy(basis, aux, sqrt4, sqrt2, 1e-9, True, cp_stream=stream)
        result = Jmat_from_density_fitting(
            dmat=cp.asarray(dmat), DF_algo=12, cholesky=False, cho_decomp_ints2c2e=None,
            df_coeff0=None, Qpq=None, ints3c2e=plan, ints2c2e=cp.asarray(metric),
            indices_dmat_tri=None, indices_dmat_tri_2=None, indicesA=None, indicesB=None,
            indicesC=None, offsets_3c2e=None, sqrt_ints4c2e_diag=cp.asarray(sqrt4),
            sqrt_diag_ints2c2e=cp.asarray(sqrt2), threshold=1e-9, strict_schwarz=True,
            basis=basis, auxbasis=aux, use_gpu=True, keep_ints3c2e_in_gpu=True,
            durationDF_gamma=0., ncores=1, durationDF_coeff=0., durationDF_Jtri=0., durationDF=0.)
        assert cp.cuda.get_current_stream() == stream
        np.testing.assert_allclose(cp.asnumpy(result[0]), expected_j, atol=1e-9, rtol=0)
        np.testing.assert_allclose(float(result[-1]), coeff @ gamma, atol=1e-9, rtol=0)


@pytest.mark.regression
def test_scf_energy_matches_algo11_on_both_backends():
    """
    The Coulomb term must not depend on the algorithm, on either backend.

    It must not be compared *across* backends: the GPU exchange-correlation driver
    (``XC_algo=3``) differs from the CPU one by ~1e-4 Ha on this system whatever the DF
    algorithm is.  The sharp statement about the DF term is that algorithm 12 reproduces
    algorithm 11 within a backend, and that it shifts the total by the same amount between
    backends as algorithm 11 does - i.e. it adds no backend dependence of its own.
    """
    from pyfock import DFT
    atoms = [[s, x + 5.0 * k, y, z] for k in range(3)
             for s, x, y, z in [['O', 0.0, 0.0, 0.117], ['H', 0.0, 0.757, -0.467],
                                ['H', 0.0, -0.757, -0.467]]]
    energies = {}
    for algo, gpu in ((11, False), (12, False), (11, True), (12, True)):
        mol = Mol(atoms=atoms)
        basis = Basis(mol, {'all': Basis.load(mol=mol, basis_name='def2-SVP')})
        aux = Basis(mol, {'all': Basis.load(mol=mol, basis_name='def2-universal-jfit')})
        dft = DFT(mol, basis, aux, xc=[1, 7], ncores=2)
        dft.DF_algo = algo
        dft.sao = True
        dft.use_gpu = gpu
        dft.XC_algo = 3 if gpu else 2
        dft.save_ao_values = not gpu
        dft.conv_crit = 1e-8
        energies[algo, gpu], _ = dft.scf()
        assert dft.converged
    assert abs(energies[12, True] - energies[11, True]) < 5e-8
    assert abs(energies[12, False] - energies[11, False]) < 5e-8
    assert abs((energies[12, True] - energies[12, False])
               - (energies[11, True] - energies[11, False])) < 5e-8
