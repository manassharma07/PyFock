"""Hardware tests for shell-blocked CUDA DF-J; CPU plans are the value oracle."""
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
from pyfock.Integrals import df_algo11_helpers as cpu
from pyfock.Integrals import df_algo11_helpers_cupy as gpu
from tests.test_df_algo11 import _system, _block_mask


@pytest.fixture(scope='module', params=[False, True], ids=['cao', 'sao'])
def water_system(request):
    return request.param, _system('two_waters', 'def2-SVP', request.param)


@pytest.mark.parametrize('strict', [False, True])
@pytest.mark.parametrize('threshold', [1e-9, 1e-12])
def test_values_and_contractions(water_system, strict, threshold):
    sao, (basis, aux, sq4, sq2, dense, _, dmat) = water_system
    ref = cpu.build_plan(basis, aux, sq4, sq2, threshold, strict, sao=sao)
    plan = gpu.build_plan_cupy(basis, aux, cp.asarray(sq4), cp.asarray(sq2),
                               threshold, strict, sao=sao, debug=True)
    np.testing.assert_array_equal(plan.pair_offset, ref.pair_offset)
    np.testing.assert_array_equal(plan.pair_ncols, ref.pair_ncols)
    np.testing.assert_allclose(cp.asnumpy(plan.values), ref.values, atol=1e-12, rtol=0)
    coeff = np.random.default_rng(8).normal(size=aux.bfs_nao)
    g = gpu.gamma_from_plan_cupy(plan, cp.asarray(dmat))
    j = gpu.J_from_plan_cupy(plan, cp.asarray(coeff))
    np.testing.assert_allclose(cp.asnumpy(g), cpu.gamma_from_plan(ref, dmat), atol=1e-10, rtol=0)
    np.testing.assert_allclose(cp.asnumpy(j), cpu.J_from_plan(ref, coeff), atol=1e-10, rtol=0)
    kept = dense * _block_mask(ref, basis, aux)
    np.testing.assert_allclose(cp.asnumpy(g), np.einsum('ijP,ij->P', kept, dmat), atol=1e-10, rtol=0)
    np.testing.assert_allclose(cp.asnumpy(j), np.einsum('ijP,P->ij', kept, coeff), atol=1e-10, rtol=0)
    np.testing.assert_array_equal(cp.asnumpy(j), cp.asnumpy(j.T))


def test_budgets_and_stream(water_system):
    sao, (basis, aux, sq4, sq2, _, _, dmat) = water_system
    full = gpu.build_plan_cupy(basis, aux, sq4, sq2, 1e-9, True, sao=sao)
    coeff = np.cos(np.arange(aux.bfs_nao))
    g = cp.asnumpy(gpu.gamma_from_plan_cupy(full, dmat))
    j = cp.asnumpy(gpu.J_from_plan_cupy(full, coeff))
    old_stream = cp.cuda.get_current_stream()
    stream = cp.cuda.Stream(non_blocking=True)
    for budget in (0.5 * full.memory_gb, 0.0):
        ref = cpu.build_plan(basis, aux, sq4, sq2, 1e-9, True, sao=sao, max_memory_gb=budget)
        plan = gpu.build_plan_cupy(basis, aux, sq4, sq2, 1e-9, True, sao=sao,
                                   max_memory_gb=budget, cp_stream=stream,
                                   batch_memory_gb=0.0001, debug=True)
        assert cp.cuda.get_current_stream() == old_stream
        assert plan.memory_gb <= budget
        assert plan.fraction_cached == ref.fraction_cached
        assert len(plan.batches) > 1
        np.testing.assert_array_equal(plan.pair_offset, ref.pair_offset)
        np.testing.assert_allclose(cp.asnumpy(plan.values), ref.values, atol=1e-12, rtol=0)
        np.testing.assert_allclose(cp.asnumpy(gpu.gamma_from_plan_cupy(plan, dmat)), g, atol=1e-12, rtol=0)
        np.testing.assert_allclose(cp.asnumpy(gpu.J_from_plan_cupy(plan, coeff)), j, atol=1e-12, rtol=0)


def test_no_significant_pairs():
    mol = Mol(atoms=[['He', 0., 0., 0.]])
    basis = Basis(mol, {'all': Basis.load(mol=mol, basis_name='def2-SVP')})
    plan = gpu.build_plan_cupy(basis, basis, np.zeros((basis.bfs_nao,) * 2),
                               np.ones(basis.bfs_nao), 1e-9, True, debug=True)
    assert plan.values.size == 0
    assert plan.fraction_cached == 1
    assert not np.any(cp.asnumpy(gpu.gamma_from_plan_cupy(plan, np.eye(basis.bfs_nao))))
    assert not np.any(cp.asnumpy(gpu.J_from_plan_cupy(plan, np.ones(basis.bfs_nao))))


@pytest.mark.parametrize('algo', [10, 11])
def test_driver_solve_and_j_share_stream(algo):
    """A fresh non-default stream must order density copies, solve and J."""
    from pyfock import Integrals
    from pyfock.DFT_Helper_Coulomb import Jmat_from_density_fitting
    from pyfock.Integrals import df_algo10_helpers as cpu10, df_algo10_helpers_cupy as gpu10
    basis, aux, sq4, sq2, _, _, dmat = _system('two_waters', 'def2-SVP', False)
    ref = cpu.build_plan(basis, aux, sq4, sq2, 1e-9, True)
    metric = Integrals.rys_2c2e_symm(aux)
    gamma = cpu.gamma_from_plan(ref, dmat)
    coeff = np.linalg.solve(metric, gamma)
    expected_j = cpu.J_from_plan(ref, coeff)
    i, j = np.tril_indices(basis.bfs_nao)
    offsets, count = cpu10.calc_offsets_3c2e_schwarz(sq4, sq2, 1e-9, True, i, j)
    stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        sq4d, sq2d = cp.asarray(sq4), cp.asarray(sq2)
        integrals = (gpu.build_plan_cupy(basis, aux, sq4, sq2, 1e-9, True)
                     if algo == 11 else gpu10.rys_3c2e_tri_schwarz_sparse_algo10_cupy(
                         basis, aux, i, j, cp.asarray(offsets), sq4d, sq2d,
                         1e-9, True, count, cp_stream=stream))
        result = Jmat_from_density_fitting(
            dmat=dmat, DF_algo=algo, cholesky=False, cho_decomp_ints2c2e=None,
            df_coeff0=None, Qpq=None, ints3c2e=integrals, ints2c2e=cp.asarray(metric),
            indices_dmat_tri=(i, j), indices_dmat_tri_2=np.tril_indices(basis.bfs_nao, -1),
            indicesA=i, indicesB=j, indicesC=None, offsets_3c2e=offsets,
            sqrt_ints4c2e_diag=sq4d, sqrt_diag_ints2c2e=sq2d, threshold=1e-9,
            strict_schwarz=True, basis=basis, auxbasis=aux, use_gpu=True,
            keep_ints3c2e_in_gpu=True, durationDF_gamma=0., ncores=1,
            durationDF_coeff=0., durationDF_Jtri=0., durationDF=0.)
        assert cp.cuda.get_current_stream() == stream
        np.testing.assert_allclose(cp.asnumpy(result[0]), expected_j, atol=1e-10, rtol=0)
        np.testing.assert_allclose(float(result[-1]), coeff @ gamma, atol=1e-10, rtol=0)


@pytest.mark.parametrize('basis_name', ['def2-TZVP', 'def2-QZVP'])
def test_f_g_water(basis_name):
    basis, aux, sq4, sq2, _, _, dmat = _system('h2o', basis_name, True)
    ref = cpu.build_plan(basis, aux, sq4, sq2, 1e-9, False, sao=True)
    plan = gpu.build_plan_cupy(basis, aux, sq4, sq2, 1e-9, False, sao=True, debug=True)
    np.testing.assert_allclose(cp.asnumpy(plan.values), ref.values, atol=1e-12, rtol=0)
    np.testing.assert_allclose(cp.asnumpy(gpu.gamma_from_plan_cupy(plan, dmat)),
                               cpu.gamma_from_plan(ref, dmat), atol=1e-10, rtol=0)


@pytest.mark.parametrize('sao', [False, True])
@pytest.mark.parametrize('orb_l,aux_l', [(5, 0), (6, 0), (0, 5), (0, 6), (6, 6)])
def test_high_l_shells(orb_l, aux_l, sao):
    """Normalized one-primitive shells exercise h/i and 10-root cooperative work.

    Noncoincident centers force the tabulated (n>5) root branch; an i-i-i block
    exercises the largest scratch class without allocating a cc-pV6Z molecule.
    """
    orbital_mol = Mol(atoms=[['He', 0., 0., 0.]])
    auxiliary_mol = Mol(atoms=[['He', 0.31, -0.17, 0.23]])
    def shell(mol, l):
        return Basis(mol, {'all': f'synthetic\n*\nhe synthetic\n*\n1 {"spdfghi"[l]}\n1.3 1.0\n*\n'})
    basis, aux = shell(orbital_mol, orb_l), shell(auxiliary_mol, aux_l)
    sq4, sq2 = np.ones((basis.bfs_nao,) * 2), np.ones(aux.bfs_nao)
    ref = cpu.build_plan(basis, aux, sq4, sq2, 1e-12, False, sao=sao)
    plan = gpu.build_plan_cupy(basis, aux, sq4, sq2, 1e-12, False, sao=sao, debug=True)
    np.testing.assert_allclose(cp.asnumpy(plan.values), ref.values, atol=1e-12, rtol=0)
    rng = np.random.default_rng(19)
    d = rng.normal(size=(basis.bfs_nao,) * 2)
    d = (d + d.T) / (2 * basis.bfs_nao)
    coeff = rng.normal(size=aux.bfs_nao)
    for budget in (None, 0):
        p = plan if budget is None else gpu.build_plan_cupy(
            basis, aux, sq4, sq2, 1e-12, False, sao=sao, max_memory_gb=0, debug=True)
        np.testing.assert_allclose(cp.asnumpy(gpu.gamma_from_plan_cupy(p, d)),
                                   cpu.gamma_from_plan(ref, d), atol=1e-10, rtol=0)
        np.testing.assert_allclose(cp.asnumpy(gpu.J_from_plan_cupy(p, coeff)),
                                   cpu.J_from_plan(ref, coeff), atol=1e-10, rtol=0)
