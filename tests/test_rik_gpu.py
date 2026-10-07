"""
Hardware tests of RI exact exchange on the GPU (:mod:`pyfock.Integrals.df_algo11_exchange_cupy`).

The orthonormalized rows of the CPU path (:mod:`pyfock.Integrals.df_algo11_exchange`) are the
value oracle for the device rows and for the K, J and gamma contractions (double and single
precision, several auxiliary block sizes), in CAO and SAO mode.  Complete RI-HF, B3LYP and PBE0
SCF runs on the GPU are compared with the CPU RI-K path, and the dynamic-precision run (fp32
exchange and XC until the energy change is small) must converge to the double-precision energy.
No external package is needed.
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

from pyfock import Basis, DFT, Integrals, Mol
from pyfock.DFT_Helper_Coulomb import _pseudo_cartesian_metric_diagonal
from pyfock.Integrals import df_algo11_exchange as cpu_x
from pyfock.Integrals import df_algo11_exchange_cupy as gpu_x
from pyfock.Integrals import df_algo11_helpers as cpu_plan
from pyfock.Integrals import df_algo11_helpers_cupy as gpu_plan
from pyfock.Integrals.schwarz_helpers import eri_4c2e_diag


WATER = [["O", 0.0, 0.0, 0.117], ["H", 0.0, 0.757, -0.467], ["H", 0.0, -0.757, -0.467]]
BASIS, AUX = 'def2-SVP', 'def2-universal-jkfit'


def _system(sao, two_waters=True):
    atoms = list(WATER) + ([[s, x + 6.0, y, z] for s, x, y, z in WATER] if two_waters else [])
    mol = Mol(atoms=atoms)
    basis = Basis(mol, {'all': Basis.load(mol=mol, basis_name=BASIS)})
    aux = Basis(mol, {'all': Basis.load(mol=mol, basis_name=AUX)})
    metric_cart = Integrals.rys_2c2e_symm(aux)
    sqrt4 = np.sqrt(np.abs(eri_4c2e_diag(basis)))
    if sao:
        metric_fit = aux.cart2sph_operator_blockwise(metric_cart)
        sqrt2 = np.sqrt(np.abs(_pseudo_cartesian_metric_diagonal(aux, metric_fit, aux.sph2cart_basis()) + 1e-12))
    else:
        metric_fit = metric_cart
        sqrt2 = np.sqrt(np.abs(np.diag(metric_cart)))
    return mol, basis, aux, sqrt4, sqrt2, metric_fit


@pytest.fixture(scope='module', params=[False, True], ids=['cao', 'sao'])
def exchanges(request):
    sao = request.param
    _, basis, aux, sqrt4, sqrt2, metric = _system(sao)
    ref_plan = cpu_plan.build_plan(basis, aux, sqrt4, sqrt2, 1e-9, False, sao=sao)
    ref = cpu_x.build_exchange(ref_plan, basis, aux, metric, sao=sao)
    plan = gpu_plan.build_plan_cupy(basis, aux, cp.asarray(sqrt4), cp.asarray(sqrt2), 1e-9, False, sao=sao)
    ex = gpu_x.build_exchange_cupy(plan, basis, aux, metric, sao=sao, release_plan_values=True)
    assert plan.values is None
    return sao, basis, aux, ref, ex


def test_device_rows_match_the_cpu_rows(exchanges):
    sao, basis, aux, ref, ex = exchanges
    assert (ex.nrows, ex.naux, ex.nao) == (ref.nrows, ref.naux, ref.nao)
    assert ex.fit_space == ('spherical' if sao else 'Cartesian')
    np.testing.assert_array_equal(ex.row_mu, ref.row_mu)
    np.testing.assert_array_equal(ex.row_nu, ref.row_nu)
    # same rows, same metric; only the Cholesky factors differ by rounding (amplified by the metric's conditioning)
    scale = np.abs(ref.B).max()
    np.testing.assert_allclose(cp.asnumpy(ex.B), ref.B, atol=1e-8 * scale, rtol=0)
    # bookkeeping of the batched contraction: every function with rows is in exactly one bin,
    # the slabs are at most 1/bin_fill times the listed rows
    assert ex.n_binned == int(np.count_nonzero(np.diff(ex.fn_ptr)))
    assert sum(e - s for s, e, _ in ex.bins) == ex.n_binned
    assert ex.listed_rows == 2 * ex.nrows - int(np.count_nonzero(ref.row_mu == ref.row_nu))
    assert ex.padded_cells <= ex.listed_rows / ex.bin_fill + 1
    assert 'GPU' in ex.summary()


def test_contractions_match_the_cpu(exchanges):
    sao, basis, aux, ref, ex = exchanges
    rng = np.random.default_rng(7)
    factor = rng.standard_normal((basis.bfs_nao, 9))
    K_ref = cpu_x.K_from_exchange(ref, factor)
    scale = np.abs(K_ref).max()
    K = gpu_x.K_from_exchange_cupy(ex, factor)
    assert isinstance(K, cp.ndarray) and K.dtype == cp.float64
    K = cp.asnumpy(K)
    np.testing.assert_allclose(K, K_ref, atol=1e-10 * scale, rtol=0)
    np.testing.assert_allclose(K, K.T, atol=1e-13 * scale, rtol=0)
    # CuPy factor and a tiny budget (one auxiliary function per block)
    K_blocked = cp.asnumpy(gpu_x.K_from_exchange_cupy(ex, cp.asarray(factor), block_memory_bytes=1))
    np.testing.assert_allclose(K_blocked, K_ref, atol=1e-10 * scale, rtol=0)
    # single-precision contraction: fp32 slabs and GEMMs, double-precision result array
    K32 = gpu_x.K_from_exchange_cupy(ex, factor, dtype=cp.float32)
    assert K32.dtype == cp.float64
    np.testing.assert_allclose(cp.asnumpy(K32), K_ref, atol=3e-5 * scale, rtol=0)
    assert np.abs(cp.asnumpy(K32) - K_ref).max() > 1e-10 * scale   # it really ran in single precision
    # Coulomb side
    dmat = factor @ factor.T
    g_ref = cpu_x.gamma_from_exchange(ref, dmat)
    g = gpu_x.gamma_from_exchange_cupy(ex, cp.asarray(dmat))
    # gamma lives in the rotated (orthonormal) fit space, so the rounding difference of the two Cholesky
    # factors shows up in it (amplified by the conditioning of the metric); the invariant norm is tight
    np.testing.assert_allclose(cp.asnumpy(g), g_ref, atol=1e-6 * np.abs(g_ref).max(), rtol=0)
    assert abs(float(g @ g) - g_ref @ g_ref) < 1e-10 * (g_ref @ g_ref)
    J_ref = cpu_x.J_from_exchange(ref, g_ref)
    J = cp.asnumpy(gpu_x.J_from_exchange_cupy(ex, g))
    np.testing.assert_allclose(J, J_ref, atol=1e-10 * np.abs(J_ref).max(), rtol=0)
    np.testing.assert_array_equal(J, J.T)
    # no occupied orbitals: zero exchange
    assert not cp.asnumpy(gpu_x.K_from_exchange_cupy(ex, np.zeros((basis.bfs_nao, 0)))).any()


def _dft(xc, sao, use_gpu, dynamic_precision=None):
    mol = Mol(atoms=[list(a) for a in WATER])
    basis = Basis(mol, {'all': Basis.load(mol=mol, basis_name=BASIS)})
    aux = Basis(mol, {'all': Basis.load(mol=mol, basis_name=AUX)})
    dft = DFT(mol, basis, aux, xc=xc, conv_crit=1e-9, use_gpu=use_gpu, ncores=2)
    dft.max_itr = 60
    dft.sao = sao
    dft.threshold_schwarz = 1e-9
    dft.strict_schwarz = False
    if dynamic_precision is not None:
        dft.dynamic_precision = dynamic_precision
    if use_gpu:
        dft.grids_options = {'use_gpu': False}   # the same grid points as the CPU reference run
    return dft


@pytest.mark.parametrize('xc,sao', [('HF', True), ('B3LYP', False), ('PBE0', True)])
def test_scf_on_the_gpu_matches_the_cpu_rik_path(xc, sao):
    gpu = _dft(xc, sao, True, dynamic_precision=False)
    e_gpu, d_gpu = gpu.scf()
    cpu = _dft(xc, sao, False)
    e_cpu, d_cpu = cpu.scf()
    assert gpu.converged and cpu.converged
    assert gpu.exx_coef == cpu.exx_coef > 0
    # same integrals and screening on both sides; the XC quadrature driver differs (algo 3 vs 2)
    assert abs(float(e_gpu) - float(e_cpu)) < 2e-7, (float(e_gpu), float(e_cpu))
    assert abs(float(gpu.Exx_energy) - float(cpu.Exx_energy)) < 2e-6
    assert np.abs(np.asarray(d_gpu) - np.asarray(d_cpu)).max() < 2e-5


def test_dynamic_precision_converges_to_the_double_precision_energy():
    dyn = _dft('HF', True, True, dynamic_precision=True)
    e_dyn, _ = dyn.scf()
    ref = _dft('HF', True, True, dynamic_precision=False)
    e_ref, _ = ref.scf()
    assert dyn.converged and ref.converged
    assert abs(float(e_dyn) - float(e_ref)) < 1e-8, (float(e_dyn), float(e_ref))
