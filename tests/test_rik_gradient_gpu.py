"""
Hardware tests of the RI exact-exchange gradient on the GPU (HF and global hybrids, ``DF_algo=11``).

The CPU routines, themselves validated against finite differences and PySCF
(``tests/test_rik_gradient.py``), are the value oracle:

* the derivative kernel with general per-pair weights (:class:`RowsGradContext`) against
  :func:`pyfock.Integrals.df_algo12_grad.grad_contract_rows`, Cartesian and spherical fit space,
  strict Schwarz on and off, with and without the gradient screening, d and f shells;
* the two-center derivative contraction with a general weight matrix;
* the raw device rows and the occupied fit blocks ``Y_P = F^T c^P F`` (one or several auxiliary
  blocks, raw or orthonormalized rows), and the block-wise three-center gradient (weights formed per
  block of auxiliary shells) against ``gradient_rows`` + ``grad_contract_rows``, with and without
  the fused Coulomb weights;
* complete ``DFT_Grad`` gradients on both devices from one converged SCF (HF, PBE0, B3LYP; CAO and
  SAO; strict Schwarz; the fused and the separate pass; a GPU SCF as well as a CPU one).
"""
import contextlib
import io

import numpy as np
import pytest

cp = pytest.importorskip('cupy')
from numba import cuda

try:
    GPU_AVAILABLE = cuda.is_available() and cp.cuda.runtime.getDeviceCount() > 0
except cp.cuda.runtime.CUDARuntimeError:
    GPU_AVAILABLE = False
pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not GPU_AVAILABLE, reason='CUDA device unavailable')]

from pyfock import Basis, DFT, DFT_Grad, Integrals, Mol
from pyfock.DFT_Helper_Coulomb import _pseudo_cartesian_metric_diagonal
from pyfock.Integrals import df_algo11_exchange as cpu_x
from pyfock.Integrals import df_algo11_exchange_cupy as gpu_x
from pyfock.Integrals import df_algo11_helpers as cpu_plan
from pyfock.Integrals import df_algo11_helpers_cupy as gpu_plan
from pyfock.Integrals import df_algo12_grad as grad12
from pyfock.Integrals.df_algo10_helpers import STRICT_PAIR_CUTOFF
from pyfock.Integrals.rys_3c2e_grad_contract_cupy import rys_3c2e_grad_contract_rows_cupy
from pyfock.Integrals.schwarz_helpers import eri_4c2e_diag


BASIS, AUX = 'def2-SVP', 'def2-universal-jkfit'
WATER = [['O', 0.0, 0.05, 0.117], ['H', 0.02, 0.757, -0.467], ['H', 0.0, -0.787, -0.427]]
TWO_WATERS = WATER + [[s, x + 3.0, y + 0.5, z] for s, x, y, z in WATER]


def _bases(atoms, basis_name=BASIS):
    mol = Mol(atoms=[list(a) for a in atoms])
    basis = Basis(mol, {'all': Basis.load(mol=mol, basis_name=basis_name)})
    aux = Basis(mol, {'all': Basis.load(mol=mol, basis_name=AUX)})
    return mol, basis, aux


class System:
    """Bases, fit metric, Schwarz diagonals and the algorithm-11 gradient plan of one geometry."""

    def __init__(self, sao, strict, atoms=TWO_WATERS, basis_name=BASIS):
        _, self.basis, self.aux = _bases(atoms, basis_name)
        self.sao, self.strict = sao, strict
        self.sqrt4 = np.sqrt(np.abs(eri_4c2e_diag(self.basis)))
        V = Integrals.rys_2c2e_symm(self.aux)
        if sao:
            self.metric = self.aux.cart2sph_operator_blockwise(V)
            diag = _pseudo_cartesian_metric_diagonal(self.aux, self.metric, self.aux.sph2cart_basis()) + 1e-12
            self.tables = cpu_x._cart2sph_tables(self.aux)[:4]
            self.nfit = int(self.tables[3].sum())
        else:
            self.metric = V
            diag = np.diag(V)
            self.tables = None
            self.nfit = self.aux.bfs_nao
        self.sqrt2 = np.sqrt(np.abs(diag))
        self.plan = grad12.build_grad_plan(self.basis, self.aux, self.sqrt4, self.sqrt2, 1e-9, strict,
                                           sao=sao, far_field=False)

    def cpu_rows(self, orthonormalize=False):
        plan = cpu_plan.build_plan(self.basis, self.aux, self.sqrt4, self.sqrt2, 1e-9, self.strict, sao=self.sao)
        return cpu_x.build_exchange(plan, self.basis, self.aux, self.metric if orthonormalize else None,
                                    sao=self.sao, orthonormalize=orthonormalize)

    def gpu_rows(self, orthonormalize=False):
        plan = gpu_plan.build_plan_cupy(self.basis, self.aux, cp.asarray(self.sqrt4), cp.asarray(self.sqrt2),
                                        1e-9, self.strict, sao=self.sao)
        return gpu_x.build_exchange_cupy(plan, self.basis, self.aux, self.metric if orthonormalize else None,
                                         sao=self.sao, release_plan_values=True, orthonormalize=orthonormalize)

    def pair_rows(self):
        nao = self.basis.bfs_nao
        mu, nu = np.tril_indices(nao)
        if self.strict:
            keep = self.sqrt4[mu, nu] ** 2 >= STRICT_PAIR_CUTOFF
            assert not keep.all()
            mu, nu = mu[keep], nu[keep]
        row_of = np.full((nao, nao), -1, dtype=np.int64)
        row_of[mu, nu] = np.arange(mu.size)
        return mu, nu, row_of


def _close(gpu, cpu, rel=1e-11):
    np.testing.assert_allclose(gpu, cpu, rtol=0, atol=rel * max(np.abs(cpu).max(), 1.0))


@pytest.mark.parametrize('sao', [False, True], ids=['cao', 'sao'])
@pytest.mark.parametrize('strict', [False, True], ids=['schwarz', 'strict'])
def test_rows_kernel_matches_cpu(sao, strict):
    s = System(sao, strict)
    mu, nu, row_of = s.pair_rows()
    rng = np.random.default_rng(5)
    # a general weight per pair and fit function, decaying with the pair's Schwarz factor like the
    # real ones, so that the gradient screening removes a part of the blocks
    G = rng.standard_normal((mu.size, s.nfit)) * s.sqrt4[mu, nu][:, None]
    for thr in (0.0, 1e-6):
        cpu = grad12.grad_contract_rows(s.plan, G, row_of, s.tables, threshold_grad=thr)
        gpu = rys_3c2e_grad_contract_rows_cupy(s.basis, s.aux, s.plan, G, mu, nu, s.tables, threshold_grad=thr)
        _close(gpu, cpu)
        assert np.abs(gpu.sum(axis=0)).max() < 1e-10 * np.abs(gpu).max()


def test_rows_kernel_with_f_functions():
    """def2-TZVP: f shells in the orbital basis change the kernel's recursion orders and array shapes."""
    s = System(True, False, atoms=WATER, basis_name='def2-TZVP')
    mu, nu, row_of = s.pair_rows()
    G = np.random.default_rng(6).standard_normal((mu.size, s.nfit))
    cpu = grad12.grad_contract_rows(s.plan, G, row_of, s.tables, threshold_grad=0.0)
    gpu = rys_3c2e_grad_contract_rows_cupy(s.basis, s.aux, s.plan, G, mu, nu, s.tables, threshold_grad=0.0)
    _close(gpu, cpu)


def test_two_center_general_weights_match_cpu():
    _, _, aux = _bases(TWO_WATERS)
    rng = np.random.default_rng(3)
    A = rng.standard_normal((aux.bfs_nao, aux.bfs_nao))
    W = A + A.T
    cpu = Integrals.rys_2c2e_grad_contract(aux, weights=W)
    _close(Integrals.rys_2c2e_grad_contract_cupy(aux, weights=W), cpu)
    _close(Integrals.rys_2c2e_grad_contract_cupy(aux, weights=cp.asarray(W)), cpu)
    c = rng.standard_normal(aux.bfs_nao)
    _close(Integrals.rys_2c2e_grad_contract_cupy(aux, weights=np.outer(c, c)),
           Integrals.rys_2c2e_grad_contract_cupy(aux, c))


@pytest.fixture(scope='module', params=[(False, False), (True, True)], ids=['cao', 'sao_strict'])
def system(request):
    sao, strict = request.param
    s = System(sao, strict)
    s.ref = s.cpu_rows()
    s.F = 0.3 * np.random.default_rng(8).standard_normal((s.basis.bfs_nao, 7))
    s.Y_ref = cpu_x.occupied_fit_blocks(s.ref, s.F, s.metric)
    # the occupied blocks before the metric solve (an identity metric leaves them as they are)
    s.Y_raw = cpu_x.occupied_fit_blocks(s.ref, s.F, np.eye(s.nfit))
    return s


def _check_fit_blocks(s, Y):
    """
    ``Y`` solves ``(P|Q) Y_Q = (F^T (ij|P) F)``. The Cartesian jkfit metric of these waters has a
    condition number near 1e12, so two correct solvers (Cholesky here, LU, the CPU's) give blocks that
    differ by ~1e-5 relative while the residuals and every contraction with the integrals agree to
    round-off; only the well-conditioned spherical fit space is compared element by element.
    """
    _close(np.einsum('PQ,Qab->Pab', s.metric, Y), s.Y_raw, 1e-11)
    if s.sao:
        _close(Y, s.Y_ref, 1e-8)


def test_raw_device_rows_and_fit_blocks_match_cpu(system):
    s = system
    ex = s.gpu_rows()
    assert not ex.orthonormal
    np.testing.assert_array_equal(ex.row_mu, s.ref.row_mu)
    np.testing.assert_array_equal(ex.row_nu, s.ref.row_nu)
    _close(cp.asnumpy(ex.B), s.ref.B, 1e-13)
    nocc = s.F.shape[1]
    for block in (None, 8 * (s.basis.bfs_nao * nocc + ex.max_bin_cells) * 50):     # one block / several
        _check_fit_blocks(s, cp.asnumpy(gpu_x.occupied_fit_blocks_cupy(ex, s.F, s.metric, block_memory_bytes=block)))
    # orthonormalized rows give the same blocks with one triangular solve
    ex_o = s.gpu_rows(orthonormalize=True)
    assert ex_o.orthonormal
    Y = gpu_x.occupied_fit_blocks_cupy(ex_o, s.F, s.metric)
    _check_fit_blocks(s, cp.asnumpy(Y))
    # the metric weights W_PQ = <Y_P, Y_Q> from the packed lower triangles
    Ym = cp.asnumpy(Y).reshape(s.nfit, -1)
    _close(cp.asnumpy(gpu_x.exchange_metric_weights_cupy(Y)), Ym @ Ym.T, 1e-13)


@pytest.mark.parametrize('coulomb', [False, True], ids=['exchange', 'fused'])
def test_blockwise_exchange_gradient_matches_cpu(system, coulomb):
    s = system
    rng = np.random.default_rng(9)
    D = s.F @ s.F.T
    c_fit = rng.standard_normal(s.nfit)
    c_cart = s.aux.cart2sph_basis().T @ c_fit if s.sao else c_fit
    # both sides contract the same Y (the device's), so that the comparison isolates the weights
    # and the derivative kernel from the conditioning of the fit (see _check_fit_blocks)
    Y = gpu_x.occupied_fit_blocks_cupy(s.gpu_rows(), s.F, s.metric)
    rows = s.cpu_rows()
    cpu_x.gradient_rows(rows, s.F, cp.asnumpy(Y), scale=-0.35, dmat=D if coulomb else None,
                        coeff=c_fit if coulomb else None)
    row_of = np.full((s.basis.bfs_nao, s.basis.bfs_nao), -1, dtype=np.int64)
    row_of[rows.row_mu, rows.row_nu] = np.arange(rows.nrows)
    cpu = grad12.grad_contract_rows(s.plan, rows.B, row_of, s.tables, threshold_grad=1e-11)
    for block in (None, 1):             # the whole auxiliary basis at once / one shell at a time
        ex = s.gpu_rows()
        gpu = gpu_x.exchange_gradient_cupy(ex, s.basis, s.aux, s.plan, s.F, Y, scale=-0.35,
                                           dmat=D if coulomb else None, coeff=c_cart if coulomb else None,
                                           fit_tables=s.tables, threshold_grad=1e-11, block_memory_bytes=block)
        assert ex.B is None
        _close(gpu, cpu, 1e-10)


def _converged(xc, sao, use_gpu, strict=False, atoms=WATER, conv=1e-9):
    mol, basis, aux = _bases(atoms)
    dft = DFT(mol, basis, aux, xc=xc, conv_crit=conv, ncores=2, gridsLevel=3, use_gpu=use_gpu)
    dft.max_itr = 60
    dft.sao = sao
    dft.strict_schwarz = strict
    with contextlib.redirect_stdout(io.StringIO()):
        dft.scf()
    assert dft.converged
    return dft


@pytest.mark.parametrize('xc,sao,scf_gpu', [('HF', True, True), ('HF', False, False),
                                            ('PBE0', True, False), ('B3LYP', False, True)])
def test_dft_grad_gpu_matches_cpu(xc, sao, scf_gpu):
    """Every term of the assembled gradient, from one converged SCF on either device."""
    dft = _converged(xc, sao, scf_gpu)
    with contextlib.redirect_stdout(io.StringIO()):
        gpu = DFT_Grad(dft, verbose=False, use_gpu=True).calculate()
        cpu = DFT_Grad(dft, verbose=False, use_gpu=False).calculate()
    assert 'coulomb_exchange_df' in gpu['gradient_components']
    for term in cpu['gradient_components']:
        np.testing.assert_allclose(gpu['gradient_components'][term], cpu['gradient_components'][term],
                                   rtol=0, atol=1e-9, err_msg=term)
    np.testing.assert_allclose(gpu['gradient'], cpu['gradient'], rtol=0, atol=1e-9)
    if xc == 'HF':
        assert np.abs(gpu['gradient'].sum(axis=0)).max() < 1e-9


def test_gpu_scf_defaults_to_gpu_gradient_and_separate_pass_with_strict_screening():
    atoms = WATER + [[s, x + 4.0, y, z + 1.0] for s, x, y, z in WATER]
    dft = _converged('PBE0', False, True, strict=True, atoms=atoms)
    with contextlib.redirect_stdout(io.StringIO()):
        grad = DFT_Grad(dft, verbose=False, threshold_schwarz_grad=1e-16)
        assert grad.use_gpu
        fused = grad.calculate()
        separate = DFT_Grad(dft, verbose=False, threshold_schwarz_grad=1e-16, separate_exchange=True).calculate()
        cpu = DFT_Grad(dft, verbose=False, threshold_schwarz_grad=1e-16, separate_exchange=True,
                       use_gpu=False).calculate()
    comp = separate['gradient_components']
    np.testing.assert_allclose(fused['gradient_components']['coulomb_exchange_df'],
                               comp['coulomb_df'] + comp['exchange_df'], atol=1e-10, rtol=0)
    for term in ('coulomb_df', 'exchange_df'):
        np.testing.assert_allclose(comp[term], cpu['gradient_components'][term], atol=1e-10, rtol=0)
    np.testing.assert_allclose(fused['gradient'], cpu['gradient'], atol=1e-9, rtol=0)
