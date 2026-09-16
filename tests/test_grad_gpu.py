"""Hardware tests for the CUDA analytical-gradient kernels; the CPU routines are the value oracle.

Every device routine added for GPU gradients is a port of a CPU one that is itself validated against
PySCF and finite differences, so the tests here only have to establish that the port agrees. They run
at two basis sets so that the shell-triplet kernels are exercised with d functions (def2-SVP) and with
f functions (def2-TZVP), which change the recursion orders and the local array shapes the kernels are
specialised on.
"""
from pathlib import Path

import numpy as np
import pytest

cp = pytest.importorskip('cupy')
from numba import cuda

try:
    GPU_AVAILABLE = cuda.is_available() and cp.cuda.runtime.getDeviceCount() > 0
except cp.cuda.runtime.CUDARuntimeError:
    GPU_AVAILABLE = False
pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not GPU_AVAILABLE, reason='CUDA device unavailable')]

from opt_einsum import contract

from pyfock import Basis, Grids, Integrals, Mol

ROOT = Path(__file__).resolve().parents[1]
H2O_XYZ = ROOT / 'examples' / 'h2o.xyz'
AUX_BASIS = 'def2-universal-jfit'


@pytest.fixture(scope='module', params=['def2-SVP', 'def2-TZVP'])
def system(request):
    """Water, a density matrix and fitting coefficients, plus the Schwarz diagonals.

    The density matrix only has to be symmetric and of a realistic magnitude: both sides evaluate
    the same contraction, so a converged one would not test anything a random one does not.
    """
    mol = Mol(coordfile=str(H2O_XYZ))
    basis = Basis(mol, {'all': Basis.load(mol=mol, basis_name=request.param)})
    auxbasis = Basis(mol, {'all': Basis.load(mol=mol, basis_name=AUX_BASIS)})
    rng = np.random.default_rng(20260916)
    a = rng.standard_normal((basis.bfs_nao, basis.bfs_nao))
    dmat = np.ascontiguousarray((a + a.T) * 0.05)
    df_coeff = np.ascontiguousarray(rng.standard_normal(auxbasis.bfs_nao) * 0.1)
    return mol, basis, auxbasis, dmat, df_coeff


def test_3c2e_grad_contract(system):
    _, basis, auxbasis, dmat, df_coeff = system
    cpu = Integrals.rys_3c2e_grad_contract(basis, auxbasis, dmat, df_coeff)
    gpu = Integrals.rys_3c2e_grad_contract_cupy(basis, auxbasis, dmat, df_coeff)
    np.testing.assert_allclose(gpu, cpu, rtol=0, atol=1e-11 * max(np.abs(cpu).max(), 1.0))


def test_2c2e_grad_contract(system):
    _, _, auxbasis, _, df_coeff = system
    cpu = Integrals.rys_2c2e_grad_contract(auxbasis, df_coeff)
    gpu = Integrals.rys_2c2e_grad_contract_cupy(auxbasis, df_coeff)
    np.testing.assert_allclose(gpu, cpu, rtol=0, atol=1e-11 * max(np.abs(cpu).max(), 1.0))


def test_nuc_grad_contract(system):
    mol, basis, _, dmat, _ = system
    cpu = Integrals.rys_nuc_grad_contract(basis, mol, dmat)
    gpu = Integrals.rys_nuc_grad_contract_cupy(basis, mol, dmat)
    np.testing.assert_allclose(gpu, cpu, rtol=0, atol=1e-11 * max(np.abs(cpu).max(), 1.0))


def test_gamma_contract(system):
    """gamma_P = sum_ij D_ij (ij|P), against the integrals rys_3c2e_symm builds and contracts."""
    _, basis, auxbasis, dmat, _ = system
    nbf = basis.bfs_nao
    ints = Integrals.rys_3c2e_symm(basis, auxbasis, slice=[0, nbf, 0, nbf, 0, auxbasis.bfs_nao],
                                   schwarz=True, threshold_schwarz=1e-9)
    cpu = contract('ijP,ij->P', ints, dmat)
    gpu = Integrals.rys_3c2e_gamma_contract_cupy(basis, auxbasis, dmat, threshold_schwarz=1e-9)
    np.testing.assert_allclose(gpu, cpu, rtol=0, atol=1e-11 * max(np.abs(cpu).max(), 1.0))


def test_one_electron_grad_contractions(system):
    """The overlap and kinetic bra-center r-gradients, contracted the way DFT_Grad contracts them."""
    _, basis, _, dmat, _ = system
    for cpu_fn, gpu_fn in ((Integrals.overlap_mat_grad_r_symm, Integrals.overlap_mat_grad_r_symm_cupy),
                           (Integrals.kin_mat_grad_r_symm, Integrals.kin_mat_grad_r_symm_cupy)):
        cpu = contract('dij,ij->id', cpu_fn(basis), dmat)
        gpu = cp.asnumpy((gpu_fn(basis) * cp.asarray(dmat)).sum(axis=2).T)
        np.testing.assert_allclose(gpu, cpu, rtol=0, atol=1e-11 * max(np.abs(cpu).max(), 1.0))


@pytest.mark.parametrize('funcid', [[1, 7], [101, 130]], ids=['lda', 'pbe'])
@pytest.mark.parametrize('grid_response', [False, True], ids=['fixed_grid', 'grid_response'])
def test_xc_grad(system, funcid, grid_response):
    mol, basis, _, _, _ = system
    # A positive-semidefinite density matrix: the functionals are only defined for rho >= 0.
    rng = np.random.default_rng(7)
    c = rng.standard_normal((basis.bfs_nao, 5))
    dmat = np.ascontiguousarray(c @ c.T / basis.bfs_nao)

    grids = Grids(mol, basis=basis, level=3, verbose=False)
    coords = np.asarray(grids.coords)
    weights = np.asarray(grids.weights)
    blocksize = 20480
    nblocks = coords.shape[0] // blocksize
    lnz, cnz = Integrals.bf_val_helpers.nonzero_ao_indices(basis, coords, blocksize, nblocks,
                                                          coords.shape[0])
    kwargs = dict(funcid=funcid, use_libxc=False, blocksize=blocksize,
                  list_nonzero_indices=lnz, count_nonzero_indices=cnz,
                  grids=grids, grid_response=grid_response)
    cpu = Integrals.eval_xc_grad_2(basis, dmat, weights, coords, ncores=2, **kwargs)
    gpu = Integrals.eval_xc_grad_2_cupy(basis, dmat, weights, coords, **kwargs)
    if not grid_response:
        cpu, gpu = (cpu,), (gpu,)
    for reference, device in zip(cpu, gpu):
        np.testing.assert_allclose(device, reference, rtol=0,
                                   atol=1e-10 * max(np.abs(reference).max(), 1.0))


@pytest.mark.parametrize('sao', [False, True], ids=['cao', 'sao'])
def test_dft_grad_end_to_end(sao):
    """The assembled gradient: one converged SCF, both devices, everything but the ECP term.

    Worth having on top of the per-routine tests because it also covers the plumbing -- the shared
    stream, the fitting-coefficient solve, the per-atom scatters and the spherical-basis branch.
    """
    from pyfock import DFT, DFT_Grad

    mol = Mol(coordfile=str(H2O_XYZ))
    basis = Basis(mol, {'all': Basis.load(mol=mol, basis_name='def2-SVP')})
    auxbasis = Basis(mol, {'all': Basis.load(mol=mol, basis_name=AUX_BASIS)})
    dft = DFT(mol, basis, auxbasis, xc=[101, 130], grids=Grids(mol, level=3, verbose=False))
    dft.conv_crit = 1e-9
    dft.max_itr = 50
    dft.sao = sao
    dft.use_gpu = True
    dft.isDF = True
    dft.scf()

    gpu = DFT_Grad(dft, verbose=False, use_gpu=True).calculate()
    cpu = DFT_Grad(dft, verbose=False, use_gpu=False).calculate()
    for term in cpu['gradient_components']:
        np.testing.assert_allclose(gpu['gradient_components'][term],
                                   cpu['gradient_components'][term], rtol=0, atol=1e-9)
    np.testing.assert_allclose(gpu['gradient'], cpu['gradient'], rtol=0, atol=1e-9)
    assert np.isfinite(gpu['gradient']).all()
