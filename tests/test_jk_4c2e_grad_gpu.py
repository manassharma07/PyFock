"""
Hardware tests of the four-center gradient on the GPU (:mod:`pyfock.Integrals.jk_4c2e_grad_cupy`),
the two-electron term of ``DFT_Grad`` without density fitting.

The CPU kernel (:mod:`pyfock.Integrals.jk_4c2e_grad`, validated against finite differences and
PySCF in ``tests/test_jk_4c2e_grad.py``) is the value oracle: HF, hybrid and pure-functional
weights, d and f shells, both integral schemes of the plan, a loose screening threshold, and the
assembled ``DFT_Grad`` gradients of a converged SCF on both devices.
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
from pyfock.Integrals import jk_4c2e, jk_4c2e_grad, jk_4c2e_grad_cupy


WATER = [['O', 0.0, 0.05, 0.117], ['H', 0.02, 0.757, -0.467], ['H', 0.0, -0.787, -0.427]]
# a second, distant water: quartets whose bra pair sits on one atom and ket pair on two (differentiated
# as (ZW|XY)), and pairs that the screening drops
TWO_WATERS = WATER + [[s, x + 3.0, y + 0.5, z] for s, x, y, z in WATER]


def _basis(atoms, name):
    mol = Mol(atoms=[list(a) for a in atoms])
    return mol, Basis(mol, {'all': Basis.load(mol=mol, basis_name=name)})


def _density(basis, seed=2):
    rng = np.random.default_rng(seed)
    C = 0.3 * rng.standard_normal((basis.bfs_nao, 5))
    return C @ C.T + 0.05 * Integrals.overlap_mat_symm(basis)


def _close(gpu, cpu, rel=1e-11):
    np.testing.assert_allclose(gpu, cpu, rtol=0, atol=rel * max(np.abs(cpu).max(), 1.0))


@pytest.mark.parametrize('name,exx,atoms', [('def2-SVP', 1.0, TWO_WATERS), ('def2-SVP', 0.25, TWO_WATERS),
                                            ('def2-SVP', 0.0, TWO_WATERS), ('def2-TZVP', 1.0, WATER)])
def test_grad_4c2e_matches_cpu(name, exx, atoms):
    _, basis = _basis(atoms, name)
    D = _density(basis)
    plan = jk_4c2e.build_plan(basis, threshold=1e-12)
    cpu = jk_4c2e_grad.grad_4c2e(plan, D, exx)
    gpu = jk_4c2e_grad_cupy.grad_4c2e_cupy(plan, D, exx)
    _close(gpu, cpu)
    assert np.abs(gpu.sum(axis=0)).max() < 1e-12 * np.abs(gpu).max()
    # a device density matrix, a loose threshold and the Obara-Saika plan give the same as on the CPU
    _close(jk_4c2e_grad_cupy.grad_4c2e_cupy(plan, cp.asarray(D), exx, threshold=1e-8),
           jk_4c2e_grad.grad_4c2e(plan, D, exx, threshold=1e-8))
    plan_os = jk_4c2e.build_plan(basis, threshold=1e-12, scheme='os')
    _close(jk_4c2e_grad_cupy.grad_4c2e_cupy(plan_os, D, exx), cpu)


@pytest.mark.parametrize('xc,sao,scf_gpu', [('HF', True, True), ('PBE0', False, False), ('PBE', True, True)])
def test_dft_grad_without_density_fitting_gpu_matches_cpu(xc, sao, scf_gpu):
    """From one converged SCF; a GPU SCF (four-center J and K on the host) defaults to the GPU gradient."""
    mol, basis = _basis(WATER, 'def2-SVP')
    dft = DFT(mol, basis, xc=xc, conv_crit=1e-9, ncores=2, gridsLevel=3, use_gpu=scf_gpu)
    dft.max_itr = 60
    dft.sao = sao
    dft.isDF = False
    dft.direct_scf = True
    with contextlib.redirect_stdout(io.StringIO()):
        dft.scf()
        assert dft.converged
        grad = DFT_Grad(dft, verbose=False, use_gpu=True if not scf_gpu else None)
        assert grad.use_gpu
        gpu = grad.calculate()
        cpu = DFT_Grad(dft, verbose=False, use_gpu=False).calculate()
    key = 'coulomb_4c2e' if xc == 'PBE' else 'coulomb_exchange_4c2e'
    assert key in gpu['gradient_components']
    for term in cpu['gradient_components']:
        np.testing.assert_allclose(gpu['gradient_components'][term], cpu['gradient_components'][term],
                                   rtol=0, atol=1e-9, err_msg=term)
    np.testing.assert_allclose(gpu['gradient'], cpu['gradient'], rtol=0, atol=1e-9)
