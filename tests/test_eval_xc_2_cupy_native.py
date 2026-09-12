"""Native XC support required by the DF CPU/GPU SCF comparison."""
import numpy as np
import pytest

cp = pytest.importorskip('cupy')
from numba import cuda
pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not cuda.is_available(), reason='CUDA unavailable')]
from pyfock import Mol, Basis, Integrals


@pytest.mark.parametrize('ids', [[1, 7], [101, 130]])
def test_native_hybrid_xc_matches_cpu(ids):
    mol = Mol(atoms=[['He', 0., 0., 0.]])
    basis = Basis(mol, {'all': Basis.load(mol=mol, basis_name='def2-SVP')})
    rng = np.random.default_rng(42)
    coords = rng.uniform(-1., 1., (7, 3))
    weights = rng.uniform(0.1, 0.2, 7)
    dmat = np.eye(basis.bfs_nao) * 0.1
    indices = [cp.arange(basis.bfs_nao) for _ in range(3)]
    counts = np.full(3, basis.bfs_nao)
    ref_e, ref_v = Integrals.eval_xc_2(basis, dmat, weights, coords, ids, False,
                                       ncores=1, blocksize=3)
    e, v = Integrals.eval_xc_2_cupy(basis, cp.asarray(dmat), weights, coords, ids,
             ncores=1, blocksize=3, use_libxc=False,
             list_nonzero_indices=indices, count_nonzero_indices=counts)
    np.testing.assert_allclose(float(e), float(ref_e), atol=1e-12, rtol=0)
    np.testing.assert_allclose(cp.asnumpy(v), ref_v, atol=1e-12, rtol=0)
