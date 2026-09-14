"""Hardware tests of the GPU generation of the native ('treutler') XC grids.

The CPU build of ``pyfock.Grids`` is the oracle: for every option of the scheme the GPU must give the
same points, in the same box order, with the same atom indices and with weights that agree to the last
bits. The pieces (atomic-grid assembly, Becke partitioning, box grouping) are checked on their own too,
and so is the fallback to the CPU when the GPU cannot be used.
"""
import contextlib
import io
import sys

import numpy as np
import pytest

cp = pytest.importorskip('cupy')
from numba import cuda

try:
    GPU_AVAILABLE = cuda.is_available() and cp.cuda.runtime.getDeviceCount() > 0
except cp.cuda.runtime.CUDARuntimeError:
    GPU_AVAILABLE = False
pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not GPU_AVAILABLE, reason='CUDA device unavailable')]

from pyfock import Grids, Mol
from pyfock import Grids_cupy as GC

GridsMod = sys.modules['pyfock.Grids']

from tests.conftest import H2O_XYZ, ROOT  # noqa: E402

SNCL4_XYZ = ROOT / 'benchmarks_tests' / 'SnCl4.xyz'
DECANE_XYZ = ROOT / 'benchmarks_tests' / 'Decane_C10H22.xyz'

# The Becke cutoff profile saturates: `0.5 * (1 - g)` with `g` within an ulp of 1 turns the ulp the FMA
# contraction of NVVM costs into a large relative change of an utterly negligible weight. Absolute
# agreement is therefore the meaningful check, and the sum of the weights has to be reproduced exactly.
WEIGHT_ATOL = 1e-11


def _molecule(xyz):
    return Mol(coordfile=str(xyz))


def _assert_same_grid(cpu, gpu):
    assert gpu.use_gpu and not cpu.use_gpu
    assert np.array_equal(cpu.coords, gpu.coords)
    assert np.array_equal(cpu.atom_idx, gpu.atom_idx)
    assert cpu.is_sorted == gpu.is_sorted
    assert cpu.element_points == gpu.element_points
    np.testing.assert_allclose(gpu.weights, cpu.weights, atol=WEIGHT_ATOL, rtol=0)
    assert abs(gpu.weights.sum() - cpu.weights.sum()) <= 1e-12 * abs(cpu.weights.sum())


@pytest.mark.parametrize('xyz, level', [(H2O_XYZ, 1), (H2O_XYZ, 3), (H2O_XYZ, 5),
                                        (SNCL4_XYZ, 3), (DECANE_XYZ, 3)])
def test_gpu_grid_matches_the_cpu_grid(xyz, level):
    mol = _molecule(xyz)
    _assert_same_grid(Grids(mol, level=level, ncores=2, verbose=False),
                      Grids(mol, level=level, ncores=2, verbose=False, use_gpu=True))


@pytest.mark.parametrize('kwargs', [
    {'pruning': None},
    {'size_adjustment': 'becke'},
    {'size_adjustment': None},
    {'sort': False},
    {'points_per_element': {'O': (60, 194), 1: (40, 110)}},
])
def test_gpu_grid_matches_the_cpu_grid_for_every_option(kwargs):
    mol = _molecule(H2O_XYZ)
    _assert_same_grid(Grids(mol, level=3, ncores=2, verbose=False, **kwargs),
                      Grids(mol, level=3, ncores=2, verbose=False, use_gpu=True, **kwargs))


def test_the_two_grids_integrate_the_same():
    """The FMA-level weight differences must not move a quadrature: the volume of the fuzzy cells and
    Gaussians sitting on the nuclei (where a density lives) come out the same on both grids."""
    mol = _molecule(DECANE_XYZ)
    cpu = Grids(mol, level=3, ncores=2, verbose=False)
    gpu = Grids(mol, level=3, ncores=2, verbose=False, use_gpu=True)
    assert abs(gpu.weights.sum() - cpu.weights.sum()) <= 1e-13 * abs(cpu.weights.sum())
    atm_coords = np.asarray(mol.coordsBohrs).reshape(-1, 3)
    for alpha in (1.0, 20.0):   # a diffuse and a sharp integrand, each integrating to the atom count
        f = np.zeros(cpu.size)
        for centre in atm_coords:
            f += (alpha / np.pi) ** 1.5 * np.exp(-alpha * ((cpu.coords - centre) ** 2).sum(axis=1))
        reference = float(cpu.weights @ f)
        assert abs(reference - mol.natoms) < 1e-3          # the grid does integrate the Gaussians
        assert abs(float(gpu.weights @ f) - reference) <= 1e-13 * abs(reference)


def test_gpu_grid_matches_the_cpu_grid_with_ghost_atoms():
    mol = Mol(atoms=[['O', 0., 0., 0.], ['H', 0., 0., 1.8], ['Ghost-O', 0., 1.5, 1.0]])
    cpu = Grids(mol, level=2, ncores=2, verbose=False)
    gpu = Grids(mol, level=2, ncores=2, verbose=False, use_gpu=True)
    assert cpu.element_points == {'O': (60, 302), 'H': (40, 194), 'Ghost': (40, 194)}
    _assert_same_grid(cpu, gpu)


def test_grid_pruning_mask_matches_the_cpu():
    """The density pruning of ``DFT.scf`` must keep exactly the same points on both back ends."""
    from pyfock import Basis, DFT
    mol = _molecule(DECANE_XYZ)
    basis = Basis(mol, {"all": Basis.load(mol=mol, basis_name="def2-SVP")})
    grids = Grids(mol, level=3, ncores=2, verbose=False)
    cpu = DFT(mol, basis, xc=[101, 130], ncores=2, use_gpu=False)
    gpu = DFT(mol, basis, xc=[101, 130], ncores=2, use_gpu=True)
    with contextlib.redirect_stdout(io.StringIO()):
        dmat = cpu.guess_dmat(mol, basis)
    mask_cpu = cpu.grid_pruning_mask(grids, basis, dmat)
    mask_gpu = gpu.grid_pruning_mask(grids, basis, dmat)
    assert mask_cpu.dtype == mask_gpu.dtype == np.dtype(bool)
    assert 0 < int((~mask_cpu).sum()) < grids.size      # the pruning does drop points, but not all
    assert np.array_equal(mask_cpu, mask_gpu)
    # the block size only decides which basis functions are screened away per block
    assert np.array_equal(gpu.grid_pruning_mask(grids, basis, dmat, blocksize=10000), mask_gpu)


def test_atomic_grid_assembly_and_box_grouping_match_the_host():
    mol = _molecule(DECANE_XYZ)
    atm_coords = np.ascontiguousarray(np.asarray(mol.coordsBohrs, dtype=np.float64).reshape(-1, 3))
    charges = np.asarray(mol.Zcharges, dtype=np.int64)
    overrides = {6: (40, 110)}
    coords, vol, atom_idx = GC.atomic_grids_cupy(atm_coords, charges, level=2, overrides=overrides)
    ref_coords, ref_vol, ref_atom_idx = [], [], []
    for ia, z in enumerate(charges):
        n_rad, n_ang = overrides.get(int(z), (None, None))
        c, v = GridsMod.single_atom_grid(int(z), level=2, n_rad=n_rad, n_ang=n_ang)
        ref_coords.append(c + atm_coords[ia])
        ref_vol.append(v)
        ref_atom_idx.append(np.full(v.shape[0], ia, dtype=np.int64))
    assert np.array_equal(cp.asnumpy(coords), np.vstack(ref_coords))
    assert np.array_equal(cp.asnumpy(vol), np.hstack(ref_vol))
    assert np.array_equal(cp.asnumpy(atom_idx), np.hstack(ref_atom_idx))
    # the box grouping is the same permutation, ties included (a stable sort on both sides)
    assert np.array_equal(cp.asnumpy(GC.box_grouping_order_cupy(atm_coords, coords)),
                          GridsMod.box_grouping_order(atm_coords, cp.asnumpy(coords)))


def test_partition_weights_match_the_cpu_kernel_for_every_atom_count_variant():
    rng = np.random.default_rng(11)
    for natm in (1, 2, 33, 130):
        atm_coords = rng.normal(scale=4.0, size=(natm, 3))
        charges = rng.integers(1, 18, natm).astype(np.int64)
        coords = rng.normal(scale=6.0, size=(5000, 3))
        atom_idx = rng.integers(0, natm, 5000).astype(np.int64)
        for scheme in ('treutler', None):
            a_table = GridsMod.size_adjustment_table(charges, scheme)
            ref = GridsMod.becke_partition_weights(coords, atom_idx, atm_coords, a_table)
            got = cp.asnumpy(GC.becke_partition_weights_cupy(coords, atom_idx, atm_coords, a_table))
            np.testing.assert_allclose(got, ref, atol=1e-12, rtol=0)
        # device arrays are accepted as well
        got = GC.becke_partition_weights_cupy(cp.asarray(coords), cp.asarray(atom_idx), atm_coords)
        np.testing.assert_allclose(cp.asnumpy(got),
                                   GridsMod.becke_partition_weights(coords, atom_idx, atm_coords),
                                   atol=1e-12, rtol=0)
    # an empty grid and a single atom need no kernel at all
    assert GC.becke_partition_weights_cupy(np.zeros((0, 3)), np.zeros(0, dtype=np.int64),
                                           np.zeros((4, 3))).shape == (0,)
    assert float(GC.becke_partition_weights_cupy(np.zeros((3, 3)), np.zeros(3, dtype=np.int64),
                                                 np.zeros((1, 3))).sum()) == 3.0


@pytest.mark.parametrize('disable', ['atoms', 'cupy'])
def test_an_unusable_gpu_falls_back_to_the_cpu(monkeypatch, capsys, disable):
    mol = _molecule(H2O_XYZ)
    if disable == 'atoms':   # a molecule with more atoms than the kernels are compiled for
        monkeypatch.setattr(GC, '_MAX_ATOMS_VARIANTS', GC._MAX_ATOMS_VARIANTS[:0])
        monkeypatch.setattr(GC, 'MAX_GPU_ATOMS', 2)
    else:                    # CuPy not installed
        monkeypatch.setattr(GC, 'cp', None)
    grids = Grids(mol, level=1, ncores=2, verbose=False, use_gpu=True)
    assert not grids.use_gpu
    assert 'could not be built on the GPU' in capsys.readouterr().out
    reference = Grids(mol, level=1, ncores=2, verbose=False)
    assert np.array_equal(grids.coords, reference.coords)
    assert np.array_equal(grids.weights, reference.weights)
    assert grids.element_points == reference.element_points


def test_use_gpu_is_ignored_by_the_numgrid_scheme(capsys):
    mol = _molecule(H2O_XYZ)
    grids = Grids(mol, level=3, ncores=2, verbose=False, scheme='numgrid', use_gpu=True)
    assert not grids.use_gpu
    assert 'only available' in capsys.readouterr().out
    assert 'CPU' in Grids(mol, level=1, ncores=2, verbose=False).describe()
    assert 'GPU' in Grids(mol, level=1, ncores=2, verbose=False, use_gpu=True).describe()
