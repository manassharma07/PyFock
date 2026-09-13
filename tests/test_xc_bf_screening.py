"""
Unit tests for the XC basis-function screening in
:mod:`pyfock.Integrals.bf_val_helpers` (``nonzero_ao_indices``) and for the
pre-packed basis data accepted by ``eval_bfs`` / ``eval_bfs_and_grad``.

``nonzero_ao_indices`` marks a basis function as significant for a block of
grid points when at least one point lies within the function's radial cutoff.
The parallel bounding-sphere kernel is compared with a plain NumPy distance
check and with the serial brute-force reference kernel on synthetic grids
built to hit all three code paths (block entirely outside a cutoff sphere,
entirely inside, and straddling it), on a PyFock DFT grid, for the empty and
the short trailing block, and across Numba thread counts.
"""
from __future__ import annotations

import numba
import numpy as np
import pytest

from pyfock import Basis, Grids, Mol
from pyfock.Integrals import bf_val_helpers as bfh


WATER = [["O", 0.0, 0.0, 0.117], ["H", 0.0, 0.757, -0.467], ["H", 0.0, -0.757, -0.467]]
TWO_WATERS = list(WATER) + [[s, x + 6.0, y, z] for s, x, y, z in WATER]


def _system(atoms, basis_name):
    mol = Mol(atoms=atoms)
    return mol, Basis(mol, {"all": Basis.load(mol=mol, basis_name=basis_name)})


def _blocks(coords, blocksize, nblocks, ngrids):
    return [coords[iblock * blocksize: min((iblock + 1) * blocksize, ngrids)] for iblock in range(nblocks + 1)]


def _reference_numpy(basis, coords, blocksize, nblocks, ngrids):
    """Independent oracle: any point of the block within the function's cutoff."""
    bfs_coords = np.asarray(basis.bfs_coords, dtype=np.float64)
    cutoff = np.asarray(basis.bfs_radius_cutoff, dtype=np.float64)
    out = []
    for block in _blocks(coords, blocksize, nblocks, ngrids):
        if block.shape[0] == 0:
            out.append(np.zeros(0, dtype=np.int64))
            continue
        dist = np.linalg.norm(block[:, None, :] - bfs_coords[None, :, :], axis=2)
        out.append(np.flatnonzero((dist < cutoff[None, :]).any(axis=0)))
    return out


def _reference_serial_kernel(basis, coords, blocksize, nblocks, ngrids):
    """The serial brute-force Numba kernel, block by block (previous implementation)."""
    bfs_coords = np.asarray(basis.bfs_coords, dtype=np.float64)
    cutoff = np.asarray(basis.bfs_radius_cutoff, dtype=np.float64)
    out = []
    for block in _blocks(coords, blocksize, nblocks, ngrids):
        indices, count = bfh.nonzero_ao_indices_batch(block, bfs_coords, cutoff)
        out.append(indices[:count].astype(np.int64))
    return out


def _sphere_branch_counts(basis, coords, blocksize, nblocks, ngrids):
    """How many (block, function) pairs the bounding-sphere test decides as
    'all outside' / 'all inside', and how many it leaves to the point scan."""
    bfs_coords = np.asarray(basis.bfs_coords, dtype=np.float64)
    cutoff = np.asarray(basis.bfs_radius_cutoff, dtype=np.float64)
    outside = inside = ambiguous = 0
    for block in _blocks(coords, blocksize, nblocks, ngrids):
        if block.shape[0] == 0:
            continue
        centre = block.mean(axis=0)
        rad = np.linalg.norm(block - centre, axis=1).max()
        dist_centre = np.linalg.norm(bfs_coords - centre, axis=1)
        outside += int(np.count_nonzero(dist_centre - rad > cutoff))
        inside += int(np.count_nonzero(dist_centre + rad < cutoff))
        ambiguous += int(np.count_nonzero(~(dist_centre - rad > cutoff) & ~(dist_centre + rad < cutoff)))
    return outside, inside, ambiguous


def _structured_grid(basis, rng, blocksize):
    """Blocks of exactly ``blocksize`` points that exercise every branch of the
    kernel: tight clusters at the atoms (inside the cutoffs of that atom's
    functions), a cluster far away (outside every cutoff), thin slabs through
    the molecule (straddling the cutoff spheres), points scattered over the
    whole box (a bounding sphere that decides nothing), a block with points at
    every centre (sees every function), and shells just inside and just outside
    the cutoff of individual functions."""
    bfs_coords = np.asarray(basis.bfs_coords, dtype=np.float64)
    cutoff = np.asarray(basis.bfs_radius_cutoff, dtype=np.float64)
    centres = np.unique(bfs_coords, axis=0)
    cmin, cmax = cutoff.min(), cutoff.max()
    blocks = []
    for centre in centres:
        blocks.append(centre + 0.1 * cmin * rng.uniform(-1.0, 1.0, size=(blocksize, 3)) / np.sqrt(3.0))
    far = centres.mean(axis=0) + np.array([3.0 * cmax + 10.0, 0.0, 0.0])
    blocks.append(far + rng.uniform(-0.5, 0.5, size=(blocksize, 3)))
    lo = centres.min(axis=0) - 1.5 * cmax
    hi = centres.max(axis=0) + 1.5 * cmax
    box = rng.uniform(lo, hi, size=(8 * blocksize, 3))
    blocks.extend(np.array_split(box[np.argsort(box[:, 0])], 8))
    blocks.append(rng.uniform(lo, hi, size=(blocksize, 3)))
    blocks.append(np.resize(centres, (blocksize, 3)) + 1.0e-3 * cmin * rng.uniform(-1.0, 1.0, size=(blocksize, 3)))
    for ibf in range(0, basis.bfs_nao, max(1, basis.bfs_nao // 5)):
        directions = rng.normal(size=(blocksize, 3))
        directions /= np.linalg.norm(directions, axis=1)[:, None]
        for factor in (1.0 - 1.0e-3, 1.0 + 1.0e-3):
            blocks.append(bfs_coords[ibf] + factor * cutoff[ibf] * directions)
    return np.concatenate(blocks)


def _check_result(basis, coords, blocksize, nblocks, ngrids):
    result, counts = bfh.nonzero_ao_indices(basis, coords, blocksize, nblocks, ngrids)
    ref_np = _reference_numpy(basis, coords, blocksize, nblocks, ngrids)
    ref_serial = _reference_serial_kernel(basis, coords, blocksize, nblocks, ngrids)
    assert len(result) == len(counts) == nblocks + 1
    for iblock in range(nblocks + 1):
        assert result[iblock].dtype == np.uint16
        assert isinstance(counts[iblock], int) and counts[iblock] == result[iblock].shape[0]
        assert np.array_equal(result[iblock].astype(np.int64), ref_np[iblock]), iblock
        assert np.array_equal(result[iblock].astype(np.int64), ref_serial[iblock]), iblock
    return result, counts


@pytest.mark.parametrize("basis_name", ["def2-SVP", "def2-TZVP"])
def test_structured_grid_matches_references(basis_name):
    _, basis = _system(TWO_WATERS, basis_name)
    rng = np.random.default_rng(7)
    blocksize = 64
    coords = _structured_grid(basis, rng, blocksize)
    ngrids = coords.shape[0]
    assert ngrids % blocksize == 0  # exact multiple: the trailing block is empty
    nblocks = ngrids // blocksize
    outside, inside, ambiguous = _sphere_branch_counts(basis, coords, blocksize, nblocks, ngrids)
    assert outside > 0 and inside > 0 and ambiguous > 0  # all three kernel branches are exercised
    result, counts = _check_result(basis, coords, blocksize, nblocks, ngrids)
    assert counts[-1] == 0 and result[-1].shape[0] == 0
    assert max(counts) == basis.bfs_nao  # the block with points at every centre sees every function
    assert min(counts[:-1]) == 0  # the far-away block sees none

    # Short trailing block: drop a few points so ngrids is not a multiple of blocksize
    coords_short = coords[:-5]
    ngrids_short = coords_short.shape[0]
    _check_result(basis, coords_short, blocksize, ngrids_short // blocksize, ngrids_short)


def test_dft_grid_matches_references():
    mol, basis = _system(TWO_WATERS, "def2-SVP")
    grid_basis = Basis(mol, {"all": Basis.load(mol=mol, basis_name="def2-QZVP")})
    grids = Grids(mol, basis=grid_basis, level=3, ncores=2)
    coords = np.asarray(grids.coords, dtype=np.float64)
    ngrids = coords.shape[0]
    blocksize = 5000
    nblocks = ngrids // blocksize
    assert nblocks >= 4
    result, counts = _check_result(basis, coords, blocksize, nblocks, ngrids)
    assert 0 < np.mean(counts[:-1] if ngrids % blocksize == 0 else counts) <= basis.bfs_nao


def test_result_independent_of_thread_count():
    _, basis = _system(TWO_WATERS, "def2-SVP")
    coords = _structured_grid(basis, np.random.default_rng(11), 32)
    ngrids = coords.shape[0]
    nblocks = ngrids // 32
    saved = numba.get_num_threads()
    try:
        results = []
        for nthreads in (1, min(4, numba.config.NUMBA_NUM_THREADS)):
            numba.set_num_threads(nthreads)
            results.append(bfh.nonzero_ao_indices(basis, coords, 32, nblocks, ngrids))
    finally:
        numba.set_num_threads(saved)
    (lst_a, cnt_a), (lst_b, cnt_b) = results
    assert cnt_a == cnt_b
    assert all(np.array_equal(a, b) for a, b in zip(lst_a, lst_b))


def test_uint16_limit_is_checked():
    class FakeBasis:
        bfs_nao = 2 ** 16 + 1
        bfs_coords = np.zeros((bfs_nao, 3))
        bfs_radius_cutoff = np.ones(bfs_nao)

    with pytest.raises(ValueError, match="uint16"):
        bfh.nonzero_ao_indices(FakeBasis(), np.zeros((10, 3)), 5, 2, 10)


def test_eval_bfs_prepacked_basis_matches_default():
    _, basis = _system(WATER, "def2-TZVP")  # includes d and f functions
    rng = np.random.default_rng(3)
    coords = rng.uniform(-3.0, 3.0, size=(50, 3))
    packed = bfh.pack_bfs_data(basis)
    assert len(packed) == 8
    assert packed[0].shape == (basis.bfs_nao, 3) and packed[4].shape == (basis.bfs_nao, max(basis.bfs_nprim))
    indices = np.arange(0, basis.bfs_nao, 3, dtype=np.uint16)

    dense = bfh.eval_bfs(basis, coords)
    assert np.array_equal(dense, bfh.eval_bfs(basis, coords, bfs_data=packed))
    sparse = bfh.eval_bfs(basis, coords, non_zero_indices=indices)
    assert np.array_equal(sparse, bfh.eval_bfs(basis, coords, non_zero_indices=indices, bfs_data=packed))
    # The dense kernel zeroes values beyond the radial cutoff; the sparse one evaluates them exactly.
    assert np.allclose(sparse, dense[:, indices], rtol=1.0e-12, atol=1.0e-6)

    values, grads = bfh.eval_bfs_and_grad(basis, coords, non_zero_indices=indices)
    values_p, grads_p = bfh.eval_bfs_and_grad(basis, coords, non_zero_indices=indices, bfs_data=packed)
    assert np.array_equal(values, values_p) and np.array_equal(grads, grads_p)
    assert grads.shape == (3, 50, indices.shape[0])
    assert np.allclose(values, sparse, rtol=1.0e-12, atol=1.0e-14)
