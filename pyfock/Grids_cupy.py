__all__ = ["gpu_available", "atomic_grids_cupy", "becke_partition_weights_cupy",
           "box_grouping_order_cupy", "build_treutler_grid_cupy", "MAX_GPU_ATOMS",
           "THREADS_PER_BLOCK", "GridsGPUError"]
# Grids_cupy.py
# Author: Manas Sharma (manassharma07@live.com)
# This is a part of CrysX (https://bragitoff.com/crysx)
"""GPU generation of the native ('treutler') XC grids of :mod:`pyfock.Grids`.

The three steps of a grid build run on the device:

1. **Assembly of the atomic grids** (:func:`atomic_grids_cupy`): only the distinct element templates
   (the single-atom grids of :func:`pyfock.Grids.single_atom_grid`, which are cached on the host) cross
   the bus. The molecular point set is built on the device by a gather from that pool, so nothing of
   size N is transferred or allocated on the host.
2. **Becke partitioning** (:func:`becke_partition_weights_cupy`): a CUDA kernel that mirrors the Numba
   CPU kernel pair for pair, one thread per grid point. Every thread keeps the distances to all atoms
   and their unnormalised cell functions in local memory, so the kernel is compiled for the few numbers
   of atoms in ``_MAX_ATOMS_VARIANTS`` and the smallest variant that fits the molecule is launched.
   This step dominates the cost of a build (N_points x N_atoms^2).
3. **Box grouping** (:func:`box_grouping_order_cupy`): the same 1.2 Bohr boxes as the host path, ordered
   by a stable device sort.

The finished grid is copied back to the host, so the rest of PyFock sees the same NumPy arrays as the
CPU path, and the two agree: the points, their atom indices and their box order are bit for bit
identical. Individual weights are not quite, because NVVM contracts multiply-adds into FMAs and that
moves ``nu`` by an ulp; where Becke's cutoff profile saturates (``0.5 * (1 - g)`` with ``g`` within an
ulp of 1) the ulp is amplified, so a handful of points on a cell boundary differ in a late digit of an
already negligible weight (up to 1e-13 in absolute value below a hundred atoms; 83 of the 5.2 million
points of olestra, 453 atoms, by up to 1.7e-8, which is 1.5e-10 of the largest weight there). The GPU
value is the more accurate one (one rounding instead of two), and it integrates the same: the volume
of the fuzzy cells agrees to 9e-14 relative and quadratures of nucleus-centred Gaussians, diffuse and
sharp, come out bit for bit equal even for olestra.
"""
import math
import warnings

import numpy as np
import numba
from numba import cuda
from numba.core.errors import NumbaPerformanceWarning

try:
    import cupy as cp
except Exception:
    # Handle the case when Cupy is not installed
    cp = None

from .Grids import single_atom_grid, size_adjustment_table


class GridsGPUError(RuntimeError):
    """Raised when a grid cannot be built on the GPU (no CUDA device, or too many atoms)."""


def gpu_available():
    """Whether CuPy and a CUDA device are available for the grid kernels."""
    if cp is None:
        return False
    try:
        return cuda.is_available() and cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


# ---------------------------------------------------------------------------------------------------------------
# Becke partitioning
# ---------------------------------------------------------------------------------------------------------------
@cuda.jit(device=True, inline=True)
def _becke_point(p, coords, atom_idx, atm_coords, inv_dist, a_table, use_adjust, d, P, out):
    # Becke's fuzzy-cell weight of one grid point for the atom it belongs to, evaluated exactly as in the
    # CPU kernel ``pyfock.Grids._becke_partition_kernel`` (all atom pairs, no screening):
    #   P_A(r) / sum_B P_B(r),  P_A = prod_{B != A} s(nu_AB),  s(nu) = (1 - f(f(f(nu))))/2,  f(x) = (3x - x^3)/2,
    #   nu_AB = mu_AB + a_AB (1 - mu_AB^2),  mu_AB = (|r - R_A| - |r - R_B|) / |R_A - R_B|.
    # ``d`` and ``P`` are the thread's scratch arrays of at least ``natm`` elements.
    natm = atm_coords.shape[0]
    x = coords[p, 0]
    y = coords[p, 1]
    z = coords[p, 2]
    for k in range(natm):
        dx = x - atm_coords[k, 0]
        dy = y - atm_coords[k, 1]
        dz = z - atm_coords[k, 2]
        d[k] = math.sqrt(dx * dx + dy * dy + dz * dz)
        P[k] = 1.0
    for i in range(natm):
        di = d[i]
        pi = P[i]   # P[i] is untouched by the iterations before this one; keep it in a register
        for j in range(i):
            g = (di - d[j]) * inv_dist[i, j]
            if use_adjust:
                g += a_table[i, j] * (1.0 - g * g)
            g = (3.0 - g * g) * g * 0.5
            g = (3.0 - g * g) * g * 0.5
            g = (3.0 - g * g) * g * 0.5
            pi *= 0.5 * (1.0 - g)
            P[j] *= 0.5 * (1.0 + g)
        P[i] = pi
    s = 0.0
    for k in range(natm):
        s += P[k]
    out[p] = P[atom_idx[p]] / s


@cuda.jit(cache=True)
def _becke_kernel_32(coords, atom_idx, atm_coords, inv_dist, a_table, use_adjust, out):
    p = cuda.grid(1)
    d = cuda.local.array(32, numba.float64)
    P = cuda.local.array(32, numba.float64)
    if p < coords.shape[0]:
        _becke_point(p, coords, atom_idx, atm_coords, inv_dist, a_table, use_adjust, d, P, out)


@cuda.jit(cache=True)
def _becke_kernel_128(coords, atom_idx, atm_coords, inv_dist, a_table, use_adjust, out):
    p = cuda.grid(1)
    d = cuda.local.array(128, numba.float64)
    P = cuda.local.array(128, numba.float64)
    if p < coords.shape[0]:
        _becke_point(p, coords, atom_idx, atm_coords, inv_dist, a_table, use_adjust, d, P, out)


@cuda.jit(cache=True)
def _becke_kernel_512(coords, atom_idx, atm_coords, inv_dist, a_table, use_adjust, out):
    p = cuda.grid(1)
    d = cuda.local.array(512, numba.float64)
    P = cuda.local.array(512, numba.float64)
    if p < coords.shape[0]:
        _becke_point(p, coords, atom_idx, atm_coords, inv_dist, a_table, use_adjust, d, P, out)


@cuda.jit(cache=True)
def _becke_kernel_2048(coords, atom_idx, atm_coords, inv_dist, a_table, use_adjust, out):
    p = cuda.grid(1)
    d = cuda.local.array(2048, numba.float64)
    P = cuda.local.array(2048, numba.float64)
    if p < coords.shape[0]:
        _becke_point(p, coords, atom_idx, atm_coords, inv_dist, a_table, use_adjust, d, P, out)


# (number of atoms the kernel fits, kernel), smallest first. Only the first natm elements of the local
# arrays are touched, so an oversized variant costs no time, but the driver reserves the backing store of
# the local memory for the whole device (16 bytes per atom of the variant and resident thread, about
# 0.1 GB for the 32-atom kernel and 1.8 GB for the 2048-atom one): take the smallest that fits.
_MAX_ATOMS_VARIANTS = ((32, _becke_kernel_32), (128, _becke_kernel_128),
                       (512, _becke_kernel_512), (2048, _becke_kernel_2048))

THREADS_PER_BLOCK = 128
"""Threads per block of the partition kernels (the kernel is insensitive to this between 32 and 256)."""

MAX_GPU_ATOMS = _MAX_ATOMS_VARIANTS[-1][0]
"""Largest number of atoms the CUDA partition kernels are compiled for."""


def _select_becke_kernel(natm):
    for max_atoms, kernel in _MAX_ATOMS_VARIANTS:
        if natm <= max_atoms:
            return kernel
    raise GridsGPUError('The GPU grid kernels are compiled for at most ' + str(MAX_GPU_ATOMS)
                        + ' atoms, this molecule has ' + str(natm) + '.')


def _stream_pair(cp_stream):
    """Return ``(cupy stream, numba external stream)`` for the given (or the current) CuPy stream."""
    if cp_stream is None:
        cp_stream = cp.cuda.get_current_stream()
    return cp_stream, cuda.external_stream(cp_stream.ptr)


def becke_partition_weights_cupy(coords, atom_idx, atm_coords, a_table=None, cp_stream=None):
    """Becke partitioning factors P_A(r)/sum_B P_B(r) of grid points belonging to the atoms ``atom_idx``.

    The GPU counterpart of :func:`pyfock.Grids.becke_partition_weights`: ``coords`` (N, 3) and
    ``atom_idx`` (N,) may be host or device arrays, ``atm_coords`` (natoms, 3) and the atomic-size
    adjustment ``a_table`` are small and read on the host. Returns a CuPy array of length N.

    Raises
    ------
    GridsGPUError
        If the molecule has more than :data:`MAX_GPU_ATOMS` atoms.
    """
    if cp is None:
        raise GridsGPUError('CuPy is not installed; the XC grids cannot be built on the GPU.')
    atm_coords = np.ascontiguousarray(np.asarray(atm_coords, dtype=np.float64).reshape(-1, 3))
    natm = atm_coords.shape[0]
    d_coords = cp.ascontiguousarray(cp.asarray(coords, dtype=cp.float64)).reshape(-1, 3)
    npts = int(d_coords.shape[0])
    out = cp.ones(npts, dtype=cp.float64)
    if natm < 2 or npts == 0:
        return out
    kernel = _select_becke_kernel(natm)

    # The same small host-side tables as the CPU path
    diff = atm_coords[:, None, :] - atm_coords[None, :, :]
    dist = np.sqrt(np.einsum('ijk,ijk->ij', diff, diff))
    inv_dist = np.zeros((natm, natm))
    off = ~np.eye(natm, dtype=bool)
    inv_dist[off] = 1.0 / dist[off]
    if a_table is None:
        a_table = np.zeros((natm, natm))
        use_adjust = False
    else:
        a_table = np.ascontiguousarray(a_table, dtype=np.float64)
        use_adjust = True

    d_atom_idx = cp.ascontiguousarray(cp.asarray(atom_idx, dtype=cp.int64))
    d_atm = cp.asarray(atm_coords)
    d_inv = cp.asarray(inv_dist)
    d_a = cp.asarray(a_table)

    cp_stream, nb_stream = _stream_pair(cp_stream)
    threads = THREADS_PER_BLOCK
    blocks = (npts + threads - 1) // threads
    with warnings.catch_warnings():
        # a small molecule has too few points to fill the device; nothing the caller can do about it
        warnings.simplefilter('ignore', NumbaPerformanceWarning)
        kernel[blocks, threads, nb_stream](d_coords, d_atom_idx, d_atm, d_inv, d_a, use_adjust, out)
    cp_stream.synchronize()
    return out


# ---------------------------------------------------------------------------------------------------------------
# Assembly of the single-atom grids
# ---------------------------------------------------------------------------------------------------------------
def atomic_grids_cupy(atm_coords, charges, level=3, pruning='regions', overrides=None):
    """Join the single-atom grids of all atoms into device arrays, in the order of the CPU path.

    Returns ``(coords, vol, atom_idx)`` as CuPy arrays: the points of every atom translated to its
    nucleus (Bohr), their volume elements ``4 pi r^2 dr w_ang`` and the index of the atom each point
    belongs to. ``overrides`` is ``{charge: (n_rad, n_ang)}``. Only the distinct element templates are
    uploaded; the molecular arrays are gathered from them on the device.
    """
    if cp is None:
        raise GridsGPUError('CuPy is not installed; the XC grids cannot be built on the GPU.')
    overrides = overrides or {}
    charges = np.asarray(charges, dtype=np.int64)
    atm_coords = np.ascontiguousarray(np.asarray(atm_coords, dtype=np.float64).reshape(-1, 3))
    natm = charges.shape[0]

    template_index = {}
    template_coords = []
    template_vol = []
    index_of_atom = np.empty(natm, dtype=np.int64)
    for ia in range(natm):
        z = int(charges[ia])
        n_rad, n_ang = overrides.get(z, (None, None))
        key = (z, n_rad, n_ang)
        index = template_index.get(key)
        if index is None:
            c, v = single_atom_grid(z, level=level, pruning=pruning, n_rad=n_rad, n_ang=n_ang)
            index = len(template_coords)
            template_index[key] = index
            template_coords.append(c)
            template_vol.append(v)
        index_of_atom[ia] = index

    template_sizes = np.array([v.shape[0] for v in template_vol], dtype=np.int64)
    template_start = np.concatenate(([0], np.cumsum(template_sizes)))[:-1]
    pool_coords = cp.asarray(np.vstack(template_coords))
    pool_vol = cp.asarray(np.hstack(template_vol))

    counts = template_sizes[index_of_atom]
    ends = np.cumsum(counts)
    npts = int(ends[-1]) if natm else 0
    starts = ends - counts
    # point p belongs to the atom whose block contains it, and to the template point at the same offset
    atom_idx = cp.searchsorted(cp.asarray(ends), cp.arange(npts, dtype=cp.int64), side='right')
    shift = cp.asarray(template_start[index_of_atom] - starts)
    src = cp.arange(npts, dtype=cp.int64) + shift[atom_idx]
    coords = pool_coords[src] + cp.asarray(atm_coords)[atom_idx]
    vol = pool_vol[src]
    return coords, vol, atom_idx


# ---------------------------------------------------------------------------------------------------------------
# Box grouping
# ---------------------------------------------------------------------------------------------------------------
def box_grouping_order_cupy(atm_coords, coords, box_size=1.2, boundary_penalty=4.2):
    """Permutation grouping grid points into cubic boxes of edge ``box_size`` Bohr, computed on the device.

    The GPU counterpart of :func:`pyfock.Grids.box_grouping_order`; ``coords`` may be a host or a device
    array and the returned permutation is a CuPy one. The box index of a point is evaluated in the same
    order as on the host, and the sort is stable, so the permutation is the same as the CPU one.
    """
    if cp is None:
        raise GridsGPUError('CuPy is not installed; the XC grids cannot be built on the GPU.')
    atm_coords = np.asarray(atm_coords, dtype=np.float64).reshape(-1, 3)
    coords = cp.asarray(coords, dtype=cp.float64).reshape(-1, 3)
    lower = atm_coords.min(axis=0) - boundary_penalty
    upper = atm_coords.max(axis=0) + boundary_penalty
    boxes = np.maximum(((upper - lower) * (1.0 / box_size)).round().astype(np.int64), 1)
    box_size = (upper - lower) / boxes
    shifted = coords - cp.asarray(lower)
    scaled = shifted * cp.asarray(1.0 / box_size)
    box_ids = cp.floor(scaled).astype(cp.int64)
    box_ids[box_ids < -1] = -1
    d_boxes = cp.asarray(boxes)
    box_ids = cp.minimum(box_ids, d_boxes)
    # linear box index with x as the slowest index (same order as sorting the (ix, iy, iz) triples)
    lin = ((box_ids[:, 0] + 1) * (int(boxes[1]) + 2) + (box_ids[:, 1] + 1)) * (int(boxes[2]) + 2) + (box_ids[:, 2] + 1)
    # stable sort: ties keep their original order, as numpy's kind='stable' does
    return cp.lexsort(cp.stack((cp.arange(lin.shape[0], dtype=cp.int64), lin)))


# ---------------------------------------------------------------------------------------------------------------
# Complete build
# ---------------------------------------------------------------------------------------------------------------
def build_treutler_grid_cupy(atm_coords, charges, level=3, pruning='regions', size_adjustment='treutler',
                             overrides=None, sort=True, box_size=1.2, cp_stream=None, to_host=True):
    """Build a complete native ('treutler') molecular grid on the GPU.

    Parameters are those of the CPU path (:meth:`pyfock.Grids.Grids._build_treutler`): ``overrides`` is
    ``{charge: (n_rad, n_ang)}`` and ``sort`` groups the points into ``box_size`` Bohr boxes.
    Returns ``(coords, weights, atom_idx)`` as NumPy arrays, or as CuPy arrays for ``to_host=False``.

    Raises
    ------
    GridsGPUError
        If CuPy or a CUDA device is unavailable, or the molecule has more than :data:`MAX_GPU_ATOMS` atoms.
    """
    if cp is None:
        raise GridsGPUError('CuPy is not installed; the XC grids cannot be built on the GPU.')
    if not gpu_available():
        raise GridsGPUError('No CUDA device is available; the XC grids cannot be built on the GPU.')
    charges = np.asarray(charges, dtype=np.int64)
    if charges.shape[0] > MAX_GPU_ATOMS:
        raise GridsGPUError('The GPU grid kernels are compiled for at most ' + str(MAX_GPU_ATOMS)
                            + ' atoms, this molecule has ' + str(charges.shape[0]) + '.')
    coords, vol, atom_idx = atomic_grids_cupy(atm_coords, charges, level=level, pruning=pruning,
                                              overrides=overrides)
    a_table = size_adjustment_table(charges, size_adjustment)
    weights = vol * becke_partition_weights_cupy(coords, atom_idx, atm_coords, a_table, cp_stream=cp_stream)
    if sort and coords.shape[0] > 0:
        perm = box_grouping_order_cupy(atm_coords, coords, box_size=box_size)
        coords = coords[perm]
        weights = weights[perm]
        atom_idx = atom_idx[perm]
    if to_host:
        return (np.ascontiguousarray(cp.asnumpy(coords)), np.ascontiguousarray(cp.asnumpy(weights)),
                np.ascontiguousarray(cp.asnumpy(atom_idx)))
    return cp.ascontiguousarray(coords), cp.ascontiguousarray(weights), cp.ascontiguousarray(atom_idx)
