"""Multipole-accelerated fp64 density fitting on CUDA (algorithm 12).

This is the GPU counterpart of :mod:`~pyfock.Integrals.df_algo12_helpers`, built the same way
the algorithm-11 CUDA driver (:mod:`~pyfock.Integrals.df_algo11_helpers_cupy`) is built: the CPU
metadata builder is the single source of screening, branch geometry, near/far classification,
the profitability rule, the near-field column map, the far-field moments and their branch-centred
translations, and of the cache selection.  Nothing about *what* is computed changes on the GPU;
only *where*.

Near field
    the generated Rys kernels of algorithm 11 with their far-field mask switched on: a primitive
    pair is skipped for an auxiliary shell when its branch is far field for that shell.  Device
    items are the same ``(pair, auxiliary shell, column)`` int32 triples grouped by angular
    momentum, and the stored row blocks have the same layout - only the significant columns are
    fewer.  ``gamma`` reuses algorithm 11's ``gamma_items`` unchanged; ``J`` uses ``j_pairs_ff``,
    which adds the far-field row value so every matrix element is written exactly once.

Far field
    the pre-translated branch-centred row moments ``Mtil`` live on the device; the branch <->
    atom coupling runs in :mod:`~pyfock.Integrals.df_algo12_cuda_core` without materializing the
    dense ``Y``/``W`` intermediates of the CPU passes.

``max_memory_gb`` limits the cached near-field *device* values only (None = all, 0 = direct);
uncached pairs are re-evaluated in bounded batches in both contraction passes, exactly as in
algorithm 11.  ``low_memory=True`` (re-translating the group moments every iteration) is a CPU-only
option and is rejected here.  No global stream/device selection or memory-pool flushing is
performed; public operations synchronize their stream before returning.
"""
from timeit import default_timer as timer

import numpy as np
from numba import cuda, njit

try:
    import cupy as cp
except ImportError:
    cp = None

from .df_algo10_helpers import EXP_ARG_CUTOFF
from .df_algo11_cuda_core import project_items, gamma_items
from .df_algo11_helpers_cupy import _Batch, _group_items, _host
from .df_algo12_helpers import DFAlgo12Plan, _plan_metadata
from .df_algo12_cuda_core import (row_coefficients, branch_moments, row_far_field,
                                  branches_to_atoms, atoms_to_branches, shell_moments,
                                  gamma_far_field, j_pairs_ff, count_columns_ff)
from .rys_helpers_cuda import DATA_X, DATA_W

__all__ = ['DFAlgo12GPUPlan', 'build_plan_cupy', 'gamma_from_plan_cupy', 'J_from_plan_cupy']

# Shared-memory budget per block for the coupling kernels; 48 KB is available without the
# opt-in attribute on every architecture this driver supports.
_SHARED_LIMIT = 40000


class DFAlgo12GPUPlan(DFAlgo12Plan):
    """CPU-compatible plan metadata, device values, moments and reusable device work lists."""

    @property
    def moments_gb(self):
        """Far-field storage on the device (the host copies are released after the upload)."""
        return (self.device['Mtil'].nbytes + self.moment_bytes) / 1e9

    @property
    def temporary_memory_gb(self):
        return self.temporary_elements * 8e-9

    def summary(self):
        return (super().summary() + f'; GPU batch workspace {self.temporary_memory_gb:.3f} GB '
                '(additional to cached values)')

    def memory_stats(self):
        """Synchronized pool/runtime samples; peaks are sampled, not NVML process peaks."""
        self.stream.synchronize()
        with cp.cuda.Device(self.device_id):
            pool = cp.get_default_memory_pool()
            free, total = cp.cuda.runtime.memGetInfo()
            used, reserved = pool.used_bytes(), pool.total_bytes()
        self.peak_pool_used_bytes = max(self.peak_pool_used_bytes, used)
        self.peak_pool_reserved_bytes = max(self.peak_pool_reserved_bytes, reserved)
        return dict(cached_bytes=self.values.nbytes, moment_bytes=self.device['Mtil'].nbytes,
                    temporary_bytes=self.temporary_elements * 8,
                    pool_used_bytes=used, pool_reserved_bytes=reserved,
                    peak_pool_used_bytes=self.peak_pool_used_bytes,
                    peak_pool_reserved_bytes=self.peak_pool_reserved_bytes,
                    device_free_bytes=free, device_total_bytes=total)


# ----------------------------------------------------------------------------
# Host-side work lists
# ----------------------------------------------------------------------------
@njit(cache=True, nogil=True, fastmath=True, error_model='numpy')
def _pp_branch_dense(work, pair_I, pair_J, shell_off, bfs_nprim, bfs_expnts, bfs_coords,
                     pp_group, grp_branch, offsets, dropped):
    """
    Branch of primitive pair ``(ip, jp)`` of shell pair ``p`` at ``offsets[p] + ip*nprimB + jp``.

    The index is dense rather than a running count of the pairs that survive the
    Gaussian-product pre-screen, so the device never has to reproduce the host's screening
    decision bit for bit: a one-ulp disagreement there would otherwise shift every later branch
    of that shell pair.  Pairs the host drops are given ``dropped``, the row of the far-field
    mask that is true everywhere, so the kernel skips them whatever its own pre-screen decides.
    """
    out = np.full(offsets[offsets.shape[0] - 1], dropped, dtype=np.int32)
    for w in range(work.shape[0]):
        p = work[w]
        a0 = shell_off[pair_I[p]]
        b0 = shell_off[pair_J[p]]
        nprimB = bfs_nprim[b0]
        ijsq = ((bfs_coords[a0, 0] - bfs_coords[b0, 0]) ** 2
                + (bfs_coords[a0, 1] - bfs_coords[b0, 1]) ** 2
                + (bfs_coords[a0, 2] - bfs_coords[b0, 2]) ** 2)
        base = offsets[p]
        q = 0
        for ip in range(bfs_nprim[a0]):
            alpha = bfs_expnts[a0, ip]
            for jp in range(nprimB):
                beta = bfs_expnts[b0, jp]
                if alpha * beta / (alpha + beta) * ijsq > EXP_ARG_CUTOFF:
                    continue
                out[base + ip * nprimB + jp] = grp_branch[p, pp_group[p, q]]
                q += 1
    return out


@njit(cache=True, nogil=True)
def _item_table_ff(work, qpair, qaux, aux_off, aux_nbf, ncols, threshold, pp_off, pp_branch,
                   ff, col_off, cols):
    """
    The ``(pair, auxiliary shell, column)`` triples of the near field: a shell is a column of the
    pair when it survives Schwarz screening *and* at least one primitive pair of the shell pair is
    not far field for it.  Two linear passes.  The emitted layout is checked against the CPU plan
    twice over - the per-pair column total against ``pair_ncols`` and every column's auxiliary
    function against ``cols`` - because the ``J`` pass indexes the *CPU* column map with the
    offsets emitted here, and a difference in the order of the kept shells would otherwise pass
    a size check unnoticed and silently mis-contract J.
    """
    count = 0
    for w in range(work.shape[0]):
        p = work[w]
        npp = pp_off[p + 1] - pp_off[p]
        for k in range(qaux.size):
            if qpair[p] * qaux[k] <= threshold:
                continue
            for q in range(npp):
                if not ff[pp_branch[pp_off[p] + q], k]:
                    count += 1
                    break
    items = np.empty((count, 3), dtype=np.int32)
    index = 0
    for w in range(work.shape[0]):
        p = work[w]
        npp = pp_off[p + 1] - pp_off[p]
        col = 0
        for k in range(qaux.size):
            if qpair[p] * qaux[k] <= threshold:
                continue
            near = False
            for q in range(npp):
                if not ff[pp_branch[pp_off[p] + q], k]:
                    near = True
                    break
            if not near:
                continue
            items[index, 0], items[index, 1], items[index, 2] = p, k, col
            for c in range(aux_nbf[k]):
                if cols[col_off[p] + col + c] != aux_off[k] + c:
                    raise AssertionError('DF12 item columns disagree with the CPU column map')
            col += aux_nbf[k]
            index += 1
        if col != ncols[p]:
            raise AssertionError('DF12 item columns disagree with CPU plan sizing')
    return items


def _fixed_width_table(pair_off, ent_LM, ent_coef):
    """
    ``(n_pairs, LM, coef)`` of the translation table in slot-major fixed-width form:
    ``LM[w * n_pairs + p]`` and ``coef[w * n_pairs + p]`` are entry ``w`` of table pair ``p``,
    with a zero coefficient (and harmless index 0) where a pair has fewer entries.  Two slots
    are always enough: a product of two real solid harmonics has at most the two components
    ``M = m + k`` and ``M = m - k``.
    """
    n_pairs = pair_off.size - 1
    counts = np.diff(pair_off)
    if counts.max() > 2:
        raise ValueError('the real addition table has more than two entries per (lm, jk)')
    LM = np.zeros(2 * n_pairs, dtype=np.int32)
    coef = np.zeros(2 * n_pairs, dtype=np.float64)
    for slot in range(2):
        present = counts > slot
        index = pair_off[:-1][present] + slot
        LM[slot * n_pairs:(slot + 1) * n_pairs][present] = ent_LM[index]
        coef[slot * n_pairs:(slot + 1) * n_pairs][present] = ent_coef[index]
    return n_pairs, LM, coef


def _make_batch(plan, work):
    """Materialize the host screening decision once; all columns are whole shells."""
    items = _item_table_ff(work, plan.Q_pair, plan.Q_aux, plan.aux_off, plan.aux_nbf,
                           plan.pair_ncols, plan.threshold, plan.pp_off, plan.pp_branch,
                           plan.ff_gpu, plan.col_off, plan.cols)
    groups = []
    low_count = 0
    if items.size:
        items, class_ranges, low_count = _group_items(
            items, plan.pair_I, plan.pair_J, plan.shell_l, plan.aux_l,
            plan.bfs_nprim, plan.shell_off, plan.aux_nprim, plan.aux_off)
        for code, start, end in class_ranges:
            code = int(code)
            groups.append(((code // 49, code // 7 % 7, code % 7), int(start), int(end)))
    row_counts = plan.pair_nrows[work]
    row_pairs = np.repeat(work, row_counts)
    row_numbers = (np.arange(row_counts.sum())
                   - np.repeat(np.cumsum(row_counts) - row_counts, row_counts))
    row_items = np.column_stack((row_pairs, row_numbers)).astype(np.int32)
    return _Batch(cp.asarray(work, dtype=cp.int64), cp.asarray(items), groups,
                  cp.asarray(row_items), low_count)


def _check_batch(plan, batch):
    counts = cp.zeros(plan.n_pairs_total, dtype=cp.int64)
    errors = cp.zeros(max(batch.items.shape[0], 1), dtype=cp.int32)
    if batch.items.shape[0]:
        count_columns_ff[(batch.items.shape[0] + 127) // 128, 128, plan.nb_stream](
            batch.items, plan.pairs, plan.aux_shells, plan.device['Q_pair'], plan.device['Q_aux'],
            plan.threshold, plan.device['pp_off'], plan.device['pp_branch'],
            plan.device['ff_gpu'], plan.device['col_off'], plan.device['cols'], counts, errors)
    plan.stream.synchronize()
    work = cp.asnumpy(batch.work)
    if np.any(cp.asnumpy(errors)) or not np.array_equal(cp.asnumpy(counts)[work], plan.pair_ncols[work]):
        raise AssertionError('DF12 GPU screening/column count differs from CPU metadata')


def _build_batch(plan, batch, offsets, values):
    from .df_algo11_cuda_kernels import KERNELS
    # As in algorithm 11: all producers and consumers run in the explicitly owned plan stream,
    # so Numba's default per-argument synchronization is bypassed with zero-copy views.
    nb_offsets = cuda.as_cuda_array(offsets, sync=False)
    nb_values = cuda.as_cuda_array(values, sync=False)
    nb_x = cuda.as_cuda_array(plan.data_x, sync=False)
    nb_w = cuda.as_cuda_array(plan.data_w, sync=False)
    if batch.low_count:
        kernel, threads, _ = KERNELS[2, 2, 4]
        kernel[(batch.low_count + threads - 1) // threads, threads, plan.nb_stream](
            plan.orbital, plan.auxiliary, plan.shells, plan.aux_shells, plan.pairs,
            cuda.as_cuda_array(batch.items[:batch.low_count], sync=False),
            nb_offsets, nb_values, nb_x, nb_w, True, plan.mask)
    for cls, start, end in batch.groups:
        if end <= batch.low_count:
            continue
        kernel, threads, cooperative = KERNELS[cls]
        items = batch.items[start:end]
        blocks = end - start if cooperative else (end - start + threads - 1) // threads
        kernel[blocks, threads, plan.nb_stream](
            plan.orbital, plan.auxiliary, plan.shells, plan.aux_shells, plan.pairs,
            cuda.as_cuda_array(items, sync=False), nb_offsets, nb_values, nb_x, nb_w,
            True, plan.mask)
    if plan.sao and batch.items.shape[0]:
        project_items[batch.items.shape[0], 64, plan.nb_stream](
            batch.items, plan.pairs, plan.aux_shells, offsets, values, plan.device['projectors'])
    if plan.debug:
        _check_batch(plan, batch)
        plan.stream.synchronize()
        if not bool(cp.isfinite(values).all()):
            raise AssertionError('Non-finite DF12 GPU block values')


# ----------------------------------------------------------------------------
# Plan construction
# ----------------------------------------------------------------------------
def build_plan_cupy(basis, auxbasis, sqrt_ints4c2e_diag, sqrt_diag_ints2c2e, threshold,
                    strict_schwarz, sao=False, max_memory_gb=None, options=None,
                    cp_stream=None, *, batch_memory_gb=None, debug=False):
    """Build CPU-identical metadata and evaluate the cached near-field blocks on the GPU.

    ``batch_memory_gb`` optionally limits extra direct workspace (default: at most 1 GB and one
    quarter of currently free memory); a whole shell-pair row block must fit in it.  Inputs may
    be NumPy or CuPy arrays.  ``debug`` verifies the device near-field screening, the column
    offsets and counts, and finiteness after every build, including direct batches.
    """
    if cp is None:
        raise ImportError('DF_algo=12 on the GPU requires CuPy')
    if not cuda.is_available():
        raise RuntimeError('DF_algo=12 on the GPU requires a CUDA device')
    for name, value in [('max_memory_gb', max_memory_gb), ('batch_memory_gb', batch_memory_gb)]:
        if value is not None and (not np.isfinite(value) or value < 0):
            raise ValueError(f'{name} must be finite and nonnegative, or None')
    if options and options.get('low_memory'):
        raise NotImplementedError('DF_algo=12 on the GPU stores the branch-centred moments; '
                                  'low_memory=True is a CPU-only option')
    # Honor producers on the caller's current stream before using another stream.
    cp.cuda.get_current_stream().synchronize()
    host = _plan_metadata(basis, auxbasis, _host(sqrt_ints4c2e_diag), _host(sqrt_diag_ints2c2e),
                          threshold, strict_schwarz, sao, max_memory_gb, options=options)
    if max(host.shell_l.max(), host.aux_l.max()) > 6:
        raise ValueError('DF_algo=12 CUDA supports orbital and auxiliary shells through l=6')
    if max(host.n_pairs_total, host.naux, host.aux_off.size) > np.iinfo(np.int32).max:
        raise ValueError('DF_algo=12 CUDA item indices exceed int32 capacity')
    plan = DFAlgo12GPUPlan()
    plan.__dict__.update(host.__dict__)
    plan.device_id = cp.cuda.Device().id
    plan.stream = cp.cuda.get_current_stream() if cp_stream is None else cp_stream
    plan.nb_stream = cuda.external_stream(plan.stream.ptr)
    plan.debug = debug
    plan.peak_pool_used_bytes = plan.peak_pool_reserved_bytes = 0
    timings = dict(plan.timings)

    # Primitive-pair branches, flattened over the significant pairs: the mask the Rys kernels
    # read.  ``ff_gpu`` is ``ff_eff`` (whose last row is near field everywhere, for the groups the
    # profitability rule leaves in the near field) plus one row that is far field everywhere, for
    # the primitive pairs the host pre-screen dropped.
    npairs = plan.pair_I.shape[0]
    nprim_pairs = np.zeros(npairs, dtype=np.int64)
    a0 = plan.shell_off[plan.pair_I[plan.sig]]
    b0 = plan.shell_off[plan.pair_J[plan.sig]]
    nprim_pairs[plan.sig] = plan.bfs_nprim[a0] * plan.bfs_nprim[b0]
    plan.pp_off = np.zeros(npairs + 1, dtype=np.int64)
    plan.pp_off[1:] = np.cumsum(nprim_pairs)
    plan.ff_gpu = np.ascontiguousarray(
        np.vstack([plan.ff_eff, np.ones((1, plan.ff_eff.shape[1]), dtype=np.bool_)]))
    plan.pp_branch = _pp_branch_dense(plan.sig, plan.pair_I, plan.pair_J, plan.shell_off,
                                      plan.bfs_nprim, plan.bfs_expnts, plan.bfs_coords,
                                      plan.pp_group, plan.grp_branch_eff, plan.pp_off,
                                      plan.ff_eff.shape[0])
    # Global row index of every function pair of every significant shell pair.
    plan.row_off = np.zeros(npairs, dtype=np.int64)
    row_counts = np.zeros(npairs, dtype=np.int64)
    row_counts[plan.sig] = plan.pair_nrows[plan.sig]
    plan.row_off[:] = np.cumsum(row_counts) - row_counts
    plan.n_rows_total = int(row_counts.sum())
    rows_all = np.column_stack((np.repeat(plan.sig, plan.pair_nrows[plan.sig]),
                                np.arange(plan.n_rows_total)
                                - np.repeat(plan.row_off[plan.sig], plan.pair_nrows[plan.sig])
                                )).astype(np.int32)
    plan.aux_bf_shell = np.repeat(np.arange(plan.aux_off.size, dtype=np.int32), plan.aux_nbf)
    # Entries grouped by branch, so the gamma pass accumulates one moment vector per branch
    # instead of one fp64 atomic per (entry, LM).
    plan.branch_entries = np.argsort(plan.entry_branch, kind='stable').astype(np.int32)
    plan.branch_entry_off = np.zeros(max(plan.n_branches, 1) + 1, dtype=np.int64)
    plan.branch_entry_off[1:] = np.cumsum(np.bincount(plan.entry_branch.astype(np.int64),
                                                      minlength=max(plan.n_branches, 1)))

    with plan.stream:
        names = ('bfs_coords', 'bfs_lmn', 'bfs_nprim', 'bfs_expnts', 'bf_coef',
                 'aux_coords', 'aux_lmn', 'aux_nprim', 'aux_expnts', 'aux_coef',
                 'shell_off', 'shell_nbf', 'shell_l', 'aux_off', 'aux_nbf', 'aux_l',
                 'pair_I', 'pair_J', 'pair_nrows', 'pair_ncols', 'pair_offset',
                 'Q_pair', 'Q_aux', 'sqrt_ints4c2e_diag', 'projectors',
                 # algorithm 12: the far-field mask of the near-field kernels ...
                 'pp_off', 'pp_branch', 'ff_gpu',
                 # ... the near-field column map ...
                 'col_off', 'cols', 'row_off',
                 # ... and the far field itself
                 'Mtil', 'entry_pair', 'entry_branch', 'entry_moff', 'branch_center',
                 'atom_coords', 'atom_shell_off', 'atom_shells', 'atom_lmax', 'aux_mom',
                 'aux_bf_shell', 'ff', 'any_ff', 'sign_big',
                 'branch_entries', 'branch_entry_off')
        plan.device = {name: cp.asarray(getattr(plan, name)) for name in names}
        plan.device['entry_pair'] = plan.device['entry_pair'].astype(cp.int64)
        plan.device['entry_branch'] = plan.device['entry_branch'].astype(cp.int64)
        plan.device['rows_all'] = cp.asarray(rows_all)
        # The real addition table has at most two non-zero LM per (lm, jk); store it slot-major
        # and zero-padded so the coupling kernels read it with two unconditional coalesced loads
        # instead of a dependent offset load and a data-dependent trip count.
        plan.npidx, tab_LM, tab_coef = _fixed_width_table(*plan.table)
        plan.device['tab_LM'] = cp.asarray(tab_LM)
        plan.device['tab_coef'] = cp.asarray(tab_coef)
        views = {n: cuda.as_cuda_array(plan.device[n], sync=False) for n in plan.device}
        plan.nb = views
        plan.orbital = tuple(views[n] for n in names[:5])
        plan.auxiliary = tuple(views[n] for n in names[5:10])
        plan.shells = tuple(views[n] for n in names[10:13])
        plan.aux_shells = tuple(views[n] for n in names[13:16])
        plan.pairs = tuple(views[n] for n in names[16:20])
        plan.mask = (views['pp_off'], views['pp_branch'], views['ff_gpu'])
        high_roots = (2 * plan.shell_l.max() + plan.aux_l.max()) // 2 + 1 > 5
        plan.data_x = cp.asarray(DATA_X) if high_roots else cp.empty(0, dtype=cp.float64)
        plan.data_w = cp.asarray(DATA_W) if high_roots else cp.empty(0, dtype=cp.float64)
        # The branch-centred moments are on the device from here on, and the group moments they
        # were translated from are only needed by the (CPU-only) low_memory passes.
        plan.moment_bytes = plan.moments.nbytes
        plan.Mtil = np.zeros(0)
        plan.moments = np.zeros(0)

        plan.n_small = (plan.l_aux_max + 1) ** 2
        plan.n_irr = (plan.lmax + plan.l_aux_max + 1) ** 2
        # One thread per LM where that fits, so the reads of Mtil are coalesced and the shared
        # accumulator is written once per thread.  The second grid dimension splits one branch's
        # entries (or one atom's branches) over several blocks: there are far fewer branches and
        # atoms than the device has resident blocks, and their work is very unevenly distributed.
        plan.bm_threads = int(min(1024, ((plan.n_big + 31) // 32) * 32))
        # One warp per branch in branches_to_atoms; its scratch is (n_irr + n_small) doubles.
        b2a_scratch = plan.n_irr + plan.n_small + plan.n_big
        plan.b2a_warps = max(1, min(8, _SHARED_LIMIT // (8 * b2a_scratch)))
        plan.b2a_shared = plan.b2a_warps * b2a_scratch * 8
        plan.a2b_shared = (plan.n_irr + plan.n_small + plan.n_big) * 8
        blocks = max(plan.n_branches, 1)
        plan.bm_split = int(max(1, min(64, -(-4096 // blocks))))
        plan.b2a_split = int(max(1, min(64, -(-4096 // (plan.atom_coords.shape[0] * plan.b2a_warps)))))
        plan.a2b_split = int(max(1, min(64, -(-4096 // blocks))))
        if plan.a2b_shared > _SHARED_LIMIT:
            raise ValueError('DF_algo=12 CUDA needs %.1f KB of shared memory for the branch '
                             'coupling; reduce lmax' % (plan.a2b_shared / 1024))

        t0 = timer()
        plan.values = cp.zeros(plan.n_elements_cached, dtype=cp.float64)
        plan.cached = _make_batch(plan, plan.work_build)
        direct = plan.work_iter[plan.pair_offset[plan.work_iter] < 0]
        plan.batches = []
        plan.temporary_elements = 0
        temporary_offsets = np.full(plan.n_pairs_total, -1, dtype=np.int64)
        if direct.size:
            free, _ = cp.cuda.runtime.memGetInfo()
            capacity = int(min(1e9, free // 4) if batch_memory_gb is None else batch_memory_gb * 1e9) // 8
            sizes = plan.pair_nrows[direct] * plan.pair_ncols[direct]
            if sizes.max() > capacity:
                raise MemoryError(f'A direct shell-pair block needs {sizes.max() * 8e-9:.6f} GB; '
                                  'increase batch_memory_gb or cache that block')
            begin, used = 0, 0
            for index, (p, size) in enumerate(zip(direct, sizes)):
                if used + size > capacity:
                    plan.batches.append(_make_batch(plan, direct[begin:index]))
                    plan.temporary_elements = max(plan.temporary_elements, int(used))
                    begin, used = index, 0
                temporary_offsets[p] = used
                used += int(size)
            plan.batches.append(_make_batch(plan, direct[begin:]))
            plan.temporary_elements = max(plan.temporary_elements, used)
        plan.temporary_offsets = cp.asarray(temporary_offsets)
        plan.nb_temporary_offsets = cuda.as_cuda_array(plan.temporary_offsets, sync=False)
        _build_batch(plan, plan.cached, plan.device['pair_offset'], plan.values)
        plan.stream.synchronize()
        timings['near_field'] = timer() - t0
        timings['total'] = timings.get('metadata_total', 0.0) + timings['near_field']
        plan.timings = timings
        plan.memory_stats()
    return plan


# ----------------------------------------------------------------------------
# Per-iteration passes
# ----------------------------------------------------------------------------
def _near_field(plan, source, gamma, out, rowff=None):
    """Near-field contraction over the cached blocks and the rebuilt direct batches."""
    def contract(batch, offsets, values):
        nb_offsets = cuda.as_cuda_array(offsets, sync=False)
        nb_values = cuda.as_cuda_array(values, sync=False)
        if gamma and batch.items.shape[0]:
            gamma_items[(batch.items.shape[0] * 8 + 127) // 128, 128, plan.nb_stream](
                batch.nb_items, plan.pairs, plan.shells, plan.aux_shells, nb_offsets, nb_values,
                plan.nb['sqrt_ints4c2e_diag'], plan.strict_schwarz, source, out)
        elif not gamma and batch.work.size:
            j_pairs_ff[(batch.rows.shape[0] * 32 + 127) // 128, 128, plan.nb_stream](
                batch.nb_rows, plan.pairs, plan.shells, plan.aux_shells, nb_offsets, nb_values,
                plan.nb['sqrt_ints4c2e_diag'], plan.strict_schwarz,
                plan.nb['col_off'], plan.nb['cols'], source, plan.nb['row_off'], rowff, out)

    contract(plan.cached, plan.device['pair_offset'], plan.values)
    if plan.batches:
        scratch = cp.empty(plan.temporary_elements, dtype=cp.float64)
        for batch in plan.batches:
            if plan.debug:
                scratch.fill(0)
            _build_batch(plan, batch, plan.temporary_offsets, scratch)
            contract(batch, plan.temporary_offsets, scratch)


def _sample_pool(plan):
    pool = cp.get_default_memory_pool()
    plan.peak_pool_used_bytes = max(plan.peak_pool_used_bytes, pool.used_bytes())
    plan.peak_pool_reserved_bytes = max(plan.peak_pool_reserved_bytes, pool.total_bytes())


def gamma_from_plan_cupy(plan, dmat_cp):
    """Return CuPy ``gamma_P = sum_ij D_ij (ij|P)``: near-field blocks plus the far-field multipoles."""
    cp.cuda.get_current_stream().synchronize()
    with cp.cuda.Device(plan.device_id), plan.stream:
        dmat = cp.ascontiguousarray(cp.asarray(dmat_cp, dtype=cp.float64))
        if dmat.shape != (plan.nao, plan.nao):
            raise ValueError(f'Expected input shape {(plan.nao, plan.nao)}, got {dmat.shape}')
        gamma = cp.zeros(plan.naux, dtype=cp.float64)
        rowcoef = cp.zeros(max(plan.n_rows_total, 1), dtype=cp.float64)
        nb_dmat = cuda.as_cuda_array(dmat, sync=False)
        nb_gamma = cuda.as_cuda_array(gamma, sync=False)
        nb_rowcoef = cuda.as_cuda_array(rowcoef, sync=False)
        if plan.n_rows_total:
            row_coefficients[(plan.n_rows_total + 127) // 128, 128, plan.nb_stream](
                plan.nb['rows_all'], plan.pairs, plan.shells, plan.nb['sqrt_ints4c2e_diag'],
                plan.strict_schwarz, nb_dmat, plan.nb['row_off'], nb_rowcoef)
        _near_field(plan, nb_dmat, True, nb_gamma)
        if plan.n_entries and plan.n_branches:
            branch_mom = cp.zeros((plan.n_branches, plan.n_big), dtype=cp.float64)
            nb_branch_mom = cuda.as_cuda_array(branch_mom, sync=False)
            branch_moments[(plan.n_branches, plan.bm_split), plan.bm_threads, plan.nb_stream,
                           plan.n_big * 8](
                plan.nb['branch_entry_off'], plan.nb['branch_entries'], plan.nb['entry_pair'],
                plan.nb['entry_moff'], plan.nb['pair_nrows'], plan.nb['row_off'],
                plan.nb['Mtil'], nb_rowcoef, plan.n_big, nb_branch_mom)
            L_K = cp.zeros((plan.aux_off.size, plan.n_small), dtype=cp.float64)
            branches_to_atoms[(plan.atom_coords.shape[0], plan.b2a_split), 32 * plan.b2a_warps,
                              plan.nb_stream, plan.b2a_shared](
                nb_branch_mom, plan.nb['branch_center'], plan.nb['atom_coords'],
                plan.nb['atom_shell_off'], plan.nb['atom_shells'], plan.nb['atom_lmax'],
                plan.nb['aux_l'], plan.nb['ff'], plan.nb['any_ff'], plan.lmax, plan.n_big,
                plan.n_small, plan.n_irr, plan.npidx, plan.nb['tab_LM'], plan.nb['tab_coef'],
                plan.nb['sign_big'], cuda.as_cuda_array(L_K, sync=False))
            gamma_far_field[(plan.naux + 127) // 128, 128, plan.nb_stream](
                cuda.as_cuda_array(L_K, sync=False), plan.nb['aux_mom'], plan.nb['aux_bf_shell'],
                plan.nb['aux_l'], nb_gamma)
        plan.stream.synchronize()
        _sample_pool(plan)
    return gamma


def J_from_plan_cupy(plan, coeff_cp):
    """Return the full symmetric CuPy Coulomb matrix: near-field blocks plus the far-field multipoles."""
    cp.cuda.get_current_stream().synchronize()
    with cp.cuda.Device(plan.device_id), plan.stream:
        coeff = cp.ascontiguousarray(cp.asarray(coeff_cp, dtype=cp.float64))
        if coeff.shape != (plan.naux,):
            raise ValueError(f'Expected input shape {(plan.naux,)}, got {coeff.shape}')
        rowff = cp.zeros(max(plan.n_rows_total, 1), dtype=cp.float64)
        nb_coeff = cuda.as_cuda_array(coeff, sync=False)
        nb_rowff = cuda.as_cuda_array(rowff, sync=False)
        if plan.n_entries and plan.n_branches:
            shell_mom = cp.zeros((plan.aux_off.size, plan.n_small), dtype=cp.float64)
            shell_moments[plan.aux_off.size, max(32, plan.n_small), plan.nb_stream](
                nb_coeff, plan.nb['aux_mom'], plan.nb['aux_off'], plan.nb['aux_nbf'],
                plan.nb['aux_l'], cuda.as_cuda_array(shell_mom, sync=False))
            branch_local = cp.zeros((plan.n_branches, plan.n_big), dtype=cp.float64)
            atoms_to_branches[(plan.n_branches, plan.a2b_split), 128, plan.nb_stream,
                              plan.a2b_shared](
                cuda.as_cuda_array(shell_mom, sync=False), plan.nb['branch_center'],
                plan.nb['atom_coords'], plan.nb['atom_shell_off'], plan.nb['atom_shells'],
                plan.nb['atom_lmax'], plan.nb['aux_l'], plan.nb['ff'], plan.nb['any_ff'],
                plan.lmax, plan.n_big, plan.n_small, plan.n_irr, plan.npidx,
                plan.nb['tab_LM'], plan.nb['tab_coef'], plan.nb['sign_big'],
                cuda.as_cuda_array(branch_local, sync=False))
            row_far_field[plan.n_entries, 128, plan.nb_stream, plan.n_big * 8](
                plan.nb['entry_pair'], plan.nb['entry_branch'], plan.nb['entry_moff'],
                plan.nb['pair_nrows'], plan.nb['row_off'], plan.nb['Mtil'],
                cuda.as_cuda_array(branch_local, sync=False), plan.n_big, nb_rowff)
        J = cp.zeros((plan.nao, plan.nao), dtype=cp.float64)
        _near_field(plan, nb_coeff, False, cuda.as_cuda_array(J, sync=False), nb_rowff)
        plan.stream.synchronize()
        _sample_pool(plan)
    return J
