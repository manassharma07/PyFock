"""Shell-blocked fp64 density fitting on CUDA (algorithm 11).

The CPU metadata builder is the single source of screening, ranking, int64
offsets and cache selection. Device items are (pair, auxiliary shell, column)
int32 triples, grouped by angular momentum and ordered by primitive cost.
Both build tiers reuse roots, recurrences and horizontal shifts for an entire
shell triple. Auxiliary projection is a separate pass over complete shell rows.

``max_memory_gb`` limits cached *device* values only (None = all, 0 = direct).
Uncached pairs are evaluated in bounded batches in both contraction passes;
``temporary_memory_gb`` reports their additional workspace. Cached values stay
on the GPU, including when the driver's ``keep_ints3c2e_in_gpu`` is false.
No global stream/device selection or memory-pool flushing is performed here.
Public operations synchronize their stream before returning.
"""
from dataclasses import dataclass

import numpy as np
from numba import cuda, njit

try:
    import cupy as cp
except ImportError:
    cp = None

from .df_algo11_helpers import DFAlgo11Plan, _plan_metadata
from .df_algo11_cuda_core import project_items, gamma_items, j_pairs, count_columns, make_column_map
from .rys_helpers_cuda import DATA_X, DATA_W

__all__ = ['DFAlgo11GPUPlan', 'build_plan_cupy', 'gamma_from_plan_cupy', 'J_from_plan_cupy']


class DFAlgo11GPUPlan(DFAlgo11Plan):
    """CPU-compatible plan metadata, device values and reusable device work lists."""

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
        return dict(cached_bytes=self.values.nbytes, temporary_bytes=self.temporary_elements * 8,
                    pool_used_bytes=used, pool_reserved_bytes=reserved,
                    peak_pool_used_bytes=self.peak_pool_used_bytes,
                    peak_pool_reserved_bytes=self.peak_pool_reserved_bytes,
                    device_free_bytes=free, device_total_bytes=total)


@dataclass
class _Batch:
    work: object
    items: object
    groups: list
    rows: object
    low_count: int

    def __post_init__(self):
        self.nb_items = cuda.as_cuda_array(self.items, sync=False)
        self.nb_work = cuda.as_cuda_array(self.work, sync=False)
        self.nb_rows = cuda.as_cuda_array(self.rows, sync=False)


def _host(array):
    return cp.asnumpy(array) if isinstance(array, cp.ndarray) else np.asarray(array)


@njit(cache=True, nogil=True)
def _item_table(work, qpair, qaux, aux_nbf, ncols, threshold):
    """Two linear passes avoid thousands of tiny NumPy allocations at plan time."""
    count = 0
    for p in work:
        for k in range(qaux.size):
            if qpair[p] * qaux[k] > threshold:
                count += 1
    items = np.empty((count, 3), dtype=np.int32)
    index = 0
    for p in work:
        col = 0
        for k in range(qaux.size):
            if qpair[p] * qaux[k] > threshold:
                items[index, 0], items[index, 1], items[index, 2] = p, k, col
                col += aux_nbf[k]
                index += 1
        if col != ncols[p]:
            raise AssertionError('DF11 item columns disagree with CPU plan sizing')
    return items


@njit(cache=True, nogil=True)
def _counting_order(keys):
    """Stable counting sort for the small integer (class, primitive-cost) domain."""
    counts = np.zeros(keys.max() + 1, dtype=np.int64)
    for key in keys:
        counts[key] += 1
    offset = 0
    for key in range(counts.size):
        size = counts[key]
        counts[key] = offset
        offset += size
    order = np.empty(keys.size, dtype=np.int64)
    for index in range(keys.size):
        key = keys[index]
        order[counts[key]] = index
        counts[key] += 1
    return order


@njit(cache=True, nogil=True)
def _group_items(items, pair_i, pair_j, shell_l, aux_l, nprim, shell_off, aux_nprim, aux_off):
    """Group in compiled loops without large temporary NumPy gather arrays."""
    max_cost = nprim.max() ** 2 * aux_nprim.max()
    cost_stride = max_cost + 1
    keys = np.empty(items.shape[0], dtype=np.int64)
    low_count = 0
    for index in range(items.shape[0]):
        p, k = items[index, 0], items[index, 1]
        i, j = pair_i[p], pair_j[p]
        a, b, c = shell_l[i], shell_l[j], aux_l[k]
        cls = 49 * a + 7 * b + c
        if a <= 2 and b <= 2 and c <= 4:
            low_count += 1
        else:
            cls += 343
        cost = nprim[shell_off[i]] * nprim[shell_off[j]] * aux_nprim[aux_off[k]]
        keys[index] = cls * cost_stride + max_cost - cost
    order = (_counting_order(keys) if keys.max() < 1000000
             else np.argsort(keys, kind='mergesort'))
    out = np.empty_like(items)
    groups = np.empty((343, 3), dtype=np.int64)
    previous = -1
    ngroups = 0
    for dest in range(order.size):
        src = order[dest]
        bucket = keys[src] // cost_stride
        if bucket != previous:
            if ngroups:
                groups[ngroups - 1, 2] = dest
            groups[ngroups, 0] = bucket % 343
            groups[ngroups, 1] = dest
            previous = bucket
            ngroups += 1
        for col in range(3):
            out[dest, col] = items[src, col]
    groups[ngroups - 1, 2] = order.size
    return out, groups[:ngroups], low_count


def _make_batch(plan, work):
    """Materialize the host screening decision once; all columns are whole shells."""
    items = _item_table(work, plan.Q_pair, plan.Q_aux, plan.aux_nbf, plan.pair_ncols, plan.threshold)
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
    errors = cp.zeros(batch.items.shape[0], dtype=cp.int32)
    if batch.items.shape[0]:
        count_columns[(batch.items.shape[0] + 127) // 128, 128, plan.nb_stream](
            batch.work, batch.items, plan.pairs, plan.aux_shells, plan.device['Q_pair'],
            plan.device['Q_aux'], plan.threshold, counts, errors)
    plan.stream.synchronize()
    work = cp.asnumpy(batch.work)
    if np.any(cp.asnumpy(errors)) or not np.array_equal(cp.asnumpy(counts)[work], plan.pair_ncols[work]):
        raise AssertionError('DF11 GPU screening/column count differs from CPU metadata')


def _build_batch(plan, batch, offsets, values):
    from .df_algo11_cuda_kernels import KERNELS
    # Avoid Numba's default per-argument stream synchronization: all producers
    # and consumers here run in the explicitly owned plan stream.
    nb_offsets = cuda.as_cuda_array(offsets, sync=False)
    nb_values = cuda.as_cuda_array(values, sync=False)
    nb_x = cuda.as_cuda_array(plan.data_x, sync=False)
    nb_w = cuda.as_cuda_array(plan.data_w, sync=False)
    if batch.low_count:
        # Measurements favor one bounded spd/d-g launch over dozens of tiny
        # class launches. High angular momentum still uses separate scratch.
        kernel, threads, _ = KERNELS[2, 2, 4]
        kernel[(batch.low_count + threads - 1) // threads, threads, plan.nb_stream](
            plan.orbital, plan.auxiliary, plan.shells, plan.aux_shells, plan.pairs,
            cuda.as_cuda_array(batch.items[:batch.low_count], sync=False),
            nb_offsets, nb_values, nb_x, nb_w)
    for cls, start, end in batch.groups:
        if end <= batch.low_count:
            continue
        kernel, threads, cooperative = KERNELS[cls]
        items = batch.items[start:end]
        blocks = end - start if cooperative else (end - start + threads - 1) // threads
        kernel[blocks, threads, plan.nb_stream](
            plan.orbital, plan.auxiliary, plan.shells, plan.aux_shells, plan.pairs,
            cuda.as_cuda_array(items, sync=False), nb_offsets, nb_values, nb_x, nb_w)
    if plan.sao and batch.items.shape[0]:
        project_items[batch.items.shape[0], 64, plan.nb_stream](
            batch.items, plan.pairs, plan.aux_shells, offsets, values, plan.device['projectors'])
    if plan.debug:
        _check_batch(plan, batch)
        plan.stream.synchronize()
        if not bool(cp.isfinite(values).all()):
            raise AssertionError('Non-finite DF11 GPU block values')


def build_plan_cupy(basis, auxbasis, sqrt_ints4c2e_diag, sqrt_diag_ints2c2e,
                    threshold, strict_schwarz, sao=False, max_memory_gb=None,
                    cp_stream=None, *, batch_memory_gb=None, debug=False):
    """Build CPU-identical metadata and evaluate cached row blocks on the GPU.

    ``batch_memory_gb`` optionally limits extra direct workspace (default: at most
    1 GB and one quarter of currently free memory). A whole shell-pair row block
    must fit in that workspace. Inputs may be NumPy or CuPy arrays. Shell angular
    momenta 0..6 are supported on both sides. ``debug`` verifies device screening,
    column offsets/counts and finiteness after every build, including direct batches.
    """
    if cp is None:
        raise ImportError('DF_algo=11 on the GPU requires CuPy')
    if not cuda.is_available():
        raise RuntimeError('DF_algo=11 on the GPU requires a CUDA device')
    for name, value in [('max_memory_gb', max_memory_gb), ('batch_memory_gb', batch_memory_gb)]:
        if value is not None and (not np.isfinite(value) or value < 0):
            raise ValueError(f'{name} must be finite and nonnegative, or None')
    # Honor producers on the caller's current stream before using another stream.
    cp.cuda.get_current_stream().synchronize()
    host = _plan_metadata(basis, auxbasis, _host(sqrt_ints4c2e_diag), _host(sqrt_diag_ints2c2e),
                          threshold, strict_schwarz, sao, max_memory_gb)
    if max(host.shell_l.max(), host.aux_l.max()) > 6:
        raise ValueError('DF_algo=11 CUDA supports orbital and auxiliary shells through l=6')
    if max(host.n_pairs_total, host.naux, host.aux_off.size) > np.iinfo(np.int32).max:
        raise ValueError('DF_algo=11 CUDA item indices exceed int32 capacity')
    plan = DFAlgo11GPUPlan()
    plan.__dict__.update(host.__dict__)
    plan.device_id = cp.cuda.Device().id
    plan.stream = cp.cuda.get_current_stream() if cp_stream is None else cp_stream
    plan.nb_stream = cuda.external_stream(plan.stream.ptr)
    plan.debug = debug
    plan.peak_pool_used_bytes = plan.peak_pool_reserved_bytes = 0
    with plan.stream:
        names = ('bfs_coords', 'bfs_lmn', 'bfs_nprim', 'bfs_expnts', 'bf_coef',
                 'aux_coords', 'aux_lmn', 'aux_nprim', 'aux_expnts', 'aux_coef',
                 'shell_off', 'shell_nbf', 'shell_l', 'aux_off', 'aux_nbf', 'aux_l',
                 'pair_I', 'pair_J', 'pair_nrows', 'pair_ncols', 'pair_offset',
                 'Q_pair', 'Q_aux', 'sqrt_ints4c2e_diag', 'projectors')
        plan.device = {name: cp.asarray(getattr(plan, name)) for name in names}
        # Numba adapts top-level CUDA-array-interface arguments, but not arrays
        # nested in tuples. Explicit zero-copy views keep the tuple ABI portable.
        views = {n: cuda.as_cuda_array(plan.device[n], sync=False) for n in names}
        plan.nb = views
        plan.orbital = tuple(views[n] for n in names[:5])
        plan.auxiliary = tuple(views[n] for n in names[5:10])
        plan.shells = tuple(views[n] for n in names[10:13])
        plan.aux_shells = tuple(views[n] for n in names[13:16])
        plan.pairs = tuple(views[n] for n in names[16:20])
        high_roots = (2 * plan.shell_l.max() + plan.aux_l.max()) // 2 + 1 > 5
        plan.data_x = cp.asarray(DATA_X) if high_roots else cp.empty(0, dtype=cp.float64)
        plan.data_w = cp.asarray(DATA_W) if high_roots else cp.empty(0, dtype=cp.float64)
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
        # One int32 per significant column, shared by every row of its pair.
        # This permits a coalesced warp dot product without a shell loop in J.
        plan.column_offsets = cp.asarray(np.cumsum(plan.pair_ncols) - plan.pair_ncols)
        plan.column_map = cp.empty(int(plan.pair_ncols.sum()), dtype=cp.int32)
        plan.nb_column_offsets = cuda.as_cuda_array(plan.column_offsets, sync=False)
        plan.nb_column_map = cuda.as_cuda_array(plan.column_map, sync=False)
        for batch in [plan.cached] + plan.batches:
            if batch.items.shape[0]:
                make_column_map[(batch.items.shape[0] + 127) // 128, 128, plan.nb_stream](
                    batch.items, plan.aux_shells, plan.column_offsets, plan.column_map)
        _build_batch(plan, plan.cached, plan.device['pair_offset'], plan.values)
        plan.memory_stats()
    return plan


def _contract(plan, source, gamma):
    cp.cuda.get_current_stream().synchronize()
    with cp.cuda.Device(plan.device_id), plan.stream:
        source = cp.ascontiguousarray(cp.asarray(source, dtype=cp.float64))
        expected = (plan.nao, plan.nao) if gamma else (plan.naux,)
        if source.shape != expected:
            raise ValueError(f'Expected input shape {expected}, got {source.shape}')
        out = cp.zeros(plan.naux if gamma else (plan.nao, plan.nao), dtype=cp.float64)
        nb_source = cuda.as_cuda_array(source, sync=False)
        nb_out = cuda.as_cuda_array(out, sync=False)

        def contract(batch, offsets, values):
            nb_offsets = cuda.as_cuda_array(offsets, sync=False)
            nb_values = cuda.as_cuda_array(values, sync=False)
            if gamma and batch.items.shape[0]:
                gamma_items[(batch.items.shape[0] * 8 + 127) // 128, 128, plan.nb_stream](
                    batch.nb_items, plan.pairs, plan.shells, plan.aux_shells, nb_offsets, nb_values,
                    plan.nb['sqrt_ints4c2e_diag'], plan.strict_schwarz, nb_source, nb_out)
            elif not gamma and batch.work.size:
                j_pairs[(batch.rows.shape[0] * 32 + 127) // 128, 128, plan.nb_stream](
                    batch.nb_rows, plan.pairs, plan.shells, plan.aux_shells, nb_offsets, nb_values,
                    plan.nb['sqrt_ints4c2e_diag'], plan.strict_schwarz,
                    plan.nb_column_offsets, plan.nb_column_map, nb_source, nb_out)

        contract(plan.cached, plan.device['pair_offset'], plan.values)
        if plan.batches:
            scratch = cp.empty(plan.temporary_elements, dtype=cp.float64)
            for batch in plan.batches:
                if plan.debug:
                    scratch.fill(0)
                _build_batch(plan, batch, plan.temporary_offsets, scratch)
                contract(batch, plan.temporary_offsets, scratch)
        plan.stream.synchronize()
        pool = cp.get_default_memory_pool()
        plan.peak_pool_used_bytes = max(plan.peak_pool_used_bytes, pool.used_bytes())
        plan.peak_pool_reserved_bytes = max(plan.peak_pool_reserved_bytes, pool.total_bytes())
    return out


def gamma_from_plan_cupy(plan, dmat_cp):
    """Return CuPy gamma_P = sum_ij D_ij (ij|P), including direct batches."""
    return _contract(plan, dmat_cp, True)


def J_from_plan_cupy(plan, coeff_cp):
    """Return the full symmetric CuPy Coulomb matrix, including direct batches."""
    return _contract(plan, coeff_cp, False)
