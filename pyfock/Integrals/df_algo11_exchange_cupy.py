"""
RI exact exchange (and Coulomb) on the GPU from the ``DF_algo=11`` CUDA plan.

The device counterpart of :mod:`~pyfock.Integrals.df_algo11_exchange`.  The screened
three-center blocks that :func:`~pyfock.Integrals.df_algo11_helpers_cupy.build_plan_cupy`
leaves on the device are expanded into one row per active function pair ``i >= j``
(strict Schwarz cut-off applied per pair), contracted with the Cartesian-to-spherical
matrices of the auxiliary shells in SAO mode (the fit space is the true spherical
auxiliary basis, as on the CPU) and orthonormalized in the fit metric in place,

    ``B = R L^-T``,   ``(P|Q) = L L^T``

(cuSOLVER Cholesky, one in-place cuBLAS ``trsm``).  ``B`` is the only three-center
storage and never leaves the GPU.

Per SCF iteration everything is cuBLAS:

* Coulomb: ``gamma = d . B`` and ``J_r = B gamma`` (two GEMVs over the rows, no metric
  solve); ``gamma . gamma`` is the DF Coulomb energy term.
* Exchange with the occupied density factor ``D = F F^T`` (``nocc`` columns):
  ``X[i, o, Q] = sum_k B[(ik), Q] F[k, o]`` and ``K_ij = sum_{o,Q} X[i, o, Q] X[j, o, Q]``.
  The rows that feed function ``i`` - its *own* rows ``(i, k <= i)`` and its *partner*
  rows ``(k > i, i)`` - are gathered for one block of ``nb`` auxiliary functions into a
  zero-padded slab ``U[i, k, Q]`` by one coalesced kernel (``B`` is read exactly twice
  per iteration).  The functions are sorted by their row count and grouped into *bins*
  of similar count, so that every bin is one strided-batched GEMM
  ``X[bin] = F_pad[bin] @ U[bin]`` with at most ``1/bin_fill - 1`` padding (25 % by
  default) and the pair sparsity built in; the rank update ``K += X X^T`` over the block
  is one ``syrk`` (one GEMM in fp32 below 1024 functions, where cuBLAS's ``ssyrk`` is several
  times slower).  ``nb`` is chosen so that ``U`` and ``X`` fit ``block_memory_bytes``.

The contraction can run in single precision (``dtype=cupy.float32``): the slabs are
converted while they are gathered (``B`` stays double), the GEMMs and the ``syrk`` run
in fp32 and ``K`` is returned as float64.  ``DFT.scf`` uses this for the early SCF
iterations under ``dynamic_precision`` (the same switch as the XC term): on consumer
GPUs with a 1/32-1/64 fp64 rate the exchange build is then 10-30x cheaper until the
last iterations are done in double precision, and the converged energy is a double
precision one.

Memory: ``B`` (``nrows * naux`` doubles) plus, during the build, the plan's cached
blocks (released as soon as the rows are filled when ``release_plan_values`` is set)
and the metric; the per-block work buffers are bounded by ``block_memory_bytes``.
All device work runs on one CuPy stream (the plan's, or the one current at build time)
and the public functions synchronize it before returning.

Nuclear gradient (:class:`~pyfock.DFT_Grad` with ``use_gpu=True``), the device counterpart of the
gradient part of :mod:`~pyfock.Integrals.df_algo11_exchange`.  The rows are built raw
(``build_exchange_cupy(..., orthonormalize=False)``) and give the occupied blocks of the fitted pair
densities, ``Y_P = F^T c^P F``, by the half transform of the exchange build, one GEMM per auxiliary
block and two triangular solves with the fit metric (:func:`occupied_fit_blocks_cupy`).  The rows
are then no longer needed: :func:`exchange_gradient_cupy` frees them and forms the three-center
weights ``scale * Gamma^P_ij (+ D_ij c_P)``, ``Gamma^P = F Y_P F^T``, directly in the Cartesian
auxiliary basis, one block of whole auxiliary shells at a time (``U = F Y``, then one padded
strided-batched GEMM per bin of functions with a similar number of own rows ``(i, j <= i)``), and
hands every block to the derivative kernel of
:class:`~pyfock.Integrals.rys_3c2e_grad_contract_cupy.RowsGradContext` at once, so that the weights
never exist in full.
"""
import numpy as np
from numba import cuda, njit, prange

try:
    import cupy as cp
    from cupy.cuda import cublas as _cublas
except ImportError:  # pragma: no cover - CPU-only install
    cp = None
    _cublas = None

from .df_algo10_helpers import STRICT_PAIR_CUTOFF
from .df_algo11_exchange import _count_active_rows, _cart2sph_tables
from .df_algo11_helpers_cupy import _item_table

__all__ = ['DFAlgo11ExchangeGPU', 'build_exchange_cupy', 'gamma_from_exchange_cupy',
           'J_from_exchange_cupy', 'K_from_exchange_cupy', 'occupied_fit_blocks_cupy',
           'exchange_metric_weights_cupy', 'exchange_gradient_cupy']


class DFAlgo11ExchangeGPU:
    """
    Device-resident orthonormalized three-center rows and the bookkeeping of the exchange build.

    Attributes of general interest
    ------------------------------
    nao, naux, nrows : int
        Orbital functions, fit functions (Cartesian aux for CAO, spherical for SAO), stored pairs.
    B : (nrows, naux) cupy.ndarray     rows sorted by ``(row_mu, row_nu)``
    row_mu, row_nu : (nrows,) int64 ndarray (host); ``row_mu_d``, ``row_nu_d`` on the device
    perm, inv_perm : (nao,) int64       functions sorted by the number of rows that feed them (descending)
    bins : list of (start, end, kmax)   contiguous ranges of ``perm`` that share one padded slab width
    rows_idx, src_idx : lists of (nbin, kmax) int32 cupy arrays
        Row of ``B`` / row of the density factor for every padded slot (``-1`` / ``nao`` = padding).
    stream, nb_stream : cupy.cuda.Stream and its Numba handle; all device work runs on it
    memory_gb : float
    block_memory_bytes : int or None
        Budget for the per-block work buffers (``None``: a quarter of the free device memory, at most 2 GB).
    bin_fill : float
        Minimum fill of a padded slab (class default 0.8, i.e. at most 25 % padding).
    """
    block_memory_bytes = None
    bin_fill = 0.8

    def __init__(self):
        self.B = None
        self.bins = []

    @property
    def memory_gb(self):
        return 0.0 if self.B is None else self.B.nbytes / 1e9

    def summary(self):
        pad = 0.0 if not getattr(self, 'listed_rows', 0) else 100.0 * (self.padded_cells / self.listed_rows - 1.0)
        return ('RI-HF (DF_algo=11, GPU): %d function pairs x %d %s fit functions, orthonormalized rows %.3f GB '
                'on the device; exchange in %d batched-GEMM bins (%.0f%% slab padding)'
                % (self.nrows, self.naux, self.fit_space, self.memory_gb, len(self.bins), pad))

    def aux_block_size(self, nocc, itemsize=8, block_memory_bytes=None):
        """Auxiliary functions per block so that the slabs ``U`` and the half-transformed ``X`` fit the budget."""
        budget = self.block_memory_bytes if block_memory_bytes is None else block_memory_bytes
        if budget is None:
            with cp.cuda.Device(self.device_id):
                free, _ = cp.cuda.runtime.memGetInfo()
            budget = min(2 * 1024 ** 3, free // 4)
        nocc = max(int(nocc), 1)
        per_col = int(itemsize) * (self.nao * nocc + self.max_bin_cells)
        return int(max(1, min(self.naux, int(budget) // max(per_col, 1))))


# ----------------------------------------------------------------------------
# Host bookkeeping kernels
# ----------------------------------------------------------------------------
@njit(parallel=True, cache=True, nogil=True, boundscheck=False)
def _active_rows(work, row_start, pair_I, pair_J, shell_off, shell_nbf, sqrt4, strict, mu, nu, rp, rloc):
    """Function indices, shell pair and local block row of every active row, in work-item order."""
    for w in prange(work.shape[0]):
        p = work[w]
        I = pair_I[p]
        J = pair_J[p]
        a0 = shell_off[I]
        b0 = shell_off[J]
        nA = shell_nbf[I]
        nB = shell_nbf[J]
        diag = I == J
        r = row_start[w]
        for ia in range(nA):
            ibmax = ia + 1 if diag else nB
            for ib in range(ibmax):
                if strict:
                    s = sqrt4[a0 + ia, b0 + ib]
                    if s * s < STRICT_PAIR_CUTOFF:
                        continue
                mu[r] = a0 + ia
                nu[r] = b0 + ib
                rp[r] = p
                rloc[r] = (ia * (ia + 1)) // 2 + ib if diag else ia * nB + ib
                r += 1


@njit(cache=True, nogil=True)
def _item_ranges(items, n_pairs):
    """First item and item count of every shell pair (the item table lists the items of a pair contiguously)."""
    first = np.full(n_pairs, -1, dtype=np.int64)
    count = np.zeros(n_pairs, dtype=np.int64)
    for t in range(items.shape[0]):
        p = items[t, 0]
        if first[p] < 0:
            first[p] = t
        count[p] += 1
    return first, count


# ----------------------------------------------------------------------------
# Device kernels
# ----------------------------------------------------------------------------
@cuda.jit(cache=True)
def _fill_rows_kernel(values, pair_offset, pair_ncols, row_p, row_loc, items, first_item, n_items,
                      aux_nbf, out_off, out_n, c2s_flat, c2s_off, sao, R):
    """
    One warp per active row: expand the block row of pair ``row_p[r]`` into row ``r`` of the
    (pre-zeroed) ``R``, lanes over the significant auxiliary shells of the pair.  In SAO mode
    the pseudo-Cartesian columns of a shell are contracted with its Cartesian-to-spherical matrix.
    """
    idx = cuda.grid(1)
    r = idx // 32
    lane = idx - 32 * r
    if r >= R.shape[0]:
        return
    p = row_p[r]
    base = pair_offset[p] + row_loc[r] * pair_ncols[p]
    f0 = first_item[p]
    f1 = f0 + n_items[p]
    for it in range(f0 + lane, f1, 32):
        K = items[it, 1]
        col = base + items[it, 2]
        nC = aux_nbf[K]
        o0 = out_off[K]
        if sao:
            nS = out_n[K]
            c0 = c2s_off[K]
            for s in range(nS):
                acc = 0.0
                for c in range(nC):
                    acc += c2s_flat[c0 + s * nC + c] * values[col + c]
                R[r, o0 + s] = acc
        else:
            for c in range(nC):
                R[r, o0 + c] = values[col + c]


@cuda.jit(cache=True)
def _gather_slab_kernel(B, rows_idx, Q0, U):
    """``U[b, k, q] = B[rows_idx[b, k], Q0 + q]`` (0 for padded slots); grid-stride, ``q`` fastest (coalesced)."""
    nbin, kmax, nq = U.shape
    total = nbin * kmax * nq
    stride = cuda.gridsize(1)
    for idx in range(cuda.grid(1), total, stride):
        q = idx % nq
        bk = idx // nq
        k = bk % kmax
        b = bk // kmax
        r = rows_idx[b, k]
        if r >= 0:
            U[b, k, q] = B[r, Q0 + q]
        else:
            U[b, k, q] = 0.0


def _nb(array):
    """Numba view of a CuPy array without Numba's implicit synchronization of the current stream."""
    return cuda.as_cuda_array(array, sync=False)


def _order_after_caller(stream):
    """
    Make ``stream`` wait for the work queued on the caller's current stream: the CuPy inputs of
    the public functions (density matrix, density factor, metric, plan values) were produced on
    whatever stream was current there, and CuPy's host-to-device copies are asynchronous.
    """
    current = cp.cuda.get_current_stream()
    if current.ptr != stream.ptr:
        stream.wait_event(current.record())


def _metric_cholesky_cupy(metric_d, fit_space='Cartesian'):
    """
    Lower Cholesky factor of the device fit metric, C-ordered; a metric that is not numerically
    positive definite gets ``1e-12 max(diag)`` on its diagonal (and a message), as on the CPU
    (:func:`~pyfock.Integrals.df_algo11_exchange.metric_cholesky`).
    """
    try:
        L = cp.linalg.cholesky(metric_d)
    except np.linalg.LinAlgError:
        eps = 1e-12 * float(cp.max(cp.diag(metric_d)))
        print('RI-HF: the %s auxiliary metric is not numerically positive definite; adding %.1e to its diagonal.'
              % (fit_space, eps), flush=True)
        L = cp.linalg.cholesky(metric_d + eps * cp.eye(metric_d.shape[0]))
    return cp.ascontiguousarray(L)


# ----------------------------------------------------------------------------
# Construction
# ----------------------------------------------------------------------------
def _function_structures(ex):
    """
    For every function ``i`` the rows of ``B`` that feed ``X[i]`` (own rows ``(i, k <= i)`` with
    density-factor row ``k``, partner rows ``(k > i, i)`` with factor row ``k``), the function
    order (heaviest first), the bins of similar row count and their padded index slabs.
    """
    nao = ex.nao
    mu, nu = ex.row_mu, ex.row_nu
    nrows = mu.shape[0]
    rows = np.arange(nrows, dtype=np.int64)
    off = np.nonzero(mu != nu)[0]
    fn = np.concatenate((mu, nu[off]))
    row = np.concatenate((rows, off))
    src = np.concatenate((nu, mu[off]))
    order = np.argsort(fn, kind='stable')
    ex.fn_rows = np.ascontiguousarray(row[order])
    ex.fn_src = np.ascontiguousarray(src[order])
    ex.fn_ptr = np.searchsorted(fn[order], np.arange(nao + 1, dtype=np.int64)).astype(np.int64)
    cnt = np.diff(ex.fn_ptr)
    perm = np.argsort(-cnt, kind='stable').astype(np.int64)
    ex.perm = perm
    ex.inv_perm = np.argsort(perm).astype(np.int64)
    fill = float(ex.bin_fill)
    bins = []
    s = 0
    while s < nao and cnt[perm[s]] > 0:
        kmax = int(cnt[perm[s]])
        e = s + 1
        while e < nao and cnt[perm[e]] > 0 and cnt[perm[e]] >= fill * kmax:
            e += 1
        bins.append((s, e, kmax))
        s = e
    ex.bins = bins
    ex.n_binned = s
    ex.listed_rows = int(fn.shape[0])
    ex.padded_cells = 0
    ex.rows_idx = []
    ex.src_idx = []
    for s, e, kmax in bins:
        ri = np.full((e - s, kmax), -1, dtype=np.int32)
        si = np.full((e - s, kmax), nao, dtype=np.int32)   # row nao of the extended factor is zero
        for t in range(s, e):
            i = perm[t]
            a, b = ex.fn_ptr[i], ex.fn_ptr[i + 1]
            ri[t - s, :b - a] = ex.fn_rows[a:b]
            si[t - s, :b - a] = ex.fn_src[a:b]
        ex.rows_idx.append(cp.asarray(ri))
        ex.src_idx.append(cp.asarray(si))
        ex.padded_cells += ri.size
    ex.rows_idx_nb = [_nb(a) for a in ex.rows_idx]
    ex.max_bin_cells = max(((e - s) * kmax for s, e, kmax in bins), default=0)
    ex.inv_perm_d = cp.asarray(ex.inv_perm)


def build_exchange_cupy(plan, basis, auxbasis, metric, sao=False, release_plan_values=False, cp_stream=None,
                        orthonormalize=True):
    """
    Convert a fully cached :class:`~pyfock.Integrals.df_algo11_helpers_cupy.DFAlgo11GPUPlan`
    into orthonormalized fit-space rows on the device for RI-HF.

    Parameters
    ----------
    plan : DFAlgo11GPUPlan      every significant shell pair must be cached (``max_memory_gb=None``)
    basis, auxbasis : Basis
    metric : (naux, naux) array (NumPy or CuPy)
        Positive-definite auxiliary metric in the fit space: the Cartesian ``(P|Q)`` for
        ``sao=False``, the spherical one for ``sao=True``.
    sao : bool                  must match the plan
    release_plan_values : bool
        Free the plan's device blocks as soon as the rows are filled (the plan can then no
        longer serve ``gamma_from_plan_cupy`` / ``J_from_plan_cupy``); bounds the peak memory.
    cp_stream : cupy.cuda.Stream or None
        Stream for all device work (default: the plan's stream, else the current one).
    orthonormalize : bool
        ``False`` keeps the raw integrals ``(ij|P)`` in ``B`` (``metric`` may then be ``None``): the
        exchange gradient applies the inverse metric to its much smaller occupied blocks instead
        (:func:`occupied_fit_blocks_cupy`).

    Returns
    -------
    DFAlgo11ExchangeGPU
    """
    if cp is None:
        raise ImportError('RI-HF on the GPU requires CuPy')
    values = getattr(plan, 'values', None)
    if not isinstance(values, cp.ndarray):
        raise TypeError('build_exchange_cupy needs a DF_algo=11 CUDA plan (build_plan_cupy) with its blocks on the device')
    if plan.n_pairs_cached != plan.n_pairs_significant:
        raise ValueError('RI-HF with DF_algo=11 needs every significant shell-pair block in memory '
                         '(max_memory_ints3c2e must be None); the plan caches %d of %d pairs.'
                         % (plan.n_pairs_cached, plan.n_pairs_significant))
    if bool(sao) != bool(plan.sao):
        raise ValueError('sao flag does not match the plan')
    if metric is None and orthonormalize:
        raise ValueError('orthonormalized rows need the fit metric')
    ex = DFAlgo11ExchangeGPU()
    ex.nao = int(plan.nao)
    ex.sao = bool(sao)
    ex.fit_space = 'spherical' if sao else 'Cartesian'
    ex.device_id = int(getattr(plan, 'device_id', cp.cuda.Device().id))
    if cp_stream is None:
        cp_stream = getattr(plan, 'stream', None) or cp.cuda.get_current_stream()
    ex.stream = cp_stream
    ex.nb_stream = cuda.external_stream(cp_stream.ptr)
    shell_off = np.ascontiguousarray(plan.shell_off, dtype=np.int64)
    shell_nbf = np.ascontiguousarray(plan.shell_nbf, dtype=np.int64)
    aux_nbf = np.ascontiguousarray(plan.aux_nbf, dtype=np.int64)

    if sao:
        c2s_flat, c2s_off, out_off, out_n, naux = _cart2sph_tables(auxbasis)
    else:
        naux = int(plan.naux)
        c2s_flat = np.zeros(1)
        c2s_off = np.zeros(aux_nbf.shape[0], dtype=np.int64)
        out_off = np.ascontiguousarray(plan.aux_off, dtype=np.int64)
        out_n = aux_nbf
    ex.naux = naux

    # active rows, sorted by (i, j)
    work = np.ascontiguousarray(plan.work_iter, dtype=np.int64)
    counts = _count_active_rows(work, plan.pair_I, plan.pair_J, shell_off, shell_nbf,
                                plan.sqrt_ints4c2e_diag, plan.strict_schwarz)
    row_start = np.zeros(work.shape[0] + 1, dtype=np.int64)
    row_start[1:] = np.cumsum(counts)
    nrows = int(row_start[-1])
    ex.nrows = nrows
    mu = np.zeros(nrows, dtype=np.int64)
    nu = np.zeros(nrows, dtype=np.int64)
    rp = np.zeros(nrows, dtype=np.int64)
    rloc = np.zeros(nrows, dtype=np.int64)
    if nrows:
        _active_rows(work, row_start[:-1], plan.pair_I, plan.pair_J, shell_off, shell_nbf,
                     plan.sqrt_ints4c2e_diag, plan.strict_schwarz, mu, nu, rp, rloc)
    order = np.lexsort((nu, mu))
    ex.row_mu = np.ascontiguousarray(mu[order])
    ex.row_nu = np.ascontiguousarray(nu[order])
    row_p = np.ascontiguousarray(rp[order])
    row_loc = np.ascontiguousarray(rloc[order])
    items = _item_table(work, plan.Q_pair, plan.Q_aux, plan.aux_nbf, plan.pair_ncols, plan.threshold)
    first_item, n_items = _item_ranges(items, int(plan.pair_I.shape[0]))

    need = nrows * naux * 8
    with cp.cuda.Device(ex.device_id):
        free, total = cp.cuda.runtime.memGetInfo()
    if need > 0.95 * free:
        raise MemoryError('RI-HF on the GPU needs %.2f GB for the orthonormalized three-center rows but only '
                          '%.2f GB of the %.2f GB device memory are free.' % (need / 1e9, free / 1e9, total / 1e9))

    _order_after_caller(ex.stream)
    with cp.cuda.Device(ex.device_id), ex.stream:
        if orthonormalize:
            metric_d = cp.ascontiguousarray(cp.asarray(metric, dtype=cp.float64))
            if metric_d.shape != (naux, naux):
                raise ValueError('metric has shape %s, expected (%d, %d) for the %s fit space'
                                 % (metric_d.shape, naux, naux, ex.fit_space))
        R = cp.zeros((nrows, naux), dtype=cp.float64)
        if nrows:
            dev = [cp.asarray(a) for a in (np.ascontiguousarray(plan.pair_offset, dtype=np.int64),
                                            np.ascontiguousarray(plan.pair_ncols, dtype=np.int64),
                                            row_p, row_loc, np.ascontiguousarray(items, dtype=np.int32),
                                            first_item, n_items, aux_nbf, out_off,
                                            np.ascontiguousarray(out_n, dtype=np.int64),
                                            np.ascontiguousarray(c2s_flat, dtype=np.float64),
                                            np.ascontiguousarray(c2s_off, dtype=np.int64))]
            _fill_rows_kernel[(nrows * 32 + 127) // 128, 128, ex.nb_stream](
                _nb(values), *[_nb(a) for a in dev], bool(sao), _nb(R))
            ex.stream.synchronize()   # the blocks may be released right after this
            del dev
        if release_plan_values:
            plan.values = None
        if orthonormalize:
            L = _metric_cholesky_cupy(metric_d, ex.fit_space)
            if nrows:
                # B = R L^-T, i.e. L B^T = R^T.  In column-major terms the C-ordered R is R^T (naux x nrows,
                # ld = naux) and the C-ordered lower L is the upper matrix L^T, so solve (L^T)^T Y = R^T in place.
                handle = cp.cuda.device.get_cublas_handle()
                _cublas.setStream(handle, ex.stream.ptr)
                one = np.array(1.0, dtype=np.float64)
                _cublas.dtrsm(handle, _cublas.CUBLAS_SIDE_LEFT, _cublas.CUBLAS_FILL_MODE_UPPER, _cublas.CUBLAS_OP_T,
                              _cublas.CUBLAS_DIAG_NON_UNIT, naux, nrows, one.ctypes.data, L.data.ptr, naux,
                              R.data.ptr, naux)
        ex.orthonormal = bool(orthonormalize)
        ex.B = R
        ex.row_mu_d = cp.asarray(ex.row_mu)
        ex.row_nu_d = cp.asarray(ex.row_nu)
        _function_structures(ex)
        ex.stream.synchronize()
    return ex


# ----------------------------------------------------------------------------
# Per-iteration contractions
# ----------------------------------------------------------------------------
def gamma_from_exchange_cupy(ex, dmat):
    """``gamma_Q = sum_ij D_ij B[(ij), Q]`` (full double sum) in the orthonormal fit space; CuPy result."""
    _order_after_caller(ex.stream)
    with cp.cuda.Device(ex.device_id), ex.stream:
        if ex.nrows == 0:
            return cp.zeros(ex.naux, dtype=cp.float64)
        D = cp.asarray(dmat, dtype=cp.float64)
        mu, nu = ex.row_mu_d, ex.row_nu_d
        d = D[mu, nu] + D[nu, mu] * (mu != nu)
        gamma = cp.matmul(d, ex.B)
        ex.stream.synchronize()
    return gamma


def J_from_exchange_cupy(ex, gamma):
    """Symmetric Coulomb matrix ``J_ij = sum_Q B[(ij), Q] gamma_Q`` over the stored pairs; CuPy result."""
    _order_after_caller(ex.stream)
    with cp.cuda.Device(ex.device_id), ex.stream:
        J = cp.zeros((ex.nao, ex.nao), dtype=cp.float64)
        if ex.nrows:
            jr = cp.matmul(ex.B, cp.asarray(gamma, dtype=cp.float64))
            J[ex.row_mu_d, ex.row_nu_d] = jr
            J[ex.row_nu_d, ex.row_mu_d] = jr
        ex.stream.synchronize()
    return J


def K_from_exchange_cupy(ex, factor, dtype=None, block_memory_bytes=None):
    """
    Exchange matrix ``K_ij = sum_kl D_kl (ik|jl)`` in the RI approximation for
    ``D = factor @ factor.T`` (``factor``: ``(nao, nocc)``, NumPy or CuPy).

    ``dtype`` (``cupy.float64`` default, or ``cupy.float32``) is the precision of the slabs,
    the GEMMs and the rank update; ``K`` is returned as a float64 CuPy array either way.
    """
    dt = np.dtype(cp.float64 if dtype is None else dtype)
    if dt not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise ValueError('dtype must be float32 or float64')
    nao = ex.nao
    _order_after_caller(ex.stream)
    with cp.cuda.Device(ex.device_id), ex.stream:
        F = cp.asarray(factor, dtype=cp.float64)
        if F.ndim != 2 or F.shape[0] != nao:
            raise ValueError('factor must have shape (nao, nocc)')
        nocc = int(F.shape[1])
        if nocc == 0 or ex.nrows == 0 or not ex.bins:
            ex.stream.synchronize()
            return cp.zeros((nao, nao), dtype=cp.float64)
        # density factor in the contraction precision, with a zero row for the padded slots,
        # gathered once per bin in the (bin, o, k) layout of the batched GEMM
        F_ext = cp.zeros((nao + 1, nocc), dtype=dt)
        F_ext[:nao] = F
        Ft = cp.ascontiguousarray(F_ext.T)
        Fp = [cp.ascontiguousarray(Ft[:, idx].transpose(1, 0, 2)) for idx in ex.src_idx]
        nb = ex.aux_block_size(nocc, dt.itemsize, block_memory_bytes)
        K = cp.zeros((nao, nao), dtype=dt)
        handle = cp.cuda.device.get_cublas_handle()
        _cublas.setStream(handle, ex.stream.ptr)
        # Rank update: syrk does half the flops of a GEMM and wins in fp64 and for large nao, but cuBLAS's
        # ssyrk runs 4x slower than sgemm for a few hundred functions and a long contracted axis
        # (RTX 5070: 2.0 vs 9.1 TFlop/s at nao = 510, 12 vs 8.3 TFlop/s at nao = 1500).
        rank_update_gemm = dt == np.float32 and nao < 1024
        syrk = _cublas.dsyrk if dt == np.float64 else _cublas.ssyrk
        one = np.array(1.0, dtype=dt)
        B_nb = _nb(ex.B)
        for Q0 in range(0, ex.naux, nb):
            nbq = min(nb, ex.naux - Q0)
            X = cp.zeros((nao, nocc, nbq), dtype=dt)      # rows of functions without pairs stay zero
            for (s, e, kmax), ridx, Fp_b in zip(ex.bins, ex.rows_idx_nb, Fp):
                U = cp.empty((e - s, kmax, nbq), dtype=dt)
                blocks = min((U.size + 255) // 256, 1 << 18)
                _gather_slab_kernel[blocks, 256, ex.nb_stream](B_nb, ridx, Q0, _nb(U))
                cp.matmul(Fp_b, U, out=X[s:e])
            # K += X X^T with X (nao, m) row-major = (m, nao) column-major: C = A^T A, A = X, lda = m.
            m = nocc * nbq
            if rank_update_gemm:
                Xr = X.reshape(nao, m)
                K += cp.matmul(Xr, Xr.T)
            else:
                syrk(handle, _cublas.CUBLAS_FILL_MODE_LOWER, _cublas.CUBLAS_OP_T, nao, m,
                     one.ctypes.data, X.data.ptr, m, one.ctypes.data, K.data.ptr, nao)
        if not rank_update_gemm:
            # cuBLAS filled the column-major lower = row-major upper triangle
            K = cp.triu(K)
            K = K + cp.triu(K, 1).T
        # back from the sorted function order
        K = K[ex.inv_perm_d][:, ex.inv_perm_d]
        K = K.astype(cp.float64, copy=False)
        ex.stream.synchronize()
    return K


# ----------------------------------------------------------------------------
# Exchange gradient
# ----------------------------------------------------------------------------
@cuda.jit(cache=True)
def _scatter_weights_kernel(out, rows_idx, scale, d_rows, coeff, coulomb, G):
    """``G[r, q] = scale * out[f, k, q] (+ d_rows[r] coeff[q])`` for ``r = rows_idx[f, k] >= 0``; grid-stride, ``q`` fastest."""
    nf, kmax, nq = out.shape
    total = nf * kmax * nq
    stride = cuda.gridsize(1)
    for idx in range(cuda.grid(1), total, stride):
        q = idx % nq
        fk = idx // nq
        k = fk % kmax
        f = fk // kmax
        r = rows_idx[f, k]
        if r >= 0:
            v = scale * out[f, k, q]
            if coulomb:
                v += d_rows[r] * coeff[q]
            G[r, q] = v


def _own_structures(ex):
    """
    The functions binned by their number of *own* rows ``(i, j <= i)`` (one contiguous range of
    ``B`` each), for the batched GEMMs of the gradient weights: the row and the density-factor row
    of every padded slot (``-1`` / ``nao`` = padding), with the bin rule of the exchange build.
    Runs on the current stream.
    """
    if getattr(ex, 'own_bins', None) is not None:
        return
    nao = ex.nao
    own_ptr = np.searchsorted(ex.row_mu, np.arange(nao + 1, dtype=np.int64)).astype(np.int64)
    cnt = np.diff(own_ptr)
    perm = np.argsort(-cnt, kind='stable').astype(np.int64)
    fill = float(ex.bin_fill)
    bins = []
    s = 0
    while s < nao and cnt[perm[s]] > 0:
        kmax = int(cnt[perm[s]])
        e = s + 1
        while e < nao and cnt[perm[e]] > 0 and cnt[perm[e]] >= fill * kmax:
            e += 1
        bins.append((s, e, kmax))
        s = e
    n_binned = s
    rows_idx = []
    src_idx = []
    for s, e, kmax in bins:
        ri = np.full((e - s, kmax), -1, dtype=np.int32)
        si = np.full((e - s, kmax), nao, dtype=np.int32)   # row nao of the extended factor is zero
        for t in range(s, e):
            i = perm[t]
            a, b = own_ptr[i], own_ptr[i + 1]
            ri[t - s, :b - a] = np.arange(a, b, dtype=np.int32)
            si[t - s, :b - a] = ex.row_nu[a:b]
        rows_idx.append(cp.asarray(ri))
        src_idx.append(cp.asarray(si))
    ex.own_bins = bins
    ex.own_perm_d = cp.asarray(perm[:n_binned])
    ex.own_rows_idx = rows_idx
    ex.own_rows_idx_nb = [_nb(a) for a in rows_idx]
    ex.own_src_idx = src_idx
    ex.own_max_bin_cells = max(((e - s) * kmax for s, e, kmax in bins), default=0)
    ex.own_max_bin_fns = max((e - s for s, e, _ in bins), default=0)


def occupied_fit_blocks_cupy(ex, factor, metric, block_memory_bytes=None):
    """
    Device counterpart of :func:`~pyfock.Integrals.df_algo11_exchange.occupied_fit_blocks`: the
    occupied blocks of the fitted exchange distributions for ``D = factor @ factor.T``,

        Y[P, a, b] = sum_Q [M]_PQ sum_ij F_ia (ij|Q) F_jb ,

    ``M`` the inverse fit metric, so that ``Y_P = F^T c^P F``.  The half transform is that of
    :func:`K_from_exchange_cupy` (fp64), followed by one GEMM per auxiliary block; raw rows
    (``build_exchange_cupy(..., orthonormalize=False)``) then need two triangular solves with the
    Cholesky factor of ``metric`` (the fit metric the rows were built for, NumPy or CuPy),
    orthonormalized ones one.  Returns a CuPy ``(naux, nocc, nocc)`` array, symmetric in ``(a, b)``.
    """
    from cupyx.scipy.linalg import solve_triangular
    nao = ex.nao
    naux = ex.naux
    _order_after_caller(ex.stream)
    with cp.cuda.Device(ex.device_id), ex.stream:
        F = cp.asarray(factor, dtype=cp.float64)
        if F.ndim != 2 or F.shape[0] != nao:
            raise ValueError('factor must have shape (nao, nocc)')
        if metric is None:
            raise ValueError('occupied_fit_blocks_cupy needs the fit metric the rows were built for.')
        nocc = int(F.shape[1])
        Y = cp.zeros((naux, nocc, nocc), dtype=cp.float64)
        if nocc == 0 or ex.nrows == 0 or not ex.bins:
            ex.stream.synchronize()
            return Y
        F_ext = cp.zeros((nao + 1, nocc), dtype=cp.float64)
        F_ext[:nao] = F
        Ft = cp.ascontiguousarray(F_ext.T)
        Fp = [cp.ascontiguousarray(Ft[:, idx].transpose(1, 0, 2)) for idx in ex.src_idx]
        # the factor rows in the sorted function order of the half-transformed blocks
        Fs = cp.ascontiguousarray(F[cp.asarray(ex.perm)].T)
        nb = ex.aux_block_size(nocc, 8, block_memory_bytes)
        B_nb = _nb(ex.B)
        for Q0 in range(0, naux, nb):
            nbq = min(nb, naux - Q0)
            X = cp.zeros((nao, nocc, nbq), dtype=cp.float64)      # rows of functions without pairs stay zero
            for (s, e, kmax), ridx, Fp_b in zip(ex.bins, ex.rows_idx_nb, Fp):
                U = cp.empty((e - s, kmax, nbq), dtype=cp.float64)
                blocks = min((U.size + 255) // 256, 1 << 18)
                _gather_slab_kernel[blocks, 256, ex.nb_stream](B_nb, ridx, Q0, _nb(U))
                cp.matmul(Fp_b, U, out=X[s:e])
            # Y[Q0 + q, a, b] = sum_i F[i, a] X[i, b, q]
            Yb = cp.matmul(Fs, X.reshape(nao, nocc * nbq)).reshape(nocc, nocc, nbq)
            Y[Q0:Q0 + nbq] = Yb.transpose(2, 0, 1)
            del X, Yb
        metric_d = cp.ascontiguousarray(cp.asarray(metric, dtype=cp.float64))
        if metric_d.shape != (naux, naux):
            raise ValueError('metric has shape %s, expected (%d, %d) for the %s fit space'
                             % (metric_d.shape, naux, naux, ex.fit_space))
        L = _metric_cholesky_cupy(metric_d, ex.fit_space)
        # The blocks are symmetric in (a, b): symmetrize, solve for the lower triangle only (half
        # the work of the triangular solves) and mirror it.
        a, b = cp.tril_indices(nocc)
        Yp = cp.ascontiguousarray(0.5 * (Y[:, a, b] + Y[:, b, a]))
        if getattr(ex, 'orthonormal', True):
            # rows B = R L^-T: the contraction gave L^-1 (P|Q) Y
            Yp = solve_triangular(L, Yp, lower=True, trans='T', overwrite_b=True)
        else:
            Yp = solve_triangular(L, Yp, lower=True, overwrite_b=True)
            Yp = solve_triangular(L, Yp, lower=True, trans='T', overwrite_b=True)
        Y[:, a, b] = Yp
        Y[:, b, a] = Yp
        ex.stream.synchronize()
    return Y


def exchange_metric_weights_cupy(Y):
    """
    ``W_PQ = <Y_P, Y_Q> = sum_ab Y_P[a, b] Y_Q[a, b]`` for the symmetric occupied fit blocks
    ``Y`` (CuPy ``(naux, nocc, nocc)``) of :func:`occupied_fit_blocks_cupy`: one ``syrk`` over
    their lower triangles, the off-diagonal elements scaled by ``sqrt(2)`` (a quarter of the flops
    of the full GEMM). Returns a CuPy ``(naux, naux)`` array on the current stream.
    """
    naux, nocc, _ = Y.shape
    W = cp.zeros((naux, naux), dtype=cp.float64)
    if naux == 0 or nocc == 0:
        return W
    a, b = cp.tril_indices(nocc)
    scale = cp.where(a == b, 1.0, np.sqrt(2.0))
    Yp = cp.ascontiguousarray(Y[:, a, b] * scale[None, :])
    m = int(Yp.shape[1])
    # W = Yp Yp^T with Yp (naux, m) row-major = (m, naux) column-major: C = A^T A, lda = m
    handle = cp.cuda.device.get_cublas_handle()
    _cublas.setStream(handle, cp.cuda.get_current_stream().ptr)
    one = np.array(1.0, dtype=np.float64)
    zero = np.array(0.0, dtype=np.float64)
    _cublas.dsyrk(handle, _cublas.CUBLAS_FILL_MODE_LOWER, _cublas.CUBLAS_OP_T, naux, m,
                  one.ctypes.data, Yp.data.ptr, m, zero.ctypes.data, W.data.ptr, naux)
    # cuBLAS filled the column-major lower = row-major upper triangle
    W = cp.triu(W)
    return W + cp.triu(W, 1).T


def _shell_blocks(ctx, per_col_bytes, budget):
    """Contiguous ranges ``(K0, K1)`` of auxiliary shells whose Cartesian columns fit ``budget``."""
    max_cols = max(int(budget // max(per_col_bytes, 1)), int(ctx.aux_nbf.max()))
    blocks = []
    K0 = 0
    cols = 0
    for K in range(ctx.nshells_aux):
        n = int(ctx.aux_nbf[K])
        if cols + n > max_cols and K > K0:
            blocks.append((K0, K))
            K0, cols = K, 0
        cols += n
    if ctx.nshells_aux > K0:
        blocks.append((K0, ctx.nshells_aux))
    return blocks


def exchange_gradient_cupy(ex, basis, auxbasis, plan, factor, Y, scale=1.0, dmat=None, coeff=None,
                           fit_tables=None, threshold_grad=1e-11, block_memory_bytes=None, release_rows=True):
    """
    Three-center part of the RI exchange gradient on the device,

        grad[A, d] = sum_{ij, P} W^P_ij d(ij|P)/dR_{A,d},   W^P_ij = scale * Gamma^P_ij (+ D_ij c_P),

    over both triangles of the function pairs that ``ex`` stores, with ``Gamma^P = F Y_P F^T`` for the
    density factor ``F`` (``factor``) and the occupied fit blocks ``Y`` (fit space) of
    :func:`occupied_fit_blocks_cupy`: the device counterpart of
    :func:`~pyfock.Integrals.df_algo11_exchange.gradient_rows` followed by
    :func:`~pyfock.Integrals.df_algo12_grad.grad_contract_rows`.  The optional ``D_ij c_P`` (``dmat``
    and the *Cartesian* fitting coefficients ``coeff``) adds the weights of the DF Coulomb gradient,
    so that one derivative pass serves both terms.

    The weights are formed in the Cartesian auxiliary basis for one block of whole auxiliary shells
    at a time (``Y`` is mapped onto the Cartesian functions with the per-shell tables
    ``fit_tables = (c2s_flat, c2s_off, sph_off, aux_nsph)`` of
    :func:`~pyfock.Integrals.df_algo11_exchange._cart2sph_tables` in SAO mode, ``None`` for a
    Cartesian fit space) and contracted at once with the derivative integrals of the gradient
    ``plan`` (:func:`~pyfock.Integrals.df_algo12_grad.build_grad_plan` with ``far_field=False``, the
    screening of the energy), so that they never exist in full; ``block_memory_bytes`` bounds the
    work memory of a block (default: a third of the free device memory, at most 4 GB).
    ``release_rows`` frees ``ex.B`` first: the rows are not needed once ``Y`` exists.

    Returns a NumPy ``(natoms, 3)`` array.
    """
    from .rys_3c2e_grad_contract_cupy import RowsGradContext, symmetric_row_of
    nao = ex.nao
    _order_after_caller(ex.stream)
    with cp.cuda.Device(ex.device_id), ex.stream:
        if release_rows:
            ex.B = None
        ctx = RowsGradContext(basis, auxbasis, plan, cp_stream=ex.stream)
        F = cp.asarray(factor, dtype=cp.float64)
        if F.ndim != 2 or F.shape[0] != nao:
            raise ValueError('factor must have shape (nao, nocc)')
        nocc = int(F.shape[1])
        if ex.nrows == 0 or nocc == 0:
            return ctx.gradient()
        Ym = cp.asarray(Y, dtype=cp.float64).reshape(-1, nocc * nocc)
        naux_fit = Ym.shape[0]
        coulomb = coeff is not None
        if coulomb:
            D = cp.asarray(dmat, dtype=cp.float64)
            d_rows = cp.ascontiguousarray(D[ex.row_mu_d, ex.row_nu_d])
            c_cart = cp.asarray(coeff, dtype=cp.float64)
            if c_cart.shape != (ctx.naux,):
                raise ValueError('coeff must hold the %d Cartesian fitting coefficients' % ctx.naux)
        else:
            d_rows = c_cart = cp.zeros(1, dtype=cp.float64)
        if fit_tables is None:
            if naux_fit != ctx.naux:
                raise ValueError('Y has %d fit functions, the Cartesian auxiliary basis %d' % (naux_fit, ctx.naux))
        else:
            c2s_flat, c2s_off, sph_off, aux_nsph = (np.asarray(t) for t in fit_tables)
            if naux_fit != int(aux_nsph.sum()):
                raise ValueError('Y has %d fit functions, the spherical auxiliary basis %d'
                                 % (naux_fit, int(aux_nsph.sum())))
        _own_structures(ex)
        F_ext = cp.zeros((nao + 1, nocc), dtype=cp.float64)
        F_ext[:nao] = F
        Fpad = [cp.ascontiguousarray(F_ext[idx]) for idx in ex.own_src_idx]      # (nf, kmax, nocc) per bin
        d_rows_nb = _nb(d_rows)
        row_of = symmetric_row_of(ex.row_mu, ex.row_nu, nao)
        if block_memory_bytes is None:
            free, _ = cp.cuda.runtime.memGetInfo()
            free += cp.get_default_memory_pool().free_bytes()
            block_memory_bytes = min(4 * 1024 ** 3, max(64 * 1024 ** 2, free // 3))
        # per weight column: the weights, U, the mapped Y and the largest bin's gathered U and product
        per_col = 8 * (ex.nrows + nao * nocc + 2 * nocc * nocc + ex.own_max_bin_fns * nocc + ex.own_max_bin_cells)
        for K0, K1 in _shell_blocks(ctx, per_col, block_memory_bytes):
            c0, c1 = ctx.column_range(K0, K1)
            nc = c1 - c0
            if fit_tables is None:
                Yc = Ym[c0:c1]
            else:
                s0 = int(sph_off[K0])
                s1 = int(sph_off[K1 - 1] + aux_nsph[K1 - 1])
                T = np.zeros((s1 - s0, nc))
                for K in range(K0, K1):
                    nC, nS = int(ctx.aux_nbf[K]), int(aux_nsph[K])
                    r0, q0 = int(sph_off[K]) - s0, int(ctx.aux_off[K]) - c0
                    T[r0:r0 + nS, q0:q0 + nC] = c2s_flat[c2s_off[K]:c2s_off[K] + nS * nC].reshape(nS, nC)
                Yc = cp.matmul(cp.asarray(T).T, Ym[s0:s1])
            # U[i, b, q] = sum_a F[i, a] Y[q, a, b]
            Ycc = cp.ascontiguousarray(Yc.reshape(nc, nocc, nocc).transpose(1, 2, 0)).reshape(nocc, nocc * nc)
            U = cp.matmul(F, Ycc).reshape(nao, nocc, nc)
            del Yc, Ycc
            G = cp.empty((ex.nrows, nc), dtype=cp.float64)
            coeff_blk = cp.ascontiguousarray(c_cart[c0:c1]) if coulomb else c_cart
            G_nb = _nb(G)
            coeff_nb = _nb(coeff_blk)
            for (s, e, kmax), ridx_nb, Fp in zip(ex.own_bins, ex.own_rows_idx_nb, Fpad):
                out = cp.matmul(Fp, U[ex.own_perm_d[s:e]])              # (nf, kmax, nc)
                blocks = min((out.size + 255) // 256, 1 << 18)
                _scatter_weights_kernel[blocks, 256, ex.nb_stream](
                    _nb(out), ridx_nb, float(scale), d_rows_nb, coeff_nb, coulomb, G_nb)
                del out
            del U
            ctx.contract(G, row_of, K0, K1, threshold_grad)
            del G
        grad = ctx.gradient()
    return grad
