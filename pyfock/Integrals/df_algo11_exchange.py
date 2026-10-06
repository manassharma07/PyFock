"""
RI Hartree-Fock exchange (and Coulomb) from the ``DF_algo=11`` shell-pair blocks.

:mod:`~pyfock.Integrals.df_algo11_helpers` stores the screened three-center
integrals ``(ij|P)`` as one row block per significant shell pair.  For exact
exchange the metric has to be applied to the auxiliary index once, because a
per-iteration solve would cost ``naux^2 * nao * nocc``.  After the integral build
this module therefore converts the blocks into

    ``B[r, Q] = sum_P (ij|P) [L^-T]_PQ``,   ``(P|Q) = L L^T``,

one row ``r`` per stored (and strict-Schwarz-active) function pair ``i >= j``.
In SAO mode the fit space is the *true spherical* auxiliary basis: the stored
pseudo-Cartesian shells are contracted with the Cartesian-to-spherical matrices
while the rows are filled and the metric is the spherical one, so no ``1e-12``
regularization is needed and the aux dimension is the spherical count.

Row order.  The rows are sorted by ``(i, j)``, so the rows of one function ``i``
(its *own* rows, partners ``j <= i``) form one contiguous block of ``B`` that BLAS
can read in place.  The rows in which ``i`` is the smaller index (``(k, i)``,
``k > i``; its *partner* rows) are scattered over the blocks of the ``k``; they are
either copied once into a second, partner-ordered array ``P`` (one extra copy of
``B``, used when memory allows) or gathered on the fly in every iteration.

Per SCF iteration (plain numpy BLAS; Numba only for gathers and scatters):

* Coulomb: ``gamma_Q = sum_r w_r D_r B[r, Q]`` (``w = 1`` for ``i == j``, else ``2``),
  ``J_r = sum_Q B[r, Q] gamma_Q`` (no metric solve: the rows are orthonormalized),
  and the DF Coulomb energy term ``gamma . gamma``.  Identical to the algorithm-11
  ``gamma -> Cholesky solve -> J`` path up to rounding.
* Exchange with the occupied density factor ``D = F F^T`` (``nocc`` columns):
  ``K_ij = sum_{Q,o} X[i, Q, o] X[j, Q, o]``, ``X[i, Q, o] = sum_k B[(ik), Q] F[k, o]``.
  The auxiliary index is processed in blocks.  Within a block the half transform is
  one DGEMM per function ``i`` over its own rows (a contiguous row range of ``B``)
  plus one over its partner rows (``P`` or a gathered slab), ``U^T @ F[partners]``,
  with the pair sparsity built in.  The functions are distributed dynamically over
  ``numba.get_num_threads()`` Python threads that call single-threaded BLAS: the
  many medium-sized DGEMMs thread far better this way than through the BLAS
  library's own threading, and the memory-bound gathers overlap with them.  The
  block is finished with ``K += X X^T`` split over the same threads along the
  contracted ``(Q, o)`` axis (every thread computes one partial ``K`` by ``dsyrk``,
  then the partials are summed).

Memory: ``B`` holds every active pair (``nrows * naux * 8`` bytes, comparable to
the algorithm-11 blocks it replaces) and ``P``, when stored, the same again minus
the diagonal rows; the per-block work buffers are bounded by
``DFAlgo11Exchange.block_memory_bytes`` (512 MB default).
"""
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import numba
from numba import njit, prange
import scipy.linalg

from .df_algo10_helpers import STRICT_PAIR_CUTOFF

__all__ = ['DFAlgo11Exchange', 'build_exchange', 'gamma_from_exchange', 'J_from_exchange', 'K_from_exchange']


class DFAlgo11Exchange:
    """
    Orthonormalized three-center rows and the bookkeeping for the exchange build.

    Attributes of general interest
    ------------------------------
    nao, naux, nrows : int
        Orbital functions, fit functions (Cartesian aux for CAO, spherical for SAO),
        stored function pairs.
    B : (nrows, naux) ndarray    rows sorted by ``(row_mu, row_nu)``
    row_mu, row_nu : (nrows,) int64   function indices of each row, ``mu >= nu``
    own_ptr : (nao + 1,) int64   rows ``own_ptr[i]:own_ptr[i+1]`` are the own rows of ``i``
    partner_ptr, partner_row, partner_idx : int64 arrays
        CSR over ``i`` of its partner rows (row index into ``B`` and the partner ``k > i``).
    P : (n_partner_rows, naux) ndarray or None
        The partner rows copied in partner order (``P[partner_ptr[i]:partner_ptr[i+1]]``
        belongs to ``i``); ``None`` when they are gathered on the fly.
    memory_gb : float
    block_memory_bytes : int          budget for the per-aux-block work buffers
    store_partner_slabs : bool or None
        Class-level default for :func:`build_exchange`: ``None`` decides from the
        available memory, ``True``/``False`` force the choice.
    """
    block_memory_bytes = 512 * 1024 * 1024
    store_partner_slabs = None

    def __init__(self):
        self.B = None
        self.P = None

    @property
    def memory_gb(self):
        total = 0.0 if self.B is None else self.B.nbytes
        if self.P is not None:
            total += self.P.nbytes
        return total / 1e9

    def summary(self):
        extra = (', partner rows stored (%.3f GB)' % (self.P.nbytes / 1e9) if self.P is not None
                 else ', partner rows gathered per iteration')
        return ('RI-HF (DF_algo=11): %d function pairs x %d %s fit functions, orthonormalized rows %.3f GB'
                % (self.nrows, self.naux, self.fit_space, 0.0 if self.B is None else self.B.nbytes / 1e9) + extra)

    def aux_block_size(self, nocc, block_memory_bytes=None, nthreads=None):
        """Auxiliary functions per block so that the half-transformed block and the per-thread buffers fit the budget."""
        budget = self.block_memory_bytes if block_memory_bytes is None else int(block_memory_bytes)
        nthreads = max(1, int(numba.get_num_threads() if nthreads is None else nthreads))
        nocc = max(int(nocc), 1)
        gather = 0 if self.P is not None else int(self.max_partner_rows)
        per_col = 8 * (self.nao * nocc + nthreads * (nocc + gather))
        return int(max(1, min(self.naux, budget // max(per_col, 1))))


# ----------------------------------------------------------------------------
# Numba kernels
# ----------------------------------------------------------------------------
@njit(parallel=True, cache=True, nogil=True, boundscheck=False)
def _count_active_rows(work, pair_I, pair_J, shell_off, shell_nbf, sqrt4, strict):
    counts = np.zeros(work.shape[0], dtype=np.int64)
    for w in prange(work.shape[0]):
        p = work[w]
        I = pair_I[p]
        J = pair_J[p]
        a0 = shell_off[I]
        b0 = shell_off[J]
        nA = shell_nbf[I]
        nB = shell_nbf[J]
        diag = I == J
        c = 0
        for ia in range(nA):
            ibmax = ia + 1 if diag else nB
            for ib in range(ibmax):
                if strict:
                    s = sqrt4[a0 + ia, b0 + ib]
                    if s * s < STRICT_PAIR_CUTOFF:
                        continue
                c += 1
        counts[w] = c
    return counts


@njit(parallel=True, cache=True, nogil=True, boundscheck=False)
def _active_pairs(work, row_start, pair_I, pair_J, shell_off, shell_nbf, sqrt4, strict, mu, nu):
    """Function indices ``(i >= j)`` of every active row, in work-item order."""
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
                r += 1


@njit(parallel=True, cache=True, nogil=True, boundscheck=False)
def _fill_rows(work, pair_I, pair_J, pair_offset, pair_nrows, pair_ncols, values,
               shell_off, shell_nbf, sqrt4, strict, Q_pair, Q_aux, aux_off, aux_nbf, threshold,
               sao, c2s_flat, c2s_off, sph_off, aux_nsph, row_of, R):
    """Expand the block columns of every active row ``(i, j)`` into row ``row_of[i, j]`` of the (pre-zeroed) ``R``."""
    nsh_aux = aux_off.shape[0]
    for w in prange(work.shape[0]):
        p = work[w]
        I = pair_I[p]
        J = pair_J[p]
        a0 = shell_off[I]
        b0 = shell_off[J]
        nA = shell_nbf[I]
        nB = shell_nbf[J]
        diag = I == J
        nr = pair_nrows[p]
        nc = pair_ncols[p]
        o = pair_offset[p]
        rows = values[o:o + nr * nc].reshape((nr, nc))
        Qp = Q_pair[p]
        for ia in range(nA):
            ibmax = ia + 1 if diag else nB
            for ib in range(ibmax):
                i = a0 + ia
                j = b0 + ib
                if strict:
                    s = sqrt4[i, j]
                    if s * s < STRICT_PAIR_CUTOFF:
                        continue
                rloc = (ia * (ia + 1)) // 2 + ib if diag else ia * nB + ib
                r = row_of[i, j]
                col = 0
                for K in range(nsh_aux):
                    if Qp * Q_aux[K] > threshold:
                        nC = aux_nbf[K]
                        if sao:
                            nS = aux_nsph[K]
                            c0 = c2s_off[K]
                            s0 = sph_off[K]
                            for s in range(nS):
                                acc = 0.0
                                for c in range(nC):
                                    acc += c2s_flat[c0 + s * nC + c] * rows[rloc, col + c]
                                R[r, s0 + s] = acc
                        else:
                            k0 = aux_off[K]
                            for c in range(nC):
                                R[r, k0 + c] = rows[rloc, col + c]
                        col += nC


@njit(parallel=True, cache=True, nogil=True, boundscheck=False)
def _row_density(dmat, row_mu, row_nu):
    """d_r = D_ij (i == j) or D_ij + D_ji (i != j): the weights of the double sum over i, j."""
    d = np.empty(row_mu.shape[0])
    for r in prange(row_mu.shape[0]):
        i = row_mu[r]
        j = row_nu[r]
        d[r] = dmat[i, j] if i == j else dmat[i, j] + dmat[j, i]
    return d


@njit(parallel=True, cache=True, nogil=True, boundscheck=False)
def _scatter_symmetric(jrows, row_mu, row_nu, nao):
    J = np.zeros((nao, nao))
    for r in prange(row_mu.shape[0]):
        i = row_mu[r]
        j = row_nu[r]
        J[i, j] = jrows[r]
        J[j, i] = jrows[r]
    return J


@njit(parallel=True, cache=True, nogil=True, boundscheck=False)
def _copy_rows(B, rows, out):
    """``out[k, :] = B[rows[k], :]`` (the stored partner rows)."""
    for k in prange(rows.shape[0]):
        out[k, :] = B[rows[k], :]


@njit(cache=True, nogil=True, boundscheck=False)
def _gather_rows(B, Q0, nb, rows, k0, k1, U):
    """``U[k - k0, :] = B[rows[k], Q0:Q0 + nb]`` for ``k0 <= k < k1`` (serial: called from worker threads)."""
    for k in range(k0, k1):
        U[k - k0, :] = B[rows[k], Q0:Q0 + nb]


# ----------------------------------------------------------------------------
# Construction
# ----------------------------------------------------------------------------
def _cart2sph_tables(auxbasis):
    """Packed per-shell Cartesian->spherical matrices of the auxiliary basis."""
    from ..Basis import Basis  # local import: Basis imports Numba kernels of its own
    nsh = auxbasis.nshells
    mats = [np.ascontiguousarray(Basis.cart2sph(int(auxbasis.shells[K]) - 1), dtype=np.float64) for K in range(nsh)]
    aux_nsph = np.array([m.shape[0] for m in mats], dtype=np.int64)
    ncart = np.array([m.shape[1] for m in mats], dtype=np.int64)
    if not np.array_equal(ncart, np.asarray(auxbasis.bfs_nbfshell, dtype=np.int64)):
        raise ValueError('cart2sph tables do not match the auxiliary shell sizes')
    c2s_off = np.zeros(nsh + 1, dtype=np.int64)
    c2s_off[1:] = np.cumsum(aux_nsph * ncart)
    c2s_flat = np.concatenate([m.ravel() for m in mats]) if nsh else np.zeros(0)
    sph_off = np.zeros(nsh + 1, dtype=np.int64)
    sph_off[1:] = np.cumsum(aux_nsph)
    return c2s_flat, c2s_off[:-1], sph_off[:-1], aux_nsph, int(sph_off[-1])


def _partner_structures(ex, store_partner_slabs):
    """
    CSR of the own rows (by ``row_mu``, already contiguous in the sorted row order) and of the
    partner rows (rows ``(k, i)`` with ``k > i`` listed under ``i``, sorted by ``k``); the
    partner rows are copied into ``ex.P`` when requested (``None``: when they fit comfortably
    into the available memory).
    """
    nao = ex.nao
    row_mu, row_nu = ex.row_mu, ex.row_nu
    ex.own_ptr = np.searchsorted(row_mu, np.arange(nao + 1, dtype=np.int64)).astype(np.int64)
    off = np.nonzero(row_mu != row_nu)[0].astype(np.int64)
    order = np.lexsort((row_mu[off], row_nu[off]))
    prow = np.ascontiguousarray(off[order], dtype=np.int64)
    ex.partner_row = prow
    ex.partner_idx = np.ascontiguousarray(row_mu[prow], dtype=np.int64)
    ex.partner_ptr = np.searchsorted(row_nu[prow], np.arange(nao + 1, dtype=np.int64)).astype(np.int64)
    counts = np.diff(ex.partner_ptr)
    ex.max_partner_rows = int(counts.max()) if counts.size else 0
    need = prow.shape[0] * ex.naux * 8
    store = store_partner_slabs
    if store is None:
        try:
            import psutil
            available = psutil.virtual_memory().available
        except Exception:
            available = None
        # B is already allocated; keep the copy well below what is left for the work buffers
        store = available is not None and need < 0.35 * available
    ex.P = None
    if store and prow.shape[0]:
        ex.P = np.empty((prow.shape[0], ex.naux))
        _copy_rows(ex.B, prow, ex.P)


def build_exchange(plan, basis, auxbasis, metric, sao=False, release_plan_values=False, store_partner_slabs=None):
    """
    Convert a fully cached :class:`~pyfock.Integrals.df_algo11_helpers.DFAlgo11Plan`
    into orthonormalized fit-space rows for RI-HF.

    Parameters
    ----------
    plan : DFAlgo11Plan        every significant shell pair must be cached
    basis, auxbasis : Basis
    metric : (naux, naux) ndarray
        Positive-definite auxiliary metric in the fit space: the Cartesian
        ``(P|Q)`` for ``sao=False``, the spherical one for ``sao=True``.
    sao : bool                 must match the plan
    release_plan_values : bool
        Set ``plan.values = None`` as soon as the rows are copied (the plan can then
        no longer serve ``gamma_from_plan`` / ``J_from_plan``); bounds the peak memory.
    store_partner_slabs : bool or None
        Keep a partner-ordered copy of the off-diagonal rows for the exchange build
        (see the module docstring).  ``None`` uses ``DFAlgo11Exchange.store_partner_slabs``
        (by default decided from the available memory).

    Returns
    -------
    DFAlgo11Exchange
    """
    if plan.n_pairs_cached != plan.n_pairs_significant:
        raise ValueError('RI-HF with DF_algo=11 needs every significant shell-pair block in memory '
                         '(max_memory_ints3c2e must be None); the plan caches %d of %d pairs.'
                         % (plan.n_pairs_cached, plan.n_pairs_significant))
    if bool(sao) != bool(plan.sao):
        raise ValueError('sao flag does not match the plan')
    ex = DFAlgo11Exchange()
    ex.nao = int(plan.nao)
    ex.shell_off = np.ascontiguousarray(plan.shell_off, dtype=np.int64)
    ex.shell_nbf = np.ascontiguousarray(plan.shell_nbf, dtype=np.int64)
    ex.nshells = int(ex.shell_off.shape[0])
    ex.sao = bool(sao)
    ex.fit_space = 'spherical' if sao else 'Cartesian'

    if sao:
        c2s_flat, c2s_off, sph_off, aux_nsph, naux = _cart2sph_tables(auxbasis)
    else:
        naux = int(plan.naux)
        c2s_flat = np.zeros(1)
        c2s_off = np.zeros(plan.aux_off.shape[0], dtype=np.int64)
        sph_off = np.ascontiguousarray(plan.aux_off, dtype=np.int64)
        aux_nsph = np.ascontiguousarray(plan.aux_nbf, dtype=np.int64)
    ex.naux = naux
    metric = np.ascontiguousarray(metric, dtype=np.float64)
    if metric.shape != (naux, naux):
        raise ValueError('metric has shape %s, expected (%d, %d) for the %s fit space'
                         % (metric.shape, naux, naux, ex.fit_space))

    work = np.ascontiguousarray(plan.work_iter, dtype=np.int64)
    counts = _count_active_rows(work, plan.pair_I, plan.pair_J, ex.shell_off, ex.shell_nbf,
                                plan.sqrt_ints4c2e_diag, plan.strict_schwarz)
    row_start = np.zeros(work.shape[0] + 1, dtype=np.int64)
    row_start[1:] = np.cumsum(counts)
    nrows = int(row_start[-1])
    ex.nrows = nrows

    need = nrows * naux * 8
    try:
        import psutil
        available = psutil.virtual_memory().available
    except Exception:
        available = None
    if available is not None and need > 0.9 * available:
        raise MemoryError('RI-HF with DF_algo=11 needs %.2f GB for the orthonormalized three-center rows '
                          'but only %.2f GB are available.' % (need / 1e9, available / 1e9))

    # rows sorted by (i, j): the own rows of every function are contiguous
    mu = np.zeros(nrows, dtype=np.int64)
    nu = np.zeros(nrows, dtype=np.int64)
    if nrows:
        _active_pairs(work, row_start[:-1], plan.pair_I, plan.pair_J, ex.shell_off, ex.shell_nbf,
                      plan.sqrt_ints4c2e_diag, plan.strict_schwarz, mu, nu)
    order = np.lexsort((nu, mu))
    row_mu = np.ascontiguousarray(mu[order])
    row_nu = np.ascontiguousarray(nu[order])
    row_of = np.full((ex.nao, ex.nao), -1, dtype=np.int64)
    row_of[row_mu, row_nu] = np.arange(nrows, dtype=np.int64)

    R = np.zeros((nrows, naux), dtype=np.float64)
    if nrows:
        _fill_rows(work, plan.pair_I, plan.pair_J, plan.pair_offset, plan.pair_nrows, plan.pair_ncols,
                   plan.values, ex.shell_off, ex.shell_nbf, plan.sqrt_ints4c2e_diag, plan.strict_schwarz,
                   plan.Q_pair, plan.Q_aux, plan.aux_off, plan.aux_nbf, plan.threshold,
                   ex.sao, c2s_flat, c2s_off, sph_off, aux_nsph, row_of, R)
    if release_plan_values:
        plan.values = None

    try:
        L = scipy.linalg.cholesky(metric, lower=True, check_finite=False)
    except scipy.linalg.LinAlgError:
        eps = 1e-12 * float(np.max(np.diag(metric)))
        print('RI-HF: the %s auxiliary metric is not numerically positive definite; adding %.1e to its diagonal.'
              % (ex.fit_space, eps), flush=True)
        L = scipy.linalg.cholesky(metric + eps * np.eye(naux), lower=True, check_finite=False)
    if nrows:
        # B = R L^-T, i.e. L Y = R^T with Y = B^T.  R^T is the Fortran-ordered view of R, so
        # LAPACK can solve in place; copy back only if scipy had to make a copy.
        Y = scipy.linalg.solve_triangular(L, R.T, lower=True, overwrite_b=True, check_finite=False)
        if not np.shares_memory(Y, R):
            R[...] = Y.T
    ex.B = R
    ex.row_mu = row_mu
    ex.row_nu = row_nu
    if store_partner_slabs is None:
        store_partner_slabs = DFAlgo11Exchange.store_partner_slabs
    _partner_structures(ex, store_partner_slabs)
    return ex


# ----------------------------------------------------------------------------
# Per-iteration contractions
# ----------------------------------------------------------------------------
def gamma_from_exchange(ex, dmat):
    """``gamma_Q = sum_ij D_ij B[(ij), Q]`` (full double sum) in the orthonormal fit space."""
    dmat = np.ascontiguousarray(dmat, dtype=np.float64)
    if ex.nrows == 0:
        return np.zeros(ex.naux)
    return _row_density(dmat, ex.row_mu, ex.row_nu) @ ex.B


def J_from_exchange(ex, gamma):
    """Symmetric Coulomb matrix ``J_ij = sum_Q B[(ij), Q] gamma_Q`` over the stored pairs."""
    gamma = np.ascontiguousarray(gamma, dtype=np.float64)
    if ex.nrows == 0:
        return np.zeros((ex.nao, ex.nao))
    return _scatter_symmetric(ex.B @ gamma, ex.row_mu, ex.row_nu, ex.nao)


_BLAS_CONTROLLER = None
_POOLS = {}


def _blas_threads(n):
    """Context manager limiting the BLAS libraries to ``n`` threads (the controller is scanned once)."""
    global _BLAS_CONTROLLER
    if _BLAS_CONTROLLER is None:
        from threadpoolctl import ThreadpoolController
        _BLAS_CONTROLLER = ThreadpoolController()
    return _BLAS_CONTROLLER.limit(limits=n, user_api='blas')


def _pool(nthreads):
    pool = _POOLS.get(nthreads)
    if pool is None:
        pool = _POOLS[nthreads] = ThreadPoolExecutor(max_workers=nthreads, thread_name_prefix='pyfock-rik')
    return pool


def K_from_exchange(ex, factor, block_memory_bytes=None):
    """
    Exchange matrix ``K_ij = sum_kl D_kl (ik|jl)`` in the RI approximation for
    ``D = factor @ factor.T`` (``factor``: ``(nao, nocc)``).
    """
    factor = np.ascontiguousarray(factor, dtype=np.float64)
    nao = ex.nao
    if factor.ndim != 2 or factor.shape[0] != nao:
        raise ValueError('factor must have shape (nao, nocc)')
    nocc = factor.shape[1]
    if nocc == 0 or ex.nrows == 0:
        return np.zeros((nao, nao))
    naux = ex.naux
    nthreads = max(1, int(numba.get_num_threads()))
    bq = ex.aux_block_size(nocc, block_memory_bytes, nthreads)
    B, P = ex.B, ex.P
    own_ptr, pptr, prow, pidx, row_nu = ex.own_ptr, ex.partner_ptr, ex.partner_row, ex.partner_idx, ex.row_nu
    n_own = np.diff(own_ptr)
    n_par = np.diff(pptr)
    # density-factor rows of the partners of every function (small gathers, once per build)
    F_own = [factor[row_nu[own_ptr[i]:own_ptr[i + 1]]] for i in range(nao)]
    F_par = [factor[pidx[pptr[i]:pptr[i + 1]]] for i in range(nao)]
    order = np.argsort(-(n_own + n_par), kind='stable')   # heaviest functions first
    X_flat = np.empty(nao * bq * nocc)
    Kparts = np.zeros((nthreads, nao, nao))
    local = threading.local()
    gather_cells = 0 if P is not None else ex.max_partner_rows

    def buffers():
        buf = getattr(local, 'buf', None)
        if buf is None:
            buf = local.buf = (np.empty(bq * nocc), np.empty(gather_cells * bq))
        return buf

    pool = _pool(nthreads) if nthreads > 1 else None

    def run(fn, items):
        if pool is None:
            for it in items:
                fn(it)
        else:
            list(pool.map(fn, items))

    with _blas_threads(1):
        for Q0 in range(0, naux, bq):
            nb = min(bq, naux - Q0)
            X = X_flat[:nao * nb * nocc].reshape(nao, nb, nocc)

            def half_transform(i, Q0=Q0, nb=nb, X=X):
                Xi = X[i]
                r0, r1 = own_ptr[i], own_ptr[i + 1]
                if r1 > r0:
                    np.matmul(B[r0:r1, Q0:Q0 + nb].T, F_own[i], out=Xi)
                else:
                    Xi[...] = 0.0
                p0, p1 = pptr[i], pptr[i + 1]
                if p1 > p0:
                    tmp, gbuf = buffers()
                    if P is not None:
                        U = P[p0:p1, Q0:Q0 + nb]
                    else:
                        U = gbuf[:(p1 - p0) * nb].reshape(p1 - p0, nb)
                        _gather_rows(B, Q0, nb, prow, p0, p1, U)
                    t = tmp[:nb * nocc].reshape(nb, nocc)
                    np.matmul(U.T, F_par[i], out=t)
                    Xi += t

            run(half_transform, order)
            Xr = X.reshape(nao, nb * nocc)
            bounds = np.linspace(0, nb * nocc, nthreads + 1).astype(np.int64)

            def rank_update(t, Xr=Xr, bounds=bounds):
                if bounds[t + 1] > bounds[t]:
                    Xc = Xr[:, bounds[t]:bounds[t + 1]]
                    Kparts[t] += Xc @ Xc.T   # numpy dispatches A @ A.T to dsyrk

            run(rank_update, range(nthreads))
    K = Kparts.sum(axis=0)
    return 0.5 * (K + K.T)
