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

Per SCF iteration (plain numpy/scipy BLAS; Numba only for gathers and scatters):

* Coulomb: ``gamma_Q = sum_r w_r D_r B[r, Q]`` (``w = 1`` for ``i == j``, else ``2``),
  ``J_r = sum_Q B[r, Q] gamma_Q`` (no metric solve: the rows are orthonormalized),
  and the DF Coulomb energy term ``gamma . gamma``.  Identical to the algorithm-11
  ``gamma -> Cholesky solve -> J`` path up to rounding.
* Exchange with the occupied density factor ``D = F F^T`` (``nocc`` columns):
  ``K_ij = sum_{Q,o} X[i, Q, o] X[j, Q, o]``, ``X[i, Q, o] = sum_k B[(ik), Q] F[k, o]``.
  The auxiliary index is processed in blocks.  For each block and each orbital
  shell ``I`` the stored rows involving a function of ``I`` are gathered into a
  dense slab ``U[partner, i in I, Q in block]`` whose ``partner`` axis runs over the
  functions that form a stored pair with the shell.  The half-transform is then one
  DGEMM per (shell, block), ``U^T @ F[partners]``, with the pair sparsity built in,
  and the block is finished with ``K += X X^T`` (numpy uses dsyrk for ``X @ X.T``).

Memory: ``B`` holds every active pair (``nrows * naux * 8`` bytes, comparable to
the algorithm-11 blocks it replaces); the per-block work buffers are bounded by
``DFAlgo11Exchange.block_memory_bytes`` (512 MB default).
"""
import numpy as np
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
    B : (nrows, naux) ndarray
    row_mu, row_nu : (nrows,) int64   function indices of each row, ``mu >= nu``
    memory_gb : float
    block_memory_bytes : int          budget for the per-aux-block work buffers
    """
    block_memory_bytes = 512 * 1024 * 1024

    def __init__(self):
        self.B = None

    @property
    def memory_gb(self):
        return 0.0 if self.B is None else self.B.nbytes / 1e9

    def summary(self):
        return ('RI-HF (DF_algo=11): %d function pairs x %d %s fit functions, orthonormalized rows %.3f GB'
                % (self.nrows, self.naux, self.fit_space, self.memory_gb))

    def aux_block_size(self, nocc, block_memory_bytes=None):
        """Auxiliary functions per block so that the largest slab and the half-transformed block fit the budget."""
        budget = self.block_memory_bytes if block_memory_bytes is None else int(block_memory_bytes)
        max_cells = int(np.diff(self.u_off).max()) if self.nshells else 1
        per_col = 8 * (max_cells + self.nao * max(int(nocc), 1))
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
def _fill_rows(work, row_start, pair_I, pair_J, pair_offset, pair_nrows, pair_ncols, values,
               shell_off, shell_nbf, sqrt4, strict, Q_pair, Q_aux, aux_off, aux_nbf, threshold,
               sao, c2s_flat, c2s_off, sph_off, aux_nsph, R, row_mu, row_nu):
    """Expand the block columns of every active row into the (pre-zeroed) fit-space rows of ``R``."""
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
        r = row_start[w]
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
                row_mu[r] = i
                row_nu[r] = j
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
                r += 1


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
def _gather_shell(B, Q0, nb, slab_row, slab_cell, s, e, ncell, U):
    """Slab of one shell for the aux block ``[Q0, Q0 + nb)``: ``U[cell, q] = B[row, Q0 + q]``, zero elsewhere."""
    for t in prange(ncell * nb):
        U[t] = 0.0
    for k in prange(s, e):
        dst = slab_cell[k] * nb
        r = slab_row[k]
        U[dst:dst + nb] = B[r, Q0:Q0 + nb]


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


def _slab_structures(ex, row_mu, row_nu):
    """
    For every orbital shell: the stored rows that involve one of its functions
    (CSR over shells), each mapped to a *cell* ``(partner, own function)`` of the
    shell's slab; the sorted partner functions per shell; and the slab offsets.
    Every row appears in the slab of ``mu`` and, if ``nu != mu``, in that of ``nu``.
    """
    nsh = ex.nshells
    nao = ex.nao
    shell_of = np.repeat(np.arange(nsh, dtype=np.int64), ex.shell_nbf)
    I_of = shell_of[row_mu]
    J_of = shell_of[row_nu]
    off = row_mu != row_nu
    shell = np.concatenate([I_of, J_of[off]])
    rows = np.concatenate([np.arange(row_mu.shape[0], dtype=np.int64), np.nonzero(off)[0].astype(np.int64)])
    own = np.concatenate([row_mu - ex.shell_off[I_of], (row_nu - ex.shell_off[J_of])[off]])
    partner = np.concatenate([row_nu, row_mu[off]])
    order = np.argsort(shell, kind='stable')
    shell, rows, own, partner = shell[order], rows[order], own[order], partner[order]
    slab_off = np.zeros(nsh + 1, dtype=np.int64)
    slab_off[1:] = np.cumsum(np.bincount(shell, minlength=nsh))
    keys = shell * nao + partner
    uniq, inverse = np.unique(keys, return_inverse=True)
    partner_off = np.zeros(nsh + 1, dtype=np.int64)
    partner_off[1:] = np.cumsum(np.bincount(uniq // nao, minlength=nsh))
    partner_loc = inverse.ravel() - partner_off[shell]
    ex.slab_off = slab_off
    ex.slab_row = np.ascontiguousarray(rows, dtype=np.int64)
    ex.slab_cell = np.ascontiguousarray(partner_loc * ex.shell_nbf[shell] + own, dtype=np.int64)
    ex.partner_off = partner_off
    ex.partner_idx = np.ascontiguousarray(uniq % nao, dtype=np.int64)
    u_off = np.zeros(nsh + 1, dtype=np.int64)
    u_off[1:] = np.cumsum(np.diff(partner_off) * ex.shell_nbf)
    ex.u_off = u_off


def build_exchange(plan, basis, auxbasis, metric, sao=False, release_plan_values=False):
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

    R = np.zeros((nrows, naux), dtype=np.float64)
    row_mu = np.zeros(nrows, dtype=np.int64)
    row_nu = np.zeros(nrows, dtype=np.int64)
    if nrows:
        _fill_rows(work, row_start[:-1], plan.pair_I, plan.pair_J, plan.pair_offset, plan.pair_nrows, plan.pair_ncols,
                   plan.values, ex.shell_off, ex.shell_nbf, plan.sqrt_ints4c2e_diag, plan.strict_schwarz,
                   plan.Q_pair, plan.Q_aux, plan.aux_off, plan.aux_nbf, plan.threshold,
                   ex.sao, c2s_flat, c2s_off, sph_off, aux_nsph, R, row_mu, row_nu)
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
    _slab_structures(ex, row_mu, row_nu)
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
    K = np.zeros((nao, nao))
    if nocc == 0 or ex.nrows == 0:
        return K
    naux = ex.naux
    bq = ex.aux_block_size(nocc, block_memory_bytes)
    ncells = np.diff(ex.u_off)
    npart = np.diff(ex.partner_off)
    U_buf = np.empty(int(ncells.max()) * bq)
    X_flat = np.empty(nao * bq * nocc)
    F_of = {I: np.ascontiguousarray(factor[ex.partner_idx[ex.partner_off[I]:ex.partner_off[I + 1]]])
            for I in range(ex.nshells) if npart[I] > 0}
    for Q0 in range(0, naux, bq):
        nb = min(bq, naux - Q0)
        X = X_flat[:nao * nb * nocc].reshape(nao, nb, nocc)
        for I in range(ex.nshells):
            a0 = ex.shell_off[I]
            nA = ex.shell_nbf[I]
            if npart[I] == 0:
                X[a0:a0 + nA] = 0.0
                continue
            # gather the slab of this shell and transform it while it is still in cache
            U = U_buf[:ncells[I] * nb]
            _gather_shell(ex.B, Q0, nb, ex.slab_row, ex.slab_cell, ex.slab_off[I], ex.slab_off[I + 1], ncells[I], U)
            np.matmul(U.reshape(npart[I], nA * nb).T, F_of[I], out=X[a0:a0 + nA].reshape(nA * nb, nocc))
        Xr = X.reshape(nao, nb * nocc)
        K += Xr @ Xr.T
    return 0.5 * (K + K.T)
