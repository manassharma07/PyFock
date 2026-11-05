"""
Exchange-correlation energy and potential matrix on the CPU: algorithm 3.

The grid is cut into small, spatially compact batches ("leaves") by recursive median
bisection, and every leaf gets its own list of the shells whose radial cutoff reaches at
least one of its points. Because a leaf spans a fraction of a Bohr in the dense parts of
the grid, these lists are close to the per-point minimum (a few per cent above it), while
the batches of algorithm 2 are hundreds of times larger and gather several times as many
functions, so the dense matrix products that dominate the cost shrink by an order of
magnitude.

Consecutive leaves are grouped into macro blocks (a sub-tree of the bisection), which
gather their sub-matrix of the density once and accumulate their part of the potential
matrix once. The macro blocks are distributed over the threads in advance (longest
processing time first, from a cost model of the matrix products), and one Numba parallel
region evaluates everything: basis functions (shell by shell, sharing the exponentials
between the Cartesian components, with primitives screened point by point), densities,
the functional, and the contractions, which call BLAS through ``np.dot`` with one thread
each. No Python runs inside the parallel region.

The potential matrix contribution of every macro block is kept in its own slot and
assembled afterwards by row bands, so the result does not depend on the number of
threads and no thread ever writes to another thread's rows.

The native functionals (LDA, GGA, meta-GGA and the global hybrids built from them) are
evaluated inside the parallel region, in one pass over the grid (the meta-GGAs and PW91
correlation point by point through :mod:`pyfock.XC.pointwise`). LibXC functionals are
evaluated on the whole grid between two passes, with the functional itself spread over the
threads in chunks.

The values of the basis functions (and their gradients) can be kept between SCF
iterations (``save_ao_values``): the cache is stored leaf by leaf, in the same compact
form the kernel reads, so it needs considerably less memory than the per-batch cache of
algorithm 2 and no copying when it is used.

The Numba kernels are in :mod:`pyfock.Integrals.eval_xc_3_helpers`.
"""
import heapq
import weakref
from concurrent.futures import ThreadPoolExecutor
from timeit import default_timer as timer

import numpy as np
import numba
from threadpoolctl import threadpool_limits

from pyfock import XC
from .eval_xc_2 import _xc_functionals
from .eval_xc_3_helpers import (INKERNEL_FUNCTIONALS, LMAX, _kd_order, _screen_leaves_mask, _mask_to_csr,
                                _expand_shells_to_bf, _build_ao_cache_kernel, _xc3_kernel, _assemble_v,
                                xc_workspace_size)

__all__ = ['eval_xc_3', 'XCPlan', 'build_xc_plan', 'AOCache', 'INKERNEL_FUNCTIONALS']

# ---------------------------------------------------------------------------------------
# Plan: shell data, partition, screening, schedule, optional AO cache
# ---------------------------------------------------------------------------------------
def shell_data(basis):
    """Pack the shell structure of ``basis`` into the arrays the kernels take.

    Returns a dict with ``sh_coords (nsh, 3)``, ``sh_rcut (nsh,)``, ``sh_rcut2``, ``sh_nprim``,
    ``sh_expnts (nsh, maxprim)``, ``sh_bf0``, ``sh_nbf``, ``sh_L``, ``sh_atom``, ``bf_lmn (nbf, 3)``
    and ``bf_coef (nbf, maxprim)`` (contraction coefficient times both normalization factors,
    exactly what the per-function kernels use).
    """
    nbf = basis.bfs_nao
    shell_index = np.asarray(basis.bfs_shell_index, dtype=np.int64)
    if nbf > 0 and np.any(np.diff(shell_index) < 0):
        raise ValueError('eval_xc_3 expects the basis functions of a shell to be contiguous.')
    nsh = int(shell_index.max()) + 1 if nbf > 0 else 0
    sh_nbf = np.bincount(shell_index, minlength=nsh).astype(np.int64)
    sh_bf0 = np.zeros(nsh, dtype=np.int64)
    sh_bf0[1:] = np.cumsum(sh_nbf)[:-1]
    bfs_nprim = np.asarray(basis.bfs_nprim, dtype=np.int64)
    maxprim = int(bfs_nprim.max()) if nbf > 0 else 1
    sh_coords = np.ascontiguousarray(np.asarray(basis.bfs_coords, dtype=np.float64).reshape(nbf, 3)[sh_bf0])
    sh_nprim = bfs_nprim[sh_bf0]
    sh_expnts = np.zeros((nsh, maxprim))
    bf_coef = np.zeros((nbf, maxprim))
    for i in range(nbf):
        npr = bfs_nprim[i]
        Ni = basis.bfs_contr_prim_norms[i]
        for p in range(npr):
            bf_coef[i, p] = Ni * basis.bfs_prim_norms[i][p] * basis.bfs_coeffs[i][p]
    for s in range(nsh):
        i = sh_bf0[s]
        sh_expnts[s, :bfs_nprim[i]] = np.asarray(basis.bfs_expnts[i][:bfs_nprim[i]], dtype=np.float64)
    bf_lmn = np.ascontiguousarray(np.asarray(basis.bfs_lmn, dtype=np.int64).reshape(nbf, 3))
    sh_L = bf_lmn.sum(axis=1)[sh_bf0]
    if nsh > 0 and sh_L.max() > LMAX:
        raise ValueError('eval_xc_3 supports angular momenta up to l=' + str(LMAX) + '.')
    rcut = np.asarray(basis.bfs_radius_cutoff, dtype=np.float64)
    sh_rcut = np.zeros(nsh)
    for s in range(nsh):
        sh_rcut[s] = rcut[sh_bf0[s]:sh_bf0[s] + sh_nbf[s]].max()
    sh_atom = np.asarray(basis.bfs_atoms, dtype=np.int64)[sh_bf0] if nbf > 0 else np.zeros(0, np.int64)
    return dict(sh_coords=sh_coords, sh_rcut=sh_rcut, sh_rcut2=sh_rcut * sh_rcut, sh_nprim=sh_nprim,
                sh_expnts=sh_expnts, sh_bf0=sh_bf0, sh_nbf=sh_nbf, sh_L=np.ascontiguousarray(sh_L),
                sh_atom=sh_atom, bf_lmn=bf_lmn, bf_coef=bf_coef, nbf=nbf, nsh=nsh)


class AOCache:
    """Values (``nd = 1``) or values and gradients (``nd = 4``) of the basis functions on every
    leaf of a plan, stored leaf by leaf in the layout the kernel reads."""

    def __init__(self, plan, nd, data, ptr):
        self.plan = plan
        self.nd = nd
        self.data = data
        self.ptr = ptr

    @property
    def nbytes(self):
        return self.data.nbytes


class XCPlan:
    """Partition of a grid into leaves and macro blocks with their shell lists, for a basis.

    Built once per grid (``build_xc_plan``) and reused by every call of :func:`eval_xc_3`
    (and of the gradient driver) on that grid.
    """

    def __init__(self, basis, coords, weights, leaf_size=256, leaves_per_macro=16, ncores=1, screen=True,
                 point_cutoff=True):
        t0 = timer()
        coords = np.ascontiguousarray(coords, dtype=np.float64).reshape(-1, 3)
        weights = np.ascontiguousarray(weights, dtype=np.float64).reshape(-1)
        if weights.shape[0] != coords.shape[0]:
            raise ValueError('coords and weights have different lengths.')
        self.ngrids = coords.shape[0]
        self._sample_coords = coords[::997].copy()
        self._sample_weights = weights[::991].copy()
        self.leaf_size = int(leaf_size)
        self.leaves_per_macro = int(leaves_per_macro)
        self.screen = bool(screen)
        sd = shell_data(basis)
        if not point_cutoff:
            # every listed shell is evaluated at every point of the leaf (no per-point radial cutoff)
            sd['sh_rcut2'] = np.full_like(sd['sh_rcut2'], np.inf)
        self.shell = sd
        self.nbf = sd['nbf']
        self.nsh = sd['nsh']
        nsh = self.nsh
        # ---- spatial ordering into leaves
        perm, leaf_lo, leaf_hi = _kd_order(coords, self.leaf_size)
        self.perm = perm
        self.coords = np.ascontiguousarray(coords[perm])
        self.weights = np.ascontiguousarray(weights[perm])
        self.leaf_lo = leaf_lo
        self.leaf_hi = leaf_hi
        nleaf = leaf_lo.shape[0]
        self.nleaf = nleaf
        # ---- shells per leaf
        mask = np.zeros((nleaf, nsh), dtype=np.uint8)
        if self.screen:
            _screen_leaves_mask(self.coords, leaf_lo, leaf_hi, sd['sh_coords'], sd['sh_rcut'], mask)
        else:
            mask[:] = 1
        self.leaf_sh_ptr, self.leaf_sh = _mask_to_csr(mask)
        leaf_bf_ptr, _ = _expand_shells_to_bf(self.leaf_sh_ptr, self.leaf_sh, sd['sh_bf0'], sd['sh_nbf'])
        self.leaf_nbf = np.diff(leaf_bf_ptr)
        # ---- macro blocks: runs of consecutive leaves
        lpm = max(1, self.leaves_per_macro)
        nmacro = (nleaf + lpm - 1) // lpm
        self.nmacro = nmacro
        self.macro_leaf_lo = np.arange(nmacro, dtype=np.int64) * lpm
        self.macro_leaf_hi = np.minimum(self.macro_leaf_lo + lpm, nleaf)
        mmask = np.zeros((nmacro, nsh), dtype=np.uint8)
        for ib in range(nmacro):
            mmask[ib] = mask[self.macro_leaf_lo[ib]:self.macro_leaf_hi[ib]].max(axis=0)
        self.macro_sh_ptr, self.macro_sh = _mask_to_csr(mmask)
        self.macro_bf_ptr, self.macro_bf = _expand_shells_to_bf(self.macro_sh_ptr, self.macro_sh,
                                                                sd['sh_bf0'], sd['sh_nbf'])
        self.macro_nbf = np.diff(self.macro_bf_ptr)
        self.store_ptr = np.zeros(nmacro + 1, dtype=np.int64)
        self.store_ptr[1:] = np.cumsum(self.macro_nbf.astype(np.int64) ** 2)
        self.max_k = int((leaf_hi - leaf_lo).max()) if nleaf > 0 else 0
        self.max_nb = int(self.leaf_nbf.max()) if nleaf > 0 else 0
        self.max_nbM = int(self.macro_nbf.max()) if nmacro > 0 else 0
        # ---- cost model of every macro block (for the static schedule); the matrix products
        # cost 4 k n^2 flops per leaf, the rest is linear in k n
        k = (leaf_hi - leaf_lo).astype(np.float64)
        n = self.leaf_nbf.astype(np.float64)
        leaf_gemm = 4.0 * k * n * n
        leaf_lin = k * n
        self._macro_gemm = np.add.reduceat(leaf_gemm, self.macro_leaf_lo) if nleaf > 0 else np.zeros(0)
        self._macro_lin = np.add.reduceat(leaf_lin, self.macro_leaf_lo) if nleaf > 0 else np.zeros(0)
        self._bins = {}
        self._ws = {}
        self.cache = None
        self.build_time = timer() - t0
        self._keys = None

    def workspace(self, nthreads, ndoubles, nints):
        """Per-thread workspaces ``(ws (nthreads, ndoubles), iws (nthreads, nints))``, allocated once
        and reused by every call on this plan (so that no pages are faulted in during the kernel)."""
        key = (int(nthreads), int(ndoubles), int(nints))
        if key not in self._ws:
            self._ws.clear()
            self._ws[key] = (np.empty((int(nthreads), int(ndoubles))), np.empty((int(nthreads), int(nints)), dtype=np.int64))
        return self._ws[key]

    def matches(self, coords, weights):
        """Whether the plan was built for these grid arrays (same length, same sampled values)."""
        coords = np.asarray(coords)
        weights = np.asarray(weights)
        if coords.shape[0] != self.ngrids or weights.shape[0] != self.ngrids:
            return False
        if self.ngrids == 0:
            return True
        return bool(np.array_equal(coords[::997], self._sample_coords) and np.array_equal(weights[::991], self._sample_weights))

    # ---- statistics
    def describe(self):
        k = self.leaf_hi - self.leaf_lo
        return ('XC plan: ' + str(self.ngrids) + ' points, ' + str(self.nleaf) + ' leaves of <= ' + str(self.leaf_size)
                + ' points (' + str(int(np.round(k.mean()))) + ' on average), ' + str(self.nmacro)
                + ' macro blocks; basis functions per leaf: ' + str(int(np.round(self.leaf_nbf.mean())))
                + ' on average, ' + str(int(self.max_nb)) + ' at most (per macro block: '
                + str(int(np.round(self.macro_nbf.mean()))) + ' / ' + str(int(self.max_nbM)) + '); built in '
                + str(round(self.build_time, 2)) + ' s')

    def ao_cache_bytes(self, nd):
        k = (self.leaf_hi - self.leaf_lo).astype(np.int64)
        return int(nd) * int(np.sum(k * self.leaf_nbf.astype(np.int64))) * 8

    def build_ao_cache(self, nd, ncores=None):
        """Evaluate and keep the basis functions (``nd = 1``) or their values and gradients
        (``nd = 4``) on every leaf; returns the :class:`AOCache` (also stored as ``plan.cache``)."""
        nd = int(nd)
        if nd not in (1, 4):
            raise ValueError('nd must be 1 (values) or 4 (values and gradients).')
        k = (self.leaf_hi - self.leaf_lo).astype(np.int64)
        sizes = nd * k * self.leaf_nbf.astype(np.int64)
        ptr = np.zeros(self.nleaf + 1, dtype=np.int64)
        ptr[1:] = np.cumsum(sizes)
        data = np.empty(int(ptr[-1]), dtype=np.float64)
        sd = self.shell
        if ncores is not None:
            numba.set_num_threads(int(ncores))
        _build_ao_cache_kernel(self.coords, self.leaf_lo, self.leaf_hi, self.leaf_sh_ptr, self.leaf_sh, self.leaf_nbf,
                               sd['sh_coords'], sd['sh_rcut2'], sd['sh_nprim'], sd['sh_expnts'], sd['sh_bf0'],
                               sd['sh_nbf'], sd['sh_L'], sd['bf_lmn'], sd['bf_coef'], nd, ptr, data)
        self.cache = AOCache(self, nd, data, ptr)
        return self.cache

    def bins(self, nthreads, cached):
        """Static schedule: macro blocks per thread, longest processing time first."""
        key = (int(nthreads), bool(cached))
        if key in self._bins:
            return self._bins[key]
        # with the basis functions cached the linear part is only the gather and the elementwise work
        alpha = 12.0 if cached else 60.0
        cost = self._macro_gemm + alpha * self._macro_lin
        order = np.argsort(-cost, kind='stable')
        heap = [(0.0, t) for t in range(nthreads)]
        heapq.heapify(heap)
        bins = [[] for _ in range(nthreads)]
        for ib in order:
            load, t = heapq.heappop(heap)
            bins[t].append(int(ib))
            heapq.heappush(heap, (load + float(cost[ib]), t))
        bin_ptr = np.zeros(nthreads + 1, dtype=np.int64)
        bin_ptr[1:] = np.cumsum([len(b) for b in bins])
        bin_blocks = np.array([ib for b in bins for ib in b], dtype=np.int64) if self.nmacro > 0 else np.zeros(0, np.int64)
        self._bins[key] = (bin_ptr, bin_blocks)
        return self._bins[key]


_PLAN_MEMO = []
_PLAN_MEMO_SIZE = 2


def build_xc_plan(basis, coords, weights, ncores=1, leaf_size=256, leaves_per_macro=16, screen=True,
                  point_cutoff=True):
    """Partition ``coords`` for ``basis`` (see :class:`XCPlan`); ``ncores`` threads are used
    for the screening."""
    numba.set_num_threads(int(ncores))
    return XCPlan(basis, coords, weights, leaf_size=leaf_size, leaves_per_macro=leaves_per_macro,
                  ncores=ncores, screen=screen, point_cutoff=point_cutoff)


def _get_plan(basis, coords, weights, ncores, leaf_size, leaves_per_macro, screen):
    """The memoized plan of these exact arrays (same objects, same shape, same sample of
    values), or a new one."""
    coords = np.asarray(coords)
    weights = np.asarray(weights)
    n = coords.shape[0]
    sample = (float(np.sum(coords[::997])) if n > 0 else 0.0, float(np.sum(weights[::991])) if n > 0 else 0.0)
    key = (id(basis), basis.bfs_nao, id(coords), id(weights), n, sample, int(leaf_size), int(leaves_per_macro), bool(screen))
    for entry in _PLAN_MEMO:
        if entry[0] == key and all(r() is not None for r in entry[1]):
            return entry[2]
    plan = build_xc_plan(basis, coords, weights, ncores=ncores, leaf_size=leaf_size,
                         leaves_per_macro=leaves_per_macro, screen=screen)
    refs = (weakref.ref(basis), weakref.ref(coords), weakref.ref(weights))
    _PLAN_MEMO.append((key, refs, plan))
    del _PLAN_MEMO[:-_PLAN_MEMO_SIZE]
    return plan


# ---------------------------------------------------------------------------------------
# Functionals on the whole grid (two-pass path)
# ---------------------------------------------------------------------------------------
def _functional_full_grid(funcs, xc_family, rho, sigma, tau, ncores):
    """Energy per particle and derivatives of the summed functionals on the whole grid,
    evaluated in chunks on ``ncores`` threads (pylibxc and NumPy release the GIL for the
    heavy parts)."""
    ng = rho.shape[0]
    e = np.zeros(ng)
    vrho = np.zeros(ng)
    vsigma = np.zeros(ng) if xc_family >= 2 else None
    vtau = np.zeros(ng) if xc_family == 4 else None
    nchunks = int(max(1, min(ng // 16384, 8 * max(1, ncores))))
    bounds = np.linspace(0, ng, nchunks + 1).astype(np.int64)

    def work(c):
        i0 = int(bounds[c])
        i1 = int(bounds[c + 1])
        if i1 <= i0:
            return
        r = rho[i0:i1]
        s = sigma[i0:i1] if sigma is not None else None
        tt = tau[i0:i1] if tau is not None else None
        for kind, fn, fam in funcs:
            if kind == 'libxc':
                inp = {'rho': r}
                if fam >= 2:
                    inp['sigma'] = s
                if fam == 4:
                    inp['tau'] = tt
                ret = fn.compute(inp)
                e[i0:i1] += ret['zk'].ravel()
                vrho[i0:i1] += ret['vrho'].ravel()
                if fam >= 2:
                    vsigma[i0:i1] += ret['vsigma'].ravel()
                if fam == 4:
                    vtau[i0:i1] += ret['vtau'].ravel()
            else:
                if fam == 4:
                    ret = XC.func_compute(fn, r, sigma=s, tau=tt, use_gpu=False)
                elif fam == 2:
                    ret = XC.func_compute(fn, r, sigma=s, use_gpu=False)
                else:
                    ret = XC.func_compute(fn, r, use_gpu=False)
                e[i0:i1] += ret[0]
                vrho[i0:i1] += ret[1]
                if fam >= 2:
                    vsigma[i0:i1] += ret[2]
                if fam == 4:
                    vtau[i0:i1] += ret[3]

    if nchunks == 1 or ncores <= 1:
        for c in range(nchunks):
            work(c)
    else:
        with ThreadPoolExecutor(max_workers=int(ncores)) as ex:
            list(ex.map(work, range(nchunks)))
    return e, vrho, vsigma, vtau


def _resolve_native_kernel_functionals(funcs, use_libxc):
    """(fids, coefs) of the one-pass path when every component is a Numba-compiled native
    functional (hybrids expanded into their semilocal components), else None."""
    if use_libxc:
        return None
    fids = []
    coefs = []
    for kind, fid, fam in funcs:
        if kind != 'native':
            return None
        if XC.is_hybrid(int(fid)):
            for coef, cid in XC.get_hybrid_components(int(fid)):
                fids.append(int(cid))
                coefs.append(float(coef))
        else:
            fids.append(int(fid))
            coefs.append(1.0)
    if not all(fid in INKERNEL_FUNCTIONALS for fid in fids):
        return None
    return np.asarray(fids, dtype=np.int64), np.asarray(coefs, dtype=np.float64)


_EMPTY = np.zeros(0, dtype=np.float64)
_EMPTY2 = np.zeros((3, 0), dtype=np.float64)
_EMPTY_I = np.zeros(1, dtype=np.int64)



def _run_kernel(plan, mode, nd, fam, dmat, fids, coefs, cache, bin_ptr, bin_blocks, nthreads, V_store, exc_M, nel_M,
                rho_g=None, grad_g=None, tau_g=None, vrho_g=None, vsigma_g=None, vtau_g=None):
    """Launch :func:`_xc3_kernel` for ``plan`` (see its documentation for ``mode``). BLAS must be
    limited to one thread by the caller."""
    sd = plan.shell
    use_cache = 0
    cache_data = _EMPTY
    cache_ptr = _EMPTY_I
    if cache is not None and cache.plan is plan and cache.nd == nd:
        use_cache = 1
        cache_data = cache.data
        cache_ptr = cache.ptr
    ws, iws = plan.workspace(nthreads, xc_workspace_size(plan.max_k, plan.max_nb, plan.max_nbM),
                             plan.nsh + plan.max_nb)
    _xc3_kernel(mode, nd, fam, plan.coords, plan.weights, dmat,
                sd['sh_coords'], sd['sh_rcut2'], sd['sh_nprim'], sd['sh_expnts'], sd['sh_bf0'], sd['sh_nbf'],
                sd['sh_L'], sd['bf_lmn'], sd['bf_coef'],
                plan.leaf_lo, plan.leaf_hi, plan.leaf_sh_ptr, plan.leaf_sh,
                plan.macro_leaf_lo, plan.macro_leaf_hi, plan.macro_sh_ptr, plan.macro_sh, plan.macro_bf_ptr,
                plan.macro_bf, plan.store_ptr, bin_ptr, bin_blocks, nthreads, fids, coefs,
                cache_data, cache_ptr, use_cache,
                _EMPTY if rho_g is None else rho_g, _EMPTY2 if grad_g is None else grad_g,
                _EMPTY if tau_g is None else tau_g, _EMPTY if vrho_g is None else vrho_g,
                _EMPTY if vsigma_g is None else vsigma_g, _EMPTY if vtau_g is None else vtau_g,
                V_store, exc_M, nel_M, plan.max_k, plan.max_nb, plan.max_nbM, ws, iws)


# ---------------------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------------------
def eval_xc_3(basis, dmat, weights, coords, funcid=[1, 7], use_libxc=False, spin=0, ncores=2, blocksize=5000,
              list_nonzero_indices=None, count_nonzero_indices=None, list_ao_values=None, list_ao_grad_values=None,
              debug=False, print_nelec=True, plan=None, leaf_size=256, leaves_per_macro=16, xc_bf_screen=True,
              force_two_pass=False):
    """
    Exchange-correlation energy and potential matrix for a density matrix, algorithm 3
    (the CPU default; see the module documentation).

    The arguments are those of :func:`pyfock.Integrals.eval_xc_2`, so the two are
    interchangeable. ``blocksize``, ``list_nonzero_indices`` and ``count_nonzero_indices`` are
    accepted but not used: the grid is partitioned and screened by the plan (``plan``, or one
    built for the grid on the first call and reused while the same grid arrays are passed).

    Parameters
    ----------
    basis : Basis
    dmat : (nbf, nbf) ndarray
        Density matrix in the Cartesian AO basis.
    weights, coords : ndarray
        Grid weights and coordinates ``(N, 3)``.
    funcid : int or list of int
        LibXC functional IDs (an exchange-correlation pair, or a single ID such as a global
        hybrid, whose semilocal part is evaluated here).
    use_libxc : bool
        Evaluate the functionals with pylibxc instead of the native implementations.
    ncores : int
        Number of threads.
    list_ao_values : AOCache or None
        Basis functions kept between SCF iterations (``plan.build_ao_cache``). The per-batch
        lists of algorithm 2 are not understood and are ignored (the functions are then
        evaluated on the fly).
    plan : XCPlan, optional
        A plan built with :func:`build_xc_plan` for exactly these ``coords`` and ``weights``.
    leaf_size, leaves_per_macro : int
        Partition parameters used when a plan has to be built.
    xc_bf_screen : bool
        With ``False`` every shell is evaluated on every leaf (only for checking the screening).
    force_two_pass : bool
        Evaluate the functional on the whole grid between two passes even when it could be
        evaluated inside the kernel (only for testing that path).

    Returns
    -------
    efunc : float
        Exchange-correlation energy.
    v : (nbf, nbf) ndarray
        Exchange-correlation potential matrix.
    """
    if spin != 0:
        raise NotImplementedError('eval_xc_3: only spin-unpolarized densities are supported.')
    t_start = timer()
    ncores = max(1, int(ncores))
    nthreads = min(ncores, int(numba.config.NUMBA_NUM_THREADS))
    numba.set_num_threads(nthreads)

    funcs, xc_family = _xc_functionals(funcid, use_libxc)
    native = None if force_two_pass else _resolve_native_kernel_functionals(funcs, use_libxc)
    nd = 1 if xc_family == 1 else 4

    cache = None
    if isinstance(list_ao_values, AOCache):
        cache = list_ao_values
        plan = cache.plan if plan is None else plan
    if plan is None:
        plan = _get_plan(basis, coords, weights, nthreads, leaf_size, leaves_per_macro, xc_bf_screen)
    if cache is None and isinstance(plan.cache, AOCache):
        cache = plan.cache
    if cache is not None and (cache.plan is not plan or cache.nd < nd):
        cache = None
    if cache is not None and cache.nd != nd:
        cache = None   # the kernel reads blocks of nd rows: a cache of another family does not serve
    use_cache = 1 if cache is not None else 0
    if plan.nbf != basis.bfs_nao:
        raise ValueError('The plan was built for a different basis.')
    if plan.ngrids != np.asarray(coords).shape[0]:
        raise ValueError('The plan was built for a different grid.')
    nbf = plan.nbf
    dmat = np.ascontiguousarray(dmat, dtype=np.float64)
    if dmat.shape != (nbf, nbf):
        raise ValueError('dmat has the wrong shape.')

    bin_ptr, bin_blocks = plan.bins(nthreads, bool(use_cache))
    V_store = np.empty(int(plan.store_ptr[-1]), dtype=np.float64)
    exc_M = np.zeros(plan.nmacro)
    nel_M = np.zeros(plan.nmacro)
    t_plan = timer()

    timings = {}
    with threadpool_limits(limits=1, user_api='blas'):
        if native is not None:
            fids, coefs = native
            t0 = timer()
            _run_kernel(plan, 0, nd, xc_family, dmat, fids, coefs, cache, bin_ptr, bin_blocks, nthreads,
                        V_store, exc_M, nel_M)
            timings['kernel'] = timer() - t0
            efunc = float(np.sum(exc_M))
            nelec = float(np.sum(nel_M))
        else:
            ng = plan.ngrids
            rho_g = np.empty(ng)
            grad_g = np.empty((3, ng)) if xc_family >= 2 else _EMPTY2
            tau_g = np.empty(ng) if xc_family == 4 else _EMPTY
            t0 = timer()
            _run_kernel(plan, 1, nd, xc_family, dmat, _EMPTY_I, _EMPTY, cache, bin_ptr, bin_blocks, nthreads,
                        V_store, exc_M, nel_M, rho_g=rho_g, grad_g=grad_g, tau_g=tau_g)
            timings['kernel_rho'] = timer() - t0
            t0 = timer()
            sigma_g = np.einsum('ki,ki->i', grad_g, grad_g) if xc_family >= 2 else None
            e_g, vrho_g, vsigma_g, vtau_g = _functional_full_grid(
                funcs, xc_family, rho_g, sigma_g, tau_g if xc_family == 4 else None, ncores)
            den = plan.weights * rho_g
            efunc = float(np.dot(den, e_g))
            nelec = float(np.sum(den))
            timings['functional'] = timer() - t0
            t0 = timer()
            _run_kernel(plan, 2, nd, xc_family, dmat, _EMPTY_I, _EMPTY, cache, bin_ptr, bin_blocks, nthreads,
                        V_store, exc_M, nel_M, rho_g=rho_g, grad_g=grad_g, tau_g=tau_g, vrho_g=vrho_g,
                        vsigma_g=vsigma_g if vsigma_g is not None else _EMPTY,
                        vtau_g=vtau_g if vtau_g is not None else _EMPTY)
            timings['kernel_v'] = timer() - t0
        t0 = timer()
        nbands = max(1, min(nbf, 8 * nthreads))
        v = _assemble_v(V_store, plan.store_ptr, plan.macro_bf_ptr, plan.macro_bf, nbf, nbands)
        timings['assemble'] = timer() - t0

    numba.set_num_threads(nthreads)
    if print_nelec:
        print('Number of electrons: ', nelec)
    if debug:
        timings['setup'] = t_plan - t_start
        timings['total'] = timer() - t_start
        print('eval_xc_3 timings (s):', {k: round(v_, 4) for k, v_ in timings.items()},
              '| one-pass' if native is not None else '| two-pass', '| AO cache' if use_cache else '| AO on the fly')
    return efunc, v
