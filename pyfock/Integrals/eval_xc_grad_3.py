"""
Exchange-correlation contribution to the nuclear gradient on the CPU: algorithm 3.

The same leaves, shell lists, macro blocks and static schedule as :func:`pyfock.Integrals.eval_xc_3`
(an :class:`~pyfock.Integrals.eval_xc_3.XCPlan`), with the basis functions evaluated together
with their first and, for GGA/meta-GGA functionals or the grid response, second derivatives.
Every macro block writes its part of the per-function gradient (and of the grid-translation
term) into its own slot, which are summed afterwards in a fixed order, so the result does not
depend on the number of threads.

For the fixed-grid approximation and the grid-response terms see
:func:`pyfock.Integrals.eval_xc_grad_2`, whose return values and conventions this function
shares.
"""
from timeit import default_timer as timer

import numpy as np
import numba
from numba import njit, prange
from threadpoolctl import threadpool_limits

from .eval_xc_2 import _xc_functionals
from .eval_xc_3 import (_get_plan, _run_kernel, _functional_full_grid, _resolve_native_kernel_functionals,
                        _EMPTY, _EMPTY2, _EMPTY_I)
from .eval_xc_3_helpers import _eval_ao_block, _xc_native_eval

__all__ = ['eval_xc_grad_3']

_NB = dict(cache=True, nogil=True, fastmath=True, error_model='numpy')

# ao second derivatives are stored as 0:xx 1:xy 2:xz 3:yy 4:yz 5:zz (after the 4 values/gradients)
_HESS_INDEX = np.array([[4, 5, 6], [5, 7, 8], [6, 8, 9]], dtype=np.int64)


@njit(parallel=True, **_NB)
def _xc3_grad_kernel(mode, fam, grid_response, coords, weights, dmat,
                     sh_coords, sh_rcut2, sh_nprim, sh_expnts, sh_bf0, sh_nbf, sh_L, bf_lmn, bf_coef,
                     leaf_lo, leaf_hi, leaf_sh_ptr, leaf_sh,
                     macro_leaf_lo, macro_leaf_hi, macro_sh_ptr, macro_sh, macro_bf_ptr, macro_bf,
                     bin_ptr, bin_blocks, nthreads, fids, coefs, hess_index,
                     grad_g, vrho_g, vsigma_g, vtau_g,
                     point_atom_local, macro_atom_ptr,
                     dexc_store, atom_store, eps_g, max_k, max_nb, max_nbM, ws, iws):
    """Thread ``t`` processes the macro blocks ``bin_blocks[bin_ptr[t]:bin_ptr[t+1]]``.

    mode 0: the functional (``fids``/``coefs``) is evaluated here; mode 2: ``vrho_g``,
    ``vsigma_g``, ``vtau_g`` (and the density gradient ``grad_g``) are given on the grid.
    ``dexc_store[3*macro_bf_ptr[ib] + d*nbM + i]`` receives ``dexc_dbf[d, macro_bf[i]]`` of the
    block; with the grid response ``atom_store[(macro_atom_ptr[ib] + a)*3 + b]`` receives the
    translation term of the block's local atom ``a`` and ``eps_g`` the energy density per volume.
    """
    nsh = sh_coords.shape[0]
    nd = 10 if (fam >= 2 or grid_response == 1) else 4
    need_H = (fam == 4) or (grid_response == 1 and fam >= 2)
    kn = max_k * max_nb
    for t in prange(nthreads):
        w = ws[t]
        o = 0
        ao_buf = w[o:o + 10 * kn]
        o += 10 * kn
        F_buf = w[o:o + kn]
        o += kn
        H_buf = w[o:o + 3 * kn]
        o += 3 * kn
        W_buf = w[o:o + kn]
        o += kn
        WD_buf = w[o:o + kn]
        o += kn
        Ds_buf = w[o:o + max_nb * max_nb]
        o += max_nb * max_nb
        D_M = w[o:o + max_nbM * max_nbM].reshape(max_nbM, max_nbM)
        o += max_nbM * max_nbM
        res = w[o:o + 3 * max_nb].reshape(3, max_nb)
        o += 3 * max_nb
        vec = w[o:o + 15 * max_k].reshape(15, max_k)
        rho = vec[0]
        sigma = vec[1]
        tau = vec[2]
        gx = vec[3]
        gy = vec[4]
        gz = vec[5]
        e = vec[6]
        vrho = vec[7]
        vsigma = vec[8]
        vtau = vec[9]
        cr = vec[10]
        cgx = vec[11]
        cgy = vec[12]
        cgz = vec[13]
        ct = vec[14]
        pos_in_macro = iws[t, :nsh]
        ridx = iws[t, nsh:nsh + max_nb]
        for jb in range(bin_ptr[t], bin_ptr[t + 1]):
            ib = bin_blocks[jb]
            p0 = macro_bf_ptr[ib]
            nbM = macro_bf_ptr[ib + 1] - p0
            off = 0
            for q in range(macro_sh_ptr[ib], macro_sh_ptr[ib + 1]):
                s = macro_sh[q]
                pos_in_macro[s] = off
                off += sh_nbf[s]
            for i in range(nbM):
                gi = macro_bf[p0 + i]
                for j in range(nbM):
                    D_M[i, j] = dmat[gi, macro_bf[p0 + j]]
            dbase = 3 * p0
            for i in range(3 * nbM):
                dexc_store[dbase + i] = 0.0
            abase = 3 * macro_atom_ptr[ib]
            if grid_response == 1:
                for i in range(3 * (macro_atom_ptr[ib + 1] - macro_atom_ptr[ib])):
                    atom_store[abase + i] = 0.0
            for il in range(macro_leaf_lo[ib], macro_leaf_hi[ib]):
                lo = leaf_lo[il]
                k = leaf_hi[il] - lo
                q0 = leaf_sh_ptr[il]
                nshb = leaf_sh_ptr[il + 1] - q0
                n = 0
                for q in range(q0, q0 + nshb):
                    s = leaf_sh[q]
                    pp = pos_in_macro[s]
                    for c in range(sh_nbf[s]):
                        ridx[n] = pp + c
                        n += 1
                if n == 0 or k == 0:
                    if grid_response == 1 and mode == 0:
                        for kk in range(k):
                            eps_g[lo + kk] = 0.0
                    continue
                ao = ao_buf[:nd * k * n].reshape(nd, k, n)
                _eval_ao_block(coords, lo, k, leaf_sh, q0, nshb, sh_coords, sh_rcut2, sh_nprim, sh_expnts,
                               sh_bf0, sh_nbf, sh_L, bf_lmn, bf_coef, nd, ao)
                Ds = Ds_buf[:n * n].reshape(n, n)
                for i in range(n):
                    ri = ridx[i]
                    for j in range(n):
                        Ds[i, j] = D_M[ri, ridx[j]]
                # ---- density, its gradient, tau
                F = F_buf[:k * n].reshape(k, n)
                np.dot(ao[0], Ds, F)
                for kk in range(k):
                    s0 = 0.0
                    for j in range(n):
                        s0 += F[kk, j] * ao[0, kk, j]
                    rho[kk] = s0
                if fam >= 2 or grid_response == 1:
                    for kk in range(k):
                        sx = 0.0
                        sy = 0.0
                        sz = 0.0
                        for j in range(n):
                            f = F[kk, j]
                            sx += f * ao[1, kk, j]
                            sy += f * ao[2, kk, j]
                            sz += f * ao[3, kk, j]
                        gx[kk] = 2.0 * sx
                        gy[kk] = 2.0 * sy
                        gz[kk] = 2.0 * sz
                        sigma[kk] = gx[kk] * gx[kk] + gy[kk] * gy[kk] + gz[kk] * gz[kk]
                H = H_buf[:3 * k * n].reshape(3 * k, n)
                if need_H:
                    A3 = ao[1:4].reshape(3 * k, n)
                    np.dot(A3, Ds, H)
                    if fam == 4:
                        for kk in range(k):
                            s0 = 0.0
                            for c in range(3):
                                for j in range(n):
                                    s0 += ao[1 + c, kk, j] * H[c * k + kk, j]
                            tau[kk] = 0.5 * s0
                # ---- functional
                if mode == 0:
                    _xc_native_eval(fids, coefs, rho[:k], sigma[:k], tau[:k], e[:k], vrho[:k], vsigma[:k], vtau[:k])
                    if grid_response == 1:
                        for kk in range(k):
                            eps_g[lo + kk] = rho[kk] * e[kk]
                else:
                    for kk in range(k):
                        vrho[kk] = vrho_g[lo + kk]
                        if fam >= 2:
                            vsigma[kk] = vsigma_g[lo + kk]
                            gx[kk] = grad_g[0, lo + kk]
                            gy[kk] = grad_g[1, lo + kk]
                            gz[kk] = grad_g[2, lo + kk]
                        if fam == 4:
                            vtau[kk] = vtau_g[lo + kk]
                # ---- cotangents
                for kk in range(k):
                    w = weights[lo + kk]
                    cr[kk] = w * vrho[kk]
                    if fam >= 2:
                        fs = 2.0 * w * vsigma[kk]
                        cgx[kk] = fs * gx[kk]
                        cgy[kk] = fs * gy[kk]
                        cgz[kk] = fs * gz[kk]
                    if fam == 4:
                        ct[kk] = w * vtau[kk]
                # ---- Pulay term: res[d, mu] = sum_g d_d chi_mu (aow D)[g, mu] + ...
                for d in range(3):
                    for j in range(n):
                        res[d, j] = 0.0
                if fam == 1:
                    for kk in range(k):
                        c0 = cr[kk]
                        for j in range(n):
                            fd = c0 * F[kk, j]
                            res[0, j] += ao[1, kk, j] * fd
                            res[1, j] += ao[2, kk, j] * fd
                            res[2, j] += ao[3, kk, j] * fd
                else:
                    W = W_buf[:k * n].reshape(k, n)
                    for kk in range(k):
                        c0 = cr[kk]
                        cx = cgx[kk]
                        cy = cgy[kk]
                        cz = cgz[kk]
                        for j in range(n):
                            W[kk, j] = c0 * ao[0, kk, j] + cx * ao[1, kk, j] + cy * ao[2, kk, j] + cz * ao[3, kk, j]
                    WD = WD_buf[:k * n].reshape(k, n)
                    np.dot(W, Ds, WD)
                    for kk in range(k):
                        cx = cgx[kk]
                        cy = cgy[kk]
                        cz = cgz[kk]
                        for j in range(n):
                            wd = WD[kk, j]
                            f = F[kk, j]
                            res[0, j] += ao[1, kk, j] * wd + (cx * ao[4, kk, j] + cy * ao[5, kk, j] + cz * ao[6, kk, j]) * f
                            res[1, j] += ao[2, kk, j] * wd + (cx * ao[5, kk, j] + cy * ao[7, kk, j] + cz * ao[8, kk, j]) * f
                            res[2, j] += ao[3, kk, j] * wd + (cx * ao[6, kk, j] + cy * ao[8, kk, j] + cz * ao[9, kk, j]) * f
                    if fam == 4:
                        for kk in range(k):
                            gt = 0.5 * ct[kk]
                            for j in range(n):
                                hx = gt * H[kk, j]
                                hy = gt * H[k + kk, j]
                                hz = gt * H[2 * k + kk, j]
                                res[0, j] += ao[4, kk, j] * hx + ao[5, kk, j] * hy + ao[6, kk, j] * hz
                                res[1, j] += ao[5, kk, j] * hx + ao[7, kk, j] * hy + ao[8, kk, j] * hz
                                res[2, j] += ao[6, kk, j] * hx + ao[8, kk, j] * hy + ao[9, kk, j] * hz
                for d in range(3):
                    for j in range(n):
                        dexc_store[dbase + d * nbM + ridx[j]] += res[d, j]
                # ---- grid response: the points of an atom move with it
                if grid_response == 1:
                    for kk in range(k):
                        slot = abase + 3 * point_atom_local[lo + kk]
                        for b in range(3):
                            if b == 0:
                                term = cr[kk] * gx[kk]
                            elif b == 1:
                                term = cr[kk] * gy[kk]
                            else:
                                term = cr[kk] * gz[kk]
                            if fam >= 2:
                                for a in range(3):
                                    hab = hess_index[a, b]
                                    s0 = 0.0
                                    for j in range(n):
                                        s0 += ao[hab, kk, j] * F[kk, j] + ao[1 + a, kk, j] * H[b * k + kk, j]
                                    if a == 0:
                                        ca = cgx[kk]
                                    elif a == 1:
                                        ca = cgy[kk]
                                    else:
                                        ca = cgz[kk]
                                    term += ca * 2.0 * s0
                            if fam == 4:
                                s0 = 0.0
                                for c in range(3):
                                    hbc = hess_index[b, c]
                                    for j in range(n):
                                        s0 += ao[hbc, kk, j] * H[c * k + kk, j]
                                term += ct[kk] * s0
                            atom_store[slot + b] += term


@njit(**_NB)
def _reduce_dexc(dexc_store, macro_bf_ptr, macro_bf, nbf):
    nmacro = macro_bf_ptr.shape[0] - 1
    out = np.zeros((3, nbf))
    for ib in range(nmacro):
        p0 = macro_bf_ptr[ib]
        nbM = macro_bf_ptr[ib + 1] - p0
        base = 3 * p0
        for d in range(3):
            for i in range(nbM):
                out[d, macro_bf[p0 + i]] += dexc_store[base + d * nbM + i]
    return out


@njit(**_NB)
def _reduce_atoms(atom_store, macro_atom_ptr, macro_atoms, natm):
    nmacro = macro_atom_ptr.shape[0] - 1
    out = np.zeros((natm, 3))
    for ib in range(nmacro):
        for a in range(macro_atom_ptr[ib], macro_atom_ptr[ib + 1]):
            atom = macro_atoms[a]
            for b in range(3):
                out[atom, b] += atom_store[3 * a + b]
    return out


def grad_workspace_size(max_k, max_nb, max_nbM):
    """Doubles per thread the gradient kernel needs."""
    return 16 * max_k * max_nb + max_nb * max_nb + max_nbM * max_nbM + 3 * max_nb + 15 * max_k


def _macro_atoms(plan, atom_idx):
    """Distinct atoms of every macro block (CSR) and, per point, the local index of its atom."""
    atom_idx_plan = np.ascontiguousarray(np.asarray(atom_idx, dtype=np.int64).reshape(-1)[plan.perm])
    ptr = np.zeros(plan.nmacro + 1, dtype=np.int64)
    atoms = []
    local = np.empty(plan.ngrids, dtype=np.int64)
    for ib in range(plan.nmacro):
        lo = plan.leaf_lo[plan.macro_leaf_lo[ib]]
        hi = plan.leaf_hi[plan.macro_leaf_hi[ib] - 1]
        uniq, inv = np.unique(atom_idx_plan[lo:hi], return_inverse=True)
        atoms.append(uniq)
        local[lo:hi] = inv
        ptr[ib + 1] = ptr[ib] + uniq.shape[0]
    macro_atoms = np.concatenate(atoms).astype(np.int64) if atoms else np.zeros(0, np.int64)
    return ptr, macro_atoms, local


def eval_xc_grad_3(basis, dmat, weights, coords, funcid=[1, 7], use_libxc=False, ncores=2, blocksize=5000,
                   list_nonzero_indices=None, count_nonzero_indices=None, debug=False, grids=None,
                   grid_response=False, plan=None, leaf_size=256, leaves_per_macro=16, xc_bf_screen=True,
                   force_two_pass=False):
    """
    Exchange-correlation contribution to the nuclear gradient, algorithm 3 (same leaves, screening
    and threading as :func:`pyfock.Integrals.eval_xc_3`).

    The arguments and return values are those of :func:`pyfock.Integrals.eval_xc_grad_2`:
    ``dexc_dbf`` of shape ``(3, nbf)`` with ``dExc/dR_{A,d} = -2 sum_{mu in A} dexc_dbf[d, mu]`` on a
    fixed grid, and with ``grid_response=True`` (which needs ``grids``) the tuple ``(dexc_dbf,
    atom_grad)`` where ``atom_grad`` (``natoms, 3``) is the grid-response term resolved per atom.
    ``blocksize``, ``list_nonzero_indices`` and ``count_nonzero_indices`` are accepted for
    compatibility and not used; ``plan`` may be the plan of the SCF (built for the same grid), and
    ``force_two_pass`` evaluates the functional on the whole grid between two passes (testing).
    """
    if grid_response and grids is None:
        raise ValueError('grid_response=True needs the grids object (pass grids=...).')
    t_start = timer()
    ncores = max(1, int(ncores))
    nthreads = min(ncores, int(numba.config.NUMBA_NUM_THREADS))
    numba.set_num_threads(nthreads)

    funcs, xc_family = _xc_functionals(funcid, use_libxc)
    native = None if force_two_pass else _resolve_native_kernel_functionals(funcs, use_libxc)
    if plan is None:
        plan = _get_plan(basis, coords, weights, nthreads, leaf_size, leaves_per_macro, xc_bf_screen)
    if plan.nbf != basis.bfs_nao or plan.ngrids != np.asarray(coords).shape[0]:
        raise ValueError('The plan was built for a different basis or grid.')
    sd = plan.shell
    nbf = plan.nbf
    dmat = np.ascontiguousarray(dmat, dtype=np.float64)
    # the derivative work per block scales like the energy's with more linear work (second derivatives)
    bin_ptr, bin_blocks = plan.bins(nthreads, False)

    natm = 0
    macro_atom_ptr = np.zeros(plan.nmacro + 1, dtype=np.int64)
    macro_atoms = np.zeros(0, dtype=np.int64)
    point_atom_local = np.zeros(0, dtype=np.int64)
    if grid_response:
        natm = np.asarray(grids.mol.coordsBohrs).reshape(-1, 3).shape[0]
        macro_atom_ptr, macro_atoms, point_atom_local = _macro_atoms(plan, grids.atom_idx)
    ws, iws = plan.workspace(nthreads, grad_workspace_size(plan.max_k, plan.max_nb, plan.max_nbM), plan.nsh + plan.max_nb)
    dexc_store = np.zeros(3 * int(plan.macro_bf_ptr[-1]))
    atom_store = np.zeros(3 * int(macro_atom_ptr[-1]))
    eps_g = np.zeros(plan.ngrids) if grid_response else _EMPTY

    timings = {}
    with threadpool_limits(limits=1, user_api='blas'):
        if native is not None:
            fids, coefs = native
            t0 = timer()
            _xc3_grad_kernel(0, xc_family, 1 if grid_response else 0, plan.coords, plan.weights, dmat,
                             sd['sh_coords'], sd['sh_rcut2'], sd['sh_nprim'], sd['sh_expnts'], sd['sh_bf0'],
                             sd['sh_nbf'], sd['sh_L'], sd['bf_lmn'], sd['bf_coef'],
                             plan.leaf_lo, plan.leaf_hi, plan.leaf_sh_ptr, plan.leaf_sh,
                             plan.macro_leaf_lo, plan.macro_leaf_hi, plan.macro_sh_ptr, plan.macro_sh,
                             plan.macro_bf_ptr, plan.macro_bf, bin_ptr, bin_blocks, nthreads, fids, coefs, _HESS_INDEX,
                             _EMPTY2, _EMPTY, _EMPTY, _EMPTY, point_atom_local, macro_atom_ptr,
                             dexc_store, atom_store, eps_g, plan.max_k, plan.max_nb, plan.max_nbM, ws, iws)
            timings['kernel'] = timer() - t0
        else:
            # two passes: the density from the energy kernel, the functional on the whole grid
            ng = plan.ngrids
            nd = 1 if xc_family == 1 else 4
            rho_g = np.empty(ng)
            grad_g = np.empty((3, ng)) if xc_family >= 2 else _EMPTY2
            tau_g = np.empty(ng) if xc_family == 4 else _EMPTY
            V_store = np.empty(int(plan.store_ptr[-1]))
            exc_M = np.zeros(plan.nmacro)
            nel_M = np.zeros(plan.nmacro)
            t0 = timer()
            _run_kernel(plan, 1, nd, xc_family, dmat, _EMPTY_I, _EMPTY, None, bin_ptr, bin_blocks, nthreads,
                        V_store, exc_M, nel_M, rho_g=rho_g, grad_g=grad_g, tau_g=tau_g)
            del V_store
            sigma_g = np.einsum('ki,ki->i', grad_g, grad_g) if xc_family >= 2 else None
            e_g, vrho_g, vsigma_g, vtau_g = _functional_full_grid(
                funcs, xc_family, rho_g, sigma_g, tau_g if xc_family == 4 else None, ncores)
            if grid_response:
                eps_g = rho_g * e_g
            timings['density_and_functional'] = timer() - t0
            t0 = timer()
            _xc3_grad_kernel(2, xc_family, 1 if grid_response else 0, plan.coords, plan.weights, dmat,
                             sd['sh_coords'], sd['sh_rcut2'], sd['sh_nprim'], sd['sh_expnts'], sd['sh_bf0'],
                             sd['sh_nbf'], sd['sh_L'], sd['bf_lmn'], sd['bf_coef'],
                             plan.leaf_lo, plan.leaf_hi, plan.leaf_sh_ptr, plan.leaf_sh,
                             plan.macro_leaf_lo, plan.macro_leaf_hi, plan.macro_sh_ptr, plan.macro_sh,
                             plan.macro_bf_ptr, plan.macro_bf, bin_ptr, bin_blocks, nthreads, _EMPTY_I, _EMPTY, _HESS_INDEX,
                             grad_g, vrho_g, vsigma_g if vsigma_g is not None else _EMPTY,
                             vtau_g if vtau_g is not None else _EMPTY, point_atom_local, macro_atom_ptr,
                             dexc_store, atom_store, eps_g, plan.max_k, plan.max_nb, plan.max_nbM, ws, iws)
            timings['kernel'] = timer() - t0

    dexc_dbf = _reduce_dexc(dexc_store, plan.macro_bf_ptr, plan.macro_bf, nbf)
    numba.set_num_threads(nthreads)
    if not grid_response:
        if debug:
            timings['total'] = timer() - t_start
            print('eval_xc_grad_3 timings (s):', {k: round(v, 4) for k, v in timings.items()})
        return dexc_dbf

    from .xc_grid_response import weight_response_term
    atom_grad = _reduce_atoms(atom_store, macro_atom_ptr, macro_atoms, natm)
    # the energy density per volume back in the order of the grid object
    eps = np.empty(plan.ngrids)
    eps[plan.perm] = eps_g
    t0 = timer()
    atom_grad = atom_grad + weight_response_term(grids, eps)
    timings['weight_response'] = timer() - t0
    if debug:
        timings['total'] = timer() - t_start
        print('eval_xc_grad_3 timings (s):', {k: round(v, 4) for k, v in timings.items()})
    return dexc_dbf, atom_grad
