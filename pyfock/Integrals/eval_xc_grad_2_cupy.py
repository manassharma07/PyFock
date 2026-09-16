"""GPU (CuPy) exchange-correlation contribution to the nuclear gradient.

Device counterpart of :func:`pyfock.Integrals.eval_xc_grad_2`; see that function for the formulae.
The structure is the same -- loop over grid blocks, evaluate the screened AO values, gradients and
Hessians, contract them with the functional's derivatives -- but a block is processed by a handful of
large CuPy GEMMs instead of by a joblib worker, so the blocks are taken one at a time rather than in
parallel.

Supports LDA, GGA and meta-GGA (tau-dependent) functionals, with the native PyFock functionals or
pylibxc, and the optional grid response. Laplacian-dependent meta-GGAs are not supported (neither is
the SCF).

A note on r2SCAN, which speeds up much less than the rest (~2.5x overall against ~7x for PBE). Its
native implementation has no analytic potential: it finite-differences its own energy expression,
seven evaluations of ~160 elementwise operations per call. That is ~2000 kernel launches whatever
the block size, so on the device the call is launch-bound -- measured at 24 ms for a 20k-point block
and 21 ms for a 120k-point one, against 63 ms and 425 ms on the host. Calling it once per grid block
therefore pays the fixed cost 15 times over. Evaluating it once for the whole grid would need the
two-pass shape of eval_xc_grad_skala_cupy (density pass, one functional call, contraction pass) at
the cost of a second AO evaluation; it was measured at ~0.3 s of caffeine's 0.84 s XC gradient and
judged not worth restructuring the path for. The same finite differencing is also why r2SCAN's
device/host agreement is ~1e-9 rather than the ~1e-12 an analytic functional gives.
"""
import numpy as np

try:
    import cupy as cp
except Exception:                                  # pragma: no cover - CPU-only install
    cp = None
from numba import cuda

from pyfock import XC
from . import bf_val_helpers
from .cuda_stream import gradient_stream

__all__ = ['eval_xc_grad_2_cupy']

# ao_hess components are stored as 0:xx 1:xy 2:xz 3:yy 4:yz 5:zz.
_HESS = ((0, 1, 2), (1, 3, 4), (2, 4, 5))


def _pack_basis_arrays(basis):
    maxnprim = max(basis.bfs_nprim)
    coeffs = np.zeros((basis.bfs_nao, maxnprim))
    expnts = np.zeros((basis.bfs_nao, maxnprim))
    prim_norms = np.zeros((basis.bfs_nao, maxnprim))
    for i in range(basis.bfs_nao):
        for j in range(basis.bfs_nprim[i]):
            coeffs[i, j] = basis.bfs_coeffs[i][j]
            expnts[i, j] = basis.bfs_expnts[i][j]
            prim_norms[i, j] = basis.bfs_prim_norms[i][j]
    return (cp.asarray(np.array(basis.bfs_coords)),
            cp.asarray(np.array(basis.bfs_contr_prim_norms)),
            cp.asarray(np.array(basis.bfs_nprim)),
            cp.asarray(np.array(basis.bfs_lmn)),
            cp.asarray(coeffs), cp.asarray(prim_norms), cp.asarray(expnts))


def _sum_prod(a, b):
    """einsum('mj,mj->m', a, b)."""
    return cp.sum(a * b, axis=1)


def _col_sum_prod(a, b):
    """einsum('mj,mj->j', a, b)."""
    return cp.sum(a * b, axis=0)


def eval_xc_grad_2_cupy(basis, dmat, weights, coords, funcid=(1, 7), use_libxc=False,
                        blocksize=20480, list_nonzero_indices=None, count_nonzero_indices=None,
                        grids=None, grid_response=False, cp_stream=None):
    """GPU counterpart of :func:`eval_xc_grad_2`; same return convention.

    Returns ``dexc_dbf`` of shape ``(3, nbf)`` as a NumPy array, or ``(dexc_dbf, atom_grad)`` when
    ``grid_response=True``.
    """
    if cp is None:
        raise RuntimeError('CuPy is required for eval_xc_grad_2_cupy.')
    if grid_response and grids is None:
        raise ValueError('grid_response=True needs the grids object (pass grids=...).')

    if cp_stream is None:
        cp_stream, nb_stream = gradient_stream()
    else:
        nb_stream = cuda.external_stream(cp_stream.ptr)

    xc_family_dict = {1: 'LDA', 2: 'GGA', 4: 'MGGA'}
    if use_libxc:
        import pylibxc
        funcx = pylibxc.LibXCFunctional(funcid[0], 'unpolarized')
        funcc = pylibxc.LibXCFunctional(funcid[1], 'unpolarized')
        x_family_code = funcx.get_family()
        c_family_code = funcc.get_family()
    else:
        funcx = funcc = None
        x_family_code = XC.get_family(funcid[0])
        c_family_code = XC.get_family(funcid[1])
    is_gga = (xc_family_dict[x_family_code] != 'LDA' or xc_family_dict[c_family_code] != 'LDA')
    is_mgga = (xc_family_dict[x_family_code] == 'MGGA' or xc_family_dict[c_family_code] == 'MGGA')
    need_hess = is_gga or grid_response

    with cp_stream:
        (bfs_coords, bfs_contr_prim_norms, bfs_nprim, bfs_lmn,
         bfs_coeffs, bfs_prim_norms, bfs_expnts) = _pack_basis_arrays(basis)

        coords_d = cp.asarray(coords)
        weights_d = cp.asarray(weights)
        dmat_d = cp.asarray(dmat)
        ngrids = coords_d.shape[0]
        nblocks = ngrids // blocksize

        natm = 0
        atom_idx_d = None
        if grid_response:
            atom_idx_d = cp.asarray(np.ascontiguousarray(grids.atom_idx, dtype=np.int64))
            natm = np.asarray(grids.mol.coordsBohrs).reshape(-1, 3).shape[0]

        dexc_dbf = cp.zeros((3, basis.bfs_nao))
        atom_grad = cp.zeros((max(natm, 1), 3))
        eps = cp.zeros(ngrids) if grid_response else None

        thread_x, thread_y = 8, 32
        for iblock in range(nblocks + 1):
            lo = iblock * blocksize
            hi = min(lo + blocksize, ngrids)
            if hi <= lo:
                continue
            coords_block = coords_d[lo:hi]
            weights_block = weights_d[lo:hi]
            npts = hi - lo

            if list_nonzero_indices is not None:
                nz = cp.asarray(list_nonzero_indices[iblock][0:count_nonzero_indices[iblock]])
                if nz.shape[0] == 0:
                    continue
                dmat_block = cp.ascontiguousarray(dmat_d[cp.ix_(nz, nz)])
            else:
                nz = cp.arange(basis.bfs_nao)
                dmat_block = dmat_d
            nbf_block = nz.shape[0]

            # empty, not zeros: the AO kernels write every element, and at a few tens of MB per
            # array per block the memsets are not free.
            ao = cp.empty((npts, nbf_block))
            ao_grad = cp.empty((3, npts, nbf_block))
            blocks_per_grid = ((nbf_block + thread_x - 1) // thread_x,
                               (npts + thread_y - 1) // thread_y)
            if need_hess:
                ao_hess = cp.empty((6, npts, nbf_block))
                bf_val_helpers.eval_bfs_grad_and_hess_sparse_internal_cuda[
                    blocks_per_grid, (thread_x, thread_y), nb_stream](
                    bfs_coords, bfs_contr_prim_norms, bfs_nprim, bfs_lmn, bfs_coeffs,
                    bfs_prim_norms, bfs_expnts, coords_block, nz, ao, ao_grad, ao_hess)
            else:
                ao_hess = None
                bf_val_helpers.eval_bfs_and_grad_sparse_internal_cuda[
                    blocks_per_grid, (thread_x, thread_y), nb_stream](
                    bfs_coords, bfs_contr_prim_norms, bfs_nprim, bfs_lmn, bfs_coeffs,
                    bfs_prim_norms, bfs_expnts, coords_block, nz, ao, ao_grad)

            Fmj = ao @ dmat_block                      # (chi D)[g, mu]
            rho = _sum_prod(Fmj, ao)

            sigma = tau = Hgrad = None
            rho_grad = None
            if need_hess:
                rho_grad = cp.stack((2.0 * _sum_prod(Fmj, ao_grad[0]),
                                     2.0 * _sum_prod(Fmj, ao_grad[1]),
                                     2.0 * _sum_prod(Fmj, ao_grad[2])))
                if is_gga:
                    sigma = rho_grad[0] ** 2 + rho_grad[1] ** 2 + rho_grad[2] ** 2
            if is_mgga or (grid_response and is_gga):
                Hgrad = [ao_grad[0] @ dmat_block, ao_grad[1] @ dmat_block, ao_grad[2] @ dmat_block]
                tau = 0.5 * (_sum_prod(ao_grad[0], Hgrad[0]) + _sum_prod(ao_grad[1], Hgrad[1])
                             + _sum_prod(ao_grad[2], Hgrad[2]))

            energy_density, vrho, vsigma, vtau = _functional_derivatives(
                funcid, use_libxc, funcx, funcc, x_family_code, c_family_code, xc_family_dict,
                rho, sigma, tau, need_energy=grid_response)

            F = weights_block * vrho
            res = cp.empty((3, nbf_block))
            Fk = None

            if not is_gga:
                FD = F[:, None] * Fmj
                for d in range(3):
                    res[d] = _col_sum_prod(ao_grad[d], FD)
            else:
                Fk = 2.0 * (weights_block * vsigma)[None, :] * rho_grad
                aow = F[:, None] * ao
                for k in range(3):
                    aow += Fk[k][:, None] * ao_grad[k]
                aowD = aow @ dmat_block
                for d in range(3):
                    res[d] = _col_sum_prod(ao_grad[d], aowD)
                    # sum_k Fk * d_k d_d chi_mu, contracted with (chi D)
                    hessF = (Fk[0][:, None] * ao_hess[_HESS[d][0]]
                             + Fk[1][:, None] * ao_hess[_HESS[d][1]]
                             + Fk[2][:, None] * ao_hess[_HESS[d][2]])
                    res[d] += _col_sum_prod(hessF, Fmj)

                if is_mgga:
                    Gtau = 0.5 * weights_block * vtau
                    GH = [Gtau[:, None] * Hgrad[k] for k in range(3)]
                    for d in range(3):
                        res[d] += (_col_sum_prod(ao_hess[_HESS[d][0]], GH[0])
                                   + _col_sum_prod(ao_hess[_HESS[d][1]], GH[1])
                                   + _col_sum_prod(ao_hess[_HESS[d][2]], GH[2]))

            if list_nonzero_indices is not None:
                # The screened indices of a block are unique, so a fancy-indexed += is well defined.
                dexc_dbf[:, nz] += res
            else:
                dexc_dbf += res

            if grid_response:
                # Fk is already w dE/d(grad rho); it is None exactly when the functional is an LDA.
                c_tau = (weights_block * vtau) if is_mgga else None
                atom_grad += _grid_translation_term(F, Fk, c_tau, ao_grad, ao_hess,
                                                    Fmj, Hgrad, rho_grad,
                                                    atom_idx_d[lo:hi], natm)
                eps[lo:hi] = rho * energy_density

        dexc_dbf_host = cp.asnumpy(dexc_dbf)
        atom_grad_host = cp.asnumpy(atom_grad)
        eps_host = cp.asnumpy(eps) if grid_response else None
    cp_stream.synchronize()

    if not grid_response:
        return dexc_dbf_host

    from .xc_grid_response import weight_response_term
    return dexc_dbf_host, atom_grad_host + weight_response_term(grids, eps_host)


def _functional_derivatives(funcid, use_libxc, funcx, funcc, x_family_code, c_family_code,
                            xc_family_dict, rho, sigma, tau, need_energy):
    """vrho / vsigma / vtau (and the energy density when the grid response needs it)."""
    if use_libxc:
        # pylibxc has no device entry point, so the per-point quantities make a round trip.
        rho_h = cp.asnumpy(rho)
        sigma_h = None if sigma is None else cp.asnumpy(sigma)
        tau_h = None if tau is None else cp.asnumpy(tau)
        out = []
        for func, family in ((funcx, x_family_code), (funcc, c_family_code)):
            inp = {'rho': rho_h}
            if xc_family_dict[family] != 'LDA':
                inp['sigma'] = sigma_h
            if xc_family_dict[family] == 'MGGA':
                inp['tau'] = tau_h
            out.append(func.compute(inp))
        retx, retc = out
        energy_density = cp.asarray((retx['zk'] + retc['zk']).ravel()) if need_energy else None
        vrho = cp.asarray((retx['vrho'] + retc['vrho'])[:, 0])
        vsigma = 0.0
        if xc_family_dict[x_family_code] != 'LDA':
            vsigma = vsigma + cp.asarray(retx['vsigma'][:, 0])
        if xc_family_dict[c_family_code] != 'LDA':
            vsigma = vsigma + cp.asarray(retc['vsigma'][:, 0])
        vtau = 0.0
        if xc_family_dict[x_family_code] == 'MGGA':
            vtau = vtau + cp.asarray(retx['vtau'][:, 0])
        if xc_family_dict[c_family_code] == 'MGGA':
            vtau = vtau + cp.asarray(retc['vtau'][:, 0])
        return energy_density, vrho, vsigma, vtau

    retx = XC.func_compute(funcid[0], rho, sigma=sigma, tau=tau, use_gpu=True)
    retc = XC.func_compute(funcid[1], rho, sigma=sigma, tau=tau, use_gpu=True)
    energy_density = (retx[0] + retc[0]) if need_energy else None
    vrho = retx[1] + retc[1]
    vsigma = 0.0
    if xc_family_dict[x_family_code] != 'LDA':
        vsigma = vsigma + retx[2]
    if xc_family_dict[c_family_code] != 'LDA':
        vsigma = vsigma + retc[2]
    vtau = 0.0
    if xc_family_dict[x_family_code] == 'MGGA':
        vtau = vtau + retx[3]
    if xc_family_dict[c_family_code] == 'MGGA':
        vtau = vtau + retc[3]
    return energy_density, vrho, vsigma, vtau


def _grid_translation_term(c_rho, c_grad, c_tau, ao_grad, ao_hess, Fmj, Hgrad, rho_grad,
                           atom_idx_block, natm):
    """Device version of :func:`pyfock.Integrals.xc_grid_response.grid_translation_term`."""
    translation = cp.zeros((natm, 3))
    for b in range(3):
        term = c_rho * rho_grad[b]
        if c_grad is not None:
            for a in range(3):
                # d(grad_a rho)/d b = 2 [ (d_a d_b chi) . (chi D) + (d_a chi) . (d_b chi D) ]
                hessian_ab = 2.0 * (_sum_prod(ao_hess[_HESS[a][b]], Fmj)
                                    + _sum_prod(ao_grad[a], Hgrad[b]))
                term = term + c_grad[a] * hessian_ab
        if c_tau is not None:
            # d(tau)/d b = sum_k (d_b d_k chi) . (d_k chi D)
            grad_tau = _sum_prod(ao_hess[_HESS[b][0]], Hgrad[0])
            grad_tau = grad_tau + _sum_prod(ao_hess[_HESS[b][1]], Hgrad[1])
            grad_tau = grad_tau + _sum_prod(ao_hess[_HESS[b][2]], Hgrad[2])
            term = term + c_tau * grad_tau
        translation[:, b] = cp.bincount(atom_idx_block, weights=term, minlength=natm)
    return translation
