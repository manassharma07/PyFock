import numpy as np
import random

import numba
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits

from pyfock import XC
from pyfock import Integrals


def eval_xc_grad_2(basis, dmat, weights, coords, funcid=[1, 7], use_libxc=False,
                   ncores=2, blocksize=5000, list_nonzero_indices=None,
                   count_nonzero_indices=None, debug=False):
    """
    Evaluate the exchange-correlation contribution to the nuclear gradient
    using algorithm 2 (block-parallel CPU algorithm, same structure as
    :func:`eval_xc_2`).

    For a fixed grid (no grid-weight response, same approximation as PySCF's
    default) the XC gradient is

        dExc/dR_{A,d} = -2 * sum_{mu in A} dexc_dbf[d, mu]

    where this function returns ``dexc_dbf`` with

        dexc_dbf[d, mu] = sum_g d_d(chi_mu)(g) * aow_D[g, mu]
                          + sum_g (sum_k Fk(g) * d_k d_d chi_mu(g)) * (chi D)[g, mu]   (GGA)
                          + sum_g G_tau(g) * sum_k (d_d d_k chi_mu)(g) * Hk[g, mu]     (MGGA)

        aow[g, nu] = F(g) chi_nu(g) + sum_k Fk(g) d_k chi_nu(g)
        aow_D = aow @ D ;  F = w * vrho ;  Fk = 2 * w * vsigma * (grad rho)_k
        G_tau = 0.5 * w * vtau ;  Hk[g, mu] = (d_k chi @ D)[g, mu]

    This supports LDA, GGA and meta-GGA (tau-dependent) functionals, using
    either the native PyFock functionals or pylibxc. Laplacian-dependent
    meta-GGAs are not supported (the SCF does not use the Laplacian either).

    Parameters
    ----------
    basis : Basis
        Basis set object.
    dmat : ndarray (nbf, nbf)
        Density matrix in the (Cartesian) AO basis.
    weights, coords : ndarray
        Grid weights and coordinates.
    funcid : list of int
        XC functional ids (LibXC convention).
    use_libxc : bool
        Whether to use pylibxc instead of the native PyFock functionals.
    ncores : int
        Number of threads used by joblib.
    blocksize : int
        Number of grid points per block.
    list_nonzero_indices, count_nonzero_indices : optional
        Per-block screening data of significantly contributing basis
        functions (same as used by :func:`eval_xc_2`).

    Returns
    -------
    dexc_dbf : ndarray (3, nbf)
        Per-basis-function XC gradient contributions (see above). The caller
        maps them onto atoms via ``basis.bfs_atoms``.
    """
    ngrids = coords.shape[0]
    nblocks = ngrids // blocksize

    # Pack basis data for Numba kernels
    bfs_coords = np.array([basis.bfs_coords])
    bfs_contr_prim_norms = np.array([basis.bfs_contr_prim_norms])
    bfs_lmn = np.array([basis.bfs_lmn])
    bfs_nprim = np.array([basis.bfs_nprim])
    maxnprim = max(basis.bfs_nprim)
    bfs_coeffs = np.zeros([basis.bfs_nao, maxnprim])
    bfs_expnts = np.zeros([basis.bfs_nao, maxnprim])
    bfs_prim_norms = np.zeros([basis.bfs_nao, maxnprim])
    for i in range(basis.bfs_nao):
        for j in range(basis.bfs_nprim[i]):
            bfs_coeffs[i, j] = basis.bfs_coeffs[i][j]
            bfs_expnts[i, j] = basis.bfs_expnts[i][j]
            bfs_prim_norms[i, j] = basis.bfs_prim_norms[i][j]
    bfs_data_as_np_arrays = [bfs_coords[0], bfs_contr_prim_norms[0], bfs_nprim[0], bfs_lmn[0], bfs_coeffs, bfs_prim_norms, bfs_expnts]

    xc_family_dict = {1: 'LDA', 2: 'GGA', 4: 'MGGA'}
    if use_libxc:
        import pylibxc
        funcx = pylibxc.LibXCFunctional(funcid[0], "unpolarized")
        funcc = pylibxc.LibXCFunctional(funcid[1], "unpolarized")
        x_family_code = funcx.get_family()
        c_family_code = funcc.get_family()
    else:
        x_family_code = XC.get_family(funcid[0])
        c_family_code = XC.get_family(funcid[1])
        funcx = None
        funcc = None

    numba.set_num_threads(1)

    block_indices = list(range(nblocks + 1))
    random.shuffle(block_indices)

    if 2 * ncores > nblocks:
        batch_size = 'auto'
    else:
        batch_size = nblocks // (ncores * 2)

    if list_nonzero_indices is not None:
        output = Parallel(n_jobs=ncores, backend='threading', require='sharedmem', batch_size=batch_size)(
            delayed(block_xc_grad_func)(
                weights[iblock * blocksize: min(iblock * blocksize + blocksize, ngrids)],
                coords[iblock * blocksize: min(iblock * blocksize + blocksize, ngrids)],
                dmat[np.ix_(list_nonzero_indices[iblock][0:count_nonzero_indices[iblock]], list_nonzero_indices[iblock][0:count_nonzero_indices[iblock]])],
                funcid, use_libxc, bfs_data_as_np_arrays,
                list_nonzero_indices[iblock][0:count_nonzero_indices[iblock]],
                funcx=funcx, funcc=funcc,
                x_family_code=x_family_code, c_family_code=c_family_code,
                xc_family_dict=xc_family_dict)
            for iblock in block_indices)
    else:
        full_indices = np.arange(basis.bfs_nao)
        output = Parallel(n_jobs=ncores, backend='threading', require='sharedmem', batch_size=batch_size)(
            delayed(block_xc_grad_func)(
                weights[iblock * blocksize: min(iblock * blocksize + blocksize, ngrids)],
                coords[iblock * blocksize: min(iblock * blocksize + blocksize, ngrids)],
                dmat, funcid, use_libxc, bfs_data_as_np_arrays, full_indices,
                funcx=funcx, funcc=funcc,
                x_family_code=x_family_code, c_family_code=c_family_code,
                xc_family_dict=xc_family_dict)
            for iblock in block_indices)

    dexc_dbf = np.zeros((3, basis.bfs_nao))
    indx_block_output = 0
    for iblock in block_indices:
        if list_nonzero_indices is not None:
            non_zero_indices = list_nonzero_indices[iblock][0:count_nonzero_indices[iblock]]
            dexc_dbf[:, non_zero_indices] += output[indx_block_output]
        else:
            dexc_dbf += output[indx_block_output]
        indx_block_output += 1

    numba.set_num_threads(ncores)

    output = 0

    return dexc_dbf


@threadpool_limits.wrap(limits=1, user_api='blas')
def block_xc_grad_func(weights_block, coords_block, dmat, funcid, use_libxc,
                       bfs_data_as_np_arrays, non_zero_indices,
                       funcx=None, funcc=None, x_family_code=None,
                       c_family_code=None, xc_family_dict=None):
    numba.set_num_threads(1)

    bfs_coords = bfs_data_as_np_arrays[0]
    bfs_contr_prim_norms = bfs_data_as_np_arrays[1]
    bfs_nprim = bfs_data_as_np_arrays[2]
    bfs_lmn = bfs_data_as_np_arrays[3]
    bfs_coeffs = bfs_data_as_np_arrays[4]
    bfs_prim_norms = bfs_data_as_np_arrays[5]
    bfs_expnts = bfs_data_as_np_arrays[6]

    is_gga = (xc_family_dict[x_family_code] != 'LDA' or xc_family_dict[c_family_code] != 'LDA')
    is_mgga = (xc_family_dict[x_family_code] == 'MGGA' or xc_family_dict[c_family_code] == 'MGGA')

    # AO values, gradients (and Hessians for GGA/MGGA)
    if is_gga:
        ao_value_block, ao_grad_block, ao_hess_block = Integrals.bf_val_helpers.eval_bfs_grad_and_hess_sparse_internal_serial(
            bfs_coords, bfs_contr_prim_norms, bfs_nprim, bfs_lmn, bfs_coeffs, bfs_prim_norms, bfs_expnts, coords_block, non_zero_indices)
    else:
        ao_value_block, ao_grad_block = Integrals.bf_val_helpers.eval_bfs_and_grad_sparse_internal_serial(
            bfs_coords, bfs_contr_prim_norms, bfs_nprim, bfs_lmn, bfs_coeffs, bfs_prim_norms, bfs_expnts, coords_block, non_zero_indices)

    # Density (and gradient of density for GGA, kinetic energy density for MGGA)
    Fmj = ao_value_block @ dmat  # Fmj[g, mu] = sum_nu D_mu_nu chi_nu(g)
    rho_block = np.einsum('mj,mj->m', Fmj, ao_value_block)

    sigma_block = None
    tau_block = None
    Hgrad = None
    if is_gga:
        rho_grad_x = 2.0 * np.einsum('mj,mj->m', Fmj, ao_grad_block[0])
        rho_grad_y = 2.0 * np.einsum('mj,mj->m', Fmj, ao_grad_block[1])
        rho_grad_z = 2.0 * np.einsum('mj,mj->m', Fmj, ao_grad_block[2])
        sigma_block = rho_grad_x**2 + rho_grad_y**2 + rho_grad_z**2
    if is_mgga:
        # Hk[g, mu] = sum_nu D_mu_nu (d_k chi_nu)(g) = (d_k chi @ D)[g, mu]
        Hgrad = [ao_grad_block[0] @ dmat, ao_grad_block[1] @ dmat, ao_grad_block[2] @ dmat]
        # tau = 0.5 sum_k sum_munu D_munu (d_k chi_mu)(d_k chi_nu)
        tau_block = 0.5 * (
            np.einsum('mj,mj->m', ao_grad_block[0], Hgrad[0])
            + np.einsum('mj,mj->m', ao_grad_block[1], Hgrad[1])
            + np.einsum('mj,mj->m', ao_grad_block[2], Hgrad[2])
        )

    # XC functional derivatives
    if use_libxc:
        inp = {'rho': rho_block}
        if xc_family_dict[x_family_code] != 'LDA':
            inp['sigma'] = sigma_block
        if xc_family_dict[x_family_code] == 'MGGA':
            inp['tau'] = tau_block
        retx = funcx.compute(inp)
        inp = {'rho': rho_block}
        if xc_family_dict[c_family_code] != 'LDA':
            inp['sigma'] = sigma_block
        if xc_family_dict[c_family_code] == 'MGGA':
            inp['tau'] = tau_block
        retc = funcc.compute(inp)
        vrho = (retx['vrho'] + retc['vrho'])[:, 0]
        vsigma = 0.0
        if xc_family_dict[x_family_code] != 'LDA':
            vsigma = vsigma + retx['vsigma'][:, 0]
        if xc_family_dict[c_family_code] != 'LDA':
            vsigma = vsigma + retc['vsigma'][:, 0]
        vtau = 0.0
        if xc_family_dict[x_family_code] == 'MGGA':
            vtau = vtau + retx['vtau'][:, 0]
        if xc_family_dict[c_family_code] == 'MGGA':
            vtau = vtau + retc['vtau'][:, 0]
    else:
        retx = XC.func_compute(funcid[0], rho_block, sigma=sigma_block, tau=tau_block, use_gpu=False)
        retc = XC.func_compute(funcid[1], rho_block, sigma=sigma_block, tau=tau_block, use_gpu=False)
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

    F = weights_block * vrho

    nbf_block = ao_value_block.shape[1]
    res = np.zeros((3, nbf_block))

    if not is_gga:
        # LDA: dexc_dbf[d, mu] = sum_g d_d chi_mu(g) * F(g) * (chi D)[g, mu]
        FD = F[:, None] * Fmj
        res[0] = np.einsum('mj,mj->j', ao_grad_block[0], FD)
        res[1] = np.einsum('mj,mj->j', ao_grad_block[1], FD)
        res[2] = np.einsum('mj,mj->j', ao_grad_block[2], FD)
    else:
        Fx = 2.0 * weights_block * vsigma * rho_grad_x
        Fy = 2.0 * weights_block * vsigma * rho_grad_y
        Fz = 2.0 * weights_block * vsigma * rho_grad_z

        # aow[g, nu] = F chi_nu + Fx d_x chi_nu + Fy d_y chi_nu + Fz d_z chi_nu
        aow = F[:, None] * ao_value_block
        aow += Fx[:, None] * ao_grad_block[0]
        aow += Fy[:, None] * ao_grad_block[1]
        aow += Fz[:, None] * ao_grad_block[2]
        aowD = aow @ dmat

        res[0] = np.einsum('mj,mj->j', ao_grad_block[0], aowD)
        res[1] = np.einsum('mj,mj->j', ao_grad_block[1], aowD)
        res[2] = np.einsum('mj,mj->j', ao_grad_block[2], aowD)

        # Hessian terms: sum_k Fk * d_k d_d chi_mu, contracted with (chi D)
        # ao_hess_block components: 0:xx 1:xy 2:xz 3:yy 4:yz 5:zz
        hessF = Fx[:, None] * ao_hess_block[0] + Fy[:, None] * ao_hess_block[1] + Fz[:, None] * ao_hess_block[2]
        res[0] += np.einsum('mj,mj->j', hessF, Fmj)
        hessF = Fx[:, None] * ao_hess_block[1] + Fy[:, None] * ao_hess_block[3] + Fz[:, None] * ao_hess_block[4]
        res[1] += np.einsum('mj,mj->j', hessF, Fmj)
        hessF = Fx[:, None] * ao_hess_block[2] + Fy[:, None] * ao_hess_block[4] + Fz[:, None] * ao_hess_block[5]
        res[2] += np.einsum('mj,mj->j', hessF, Fmj)

        if is_mgga:
            # tau term: dexc_dbf[d, mu] += sum_k sum_g G_tau (d_d d_k chi_mu) Hk[g, mu]
            # G_tau = 0.5 w vtau ; Hk = (d_k chi) @ D
            # d=x : (xx, xy, xz) . (Hx, Hy, Hz)
            # d=y : (xy, yy, yz) . (Hx, Hy, Hz)
            # d=z : (xz, yz, zz) . (Hx, Hy, Hz)
            Gtau = 0.5 * weights_block * vtau
            GHx = Gtau[:, None] * Hgrad[0]
            GHy = Gtau[:, None] * Hgrad[1]
            GHz = Gtau[:, None] * Hgrad[2]
            res[0] += (np.einsum('mj,mj->j', ao_hess_block[0], GHx)
                       + np.einsum('mj,mj->j', ao_hess_block[1], GHy)
                       + np.einsum('mj,mj->j', ao_hess_block[2], GHz))
            res[1] += (np.einsum('mj,mj->j', ao_hess_block[1], GHx)
                       + np.einsum('mj,mj->j', ao_hess_block[3], GHy)
                       + np.einsum('mj,mj->j', ao_hess_block[4], GHz))
            res[2] += (np.einsum('mj,mj->j', ao_hess_block[2], GHx)
                       + np.einsum('mj,mj->j', ao_hess_block[4], GHy)
                       + np.einsum('mj,mj->j', ao_hess_block[5], GHz))

    return res
