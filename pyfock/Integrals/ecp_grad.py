"""
Analytical nuclear gradient of the scalar semilocal ECP integrals.

This differentiates the *series* ECP integrals implemented in
``ecp_mat_symm`` (the routine used by the SCF), so the resulting forces are
consistent with the SCF energy surface.

The derivative of a basis function with respect to its own center obeys the
Gaussian relation (per primitive)

    d chi / dA_d = 2*alpha * chi^{+1_d} - l_d * chi^{-1_d}

and the ECP operator does not depend on the AO centers, so the bra-center
derivative of every ECP matrix element is built from the same machinery with
shifted angular momenta. The operator (ECP-center) derivative follows from
translational invariance.

For a fixed ECP center C, with M_ij = <chi_i | U_C | chi_j> and the
bra-derivative matrix dVbra[d, i, j] = dM_ij / dA_{i,d}, the contribution to
the energy gradient (E_ecp = sum_ij D_ij M_ij) is, defining
B_k[d] = sum_m dVbra[d, k, m] D_km,

    grad[atom(k), d] += 2 * B_k[d]          (basis-function / bra + ket)
    grad[atom(C), d] -= 2 * sum_k B_k[d]    (operator center, translational inv.)
"""

import math

import numpy as np
from numba import njit, prange

from .integral_helpers import calcS
from .ecp_mat_symm import (
    basis_to_arrays,
    ecp_to_arrays,
    _projector_kernel_terms,
    _channel_arrays,
    _comb_int_nb,
    _pow_int_nb,
    _factorial_int_nb,
    _sphere_monomial_integral_nb,
)


def ecp_grad_contract(basis, mol, dmat, series_order=12, coeff_tol=1.0e-16, power_shift=2):
    """
    Analytical ECP contribution to the DFT nuclear gradient.

    Returns ``grad`` of shape ``(natoms, 3)`` with
    ``grad[A, d] = sum_ij D_ij dV_ecp_ij / dR_{A, d}``.
    """
    natoms = mol.natoms
    grad = np.zeros((natoms, 3), dtype=np.float64)
    if not getattr(basis, 'has_ecp', False):
        return grad

    arrays = basis_to_arrays(basis)
    ecp_data = ecp_to_arrays(basis, power_shift)
    if np.any(ecp_data['local_quad_counts'] > 0):
        raise NotImplementedError('Analytical ECP gradient currently requires local power == power_shift.')

    dmat = np.ascontiguousarray(dmat, dtype=np.float64)
    bfs_atoms = np.asarray(basis.bfs_atoms, dtype=np.int64)
    ecp_atom_index = np.array([ecp['atom_index'] for ecp in basis.ecps], dtype=np.int64)

    coords = arrays['coords']
    contr_norms = arrays['contr_norms']
    lmn = arrays['lmn']
    nprim = arrays['nprim']
    coeffs = arrays['coeffs']
    prim_norms = arrays['prim_norms']
    expnts = arrays['expnts']

    # ----------------- Local (Gaussian) part -----------------
    for iecp in range(ecp_data['coords'].shape[0]):
        n_local = int(ecp_data['local_counts'][iecp])
        if n_local == 0:
            continue
        C = ecp_data['coords'][iecp]
        dVbra = ecp_local_bra_grad_internal(
            coords, contr_norms, lmn, nprim, coeffs, prim_norms, expnts,
            C[0], C[1], C[2],
            ecp_data['local_coeffs'][iecp], ecp_data['local_expnts'][iecp], n_local,
        )
        _accumulate_center_grad(grad, dVbra, dmat, bfs_atoms, int(ecp_atom_index[iecp]))

    # ----------------- Projector part -----------------
    channels_by_ecp = {}
    for channel in ecp_data['channels']:
        channels_by_ecp.setdefault(channel['ecp_index'], []).append(channel)

    for iecp, channels in channels_by_ecp.items():
        center = ecp_data['coords'][iecp]

        kernel_by_l = {}
        moment_powers = set()
        for channel in channels:
            l = channel['l']
            kernel_by_l[l] = _projector_kernel_terms(l)
            for power in kernel_by_l[l]:
                moment_powers.add(power)
        moment_powers = sorted(moment_powers)
        moment_power_array = np.array(moment_powers, dtype=np.int64)

        (mom_counts, mom_alphas, mom_powers, mom_coeffs,
         d_counts, d_alphas, d_powers, d_coeffs) = _build_moment_and_grad_arrays_internal(
            coords, contr_norms, lmn, nprim, coeffs, prim_norms, expnts,
            center, moment_power_array, series_order, coeff_tol,
        )

        (kernel_counts, kernel_indices, kernel_coeffs,
         ch_term_counts, ch_term_coeffs, ch_term_powers, ch_term_expnts) = _channel_arrays(
            channels, kernel_by_l, moment_powers)

        max_power = _grad_max_radial_power(mom_powers, mom_counts, d_powers, d_counts,
                                           ch_term_powers, power_shift)
        gamma_half = np.zeros(max_power + 1, dtype=np.float64)
        for p in range(max_power + 1):
            gamma_half[p] = 0.5 * math.gamma(0.5 * (p + 1))

        dVbra = _assemble_projector_bra_grad_internal(
            mom_counts, mom_alphas, mom_powers, mom_coeffs,
            d_counts, d_alphas, d_powers, d_coeffs,
            kernel_counts, kernel_indices, kernel_coeffs,
            ch_term_counts, ch_term_coeffs, ch_term_powers, ch_term_expnts,
            gamma_half, power_shift,
        )
        _accumulate_center_grad(grad, dVbra, dmat, bfs_atoms, int(ecp_atom_index[iecp]))

    return grad


def _accumulate_center_grad(grad, dVbra, dmat, bfs_atoms, ecp_atom):
    """
    Map a single ECP center's bra-derivative matrix dVbra[d,i,j] onto atoms:

        B[i, d] = sum_j dVbra[d, i, j] D_ij
        grad[atom(i)] += 2 B[i] ;  grad[ecp_atom] -= 2 sum_i B[i]
    """
    # B[i, d] = sum_j dVbra[d, i, j] * D[i, j]
    B = np.einsum('dij,ij->id', dVbra, dmat, optimize=True)
    np.add.at(grad, bfs_atoms, 2.0 * B)
    grad[ecp_atom] -= 2.0 * B.sum(axis=0)


@njit(parallel=True, cache=True, fastmath=True, error_model='numpy')
def ecp_local_bra_grad_internal(bfs_coords, bfs_contr_prim_norms, bfs_lmn, bfs_nprim,
                                bfs_coeffs, bfs_prim_norms, bfs_expnts,
                                Cx, Cy, Cz, local_coeffs, local_expnts, n_local):
    """Bra-center derivative of the local (Gaussian) ECP matrix for one center."""
    nao = bfs_coords.shape[0]
    dV = np.zeros((3, nao, nao), dtype=np.float64)

    for i in prange(nao):
        Ix = bfs_coords[i, 0]
        Iy = bfs_coords[i, 1]
        Iz = bfs_coords[i, 2]
        lix = bfs_lmn[i, 0]
        liy = bfs_lmn[i, 1]
        liz = bfs_lmn[i, 2]
        Ni = bfs_contr_prim_norms[i]
        I2 = Ix * Ix + Iy * Iy + Iz * Iz
        for j in range(nao):
            Jx = bfs_coords[j, 0]
            Jy = bfs_coords[j, 1]
            Jz = bfs_coords[j, 2]
            ljx = bfs_lmn[j, 0]
            ljy = bfs_lmn[j, 1]
            ljz = bfs_lmn[j, 2]
            Nj = bfs_contr_prim_norms[j]
            J2 = Jx * Jx + Jy * Jy + Jz * Jz

            gx = 0.0
            gy = 0.0
            gz = 0.0
            for iterm in range(n_local):
                ecp_coeff = local_coeffs[iterm]
                ecp_expnt = local_expnts[iterm]
                C2 = ecp_expnt * (Cx * Cx + Cy * Cy + Cz * Cz)
                for ik in range(bfs_nprim[i]):
                    alphaik = bfs_expnts[i, ik]
                    dik = bfs_coeffs[i, ik]
                    Nik = bfs_prim_norms[i, ik]
                    for jk in range(bfs_nprim[j]):
                        alphajk = bfs_expnts[j, jk]
                        djk = bfs_coeffs[j, jk]
                        Njk = bfs_prim_norms[j, jk]

                        gamma = alphaik + alphajk + ecp_expnt
                        Px = (alphaik * Ix + alphajk * Jx + ecp_expnt * Cx) / gamma
                        Py = (alphaik * Iy + alphajk * Jy + ecp_expnt * Cy) / gamma
                        Pz = (alphaik * Iz + alphajk * Jz + ecp_expnt * Cz) / gamma
                        exponent = (-alphaik * I2 - alphajk * J2 - C2
                                    + gamma * (Px * Px + Py * Py + Pz * Pz))
                        screen = math.exp(exponent)
                        # Use the same screening threshold as the energy routine
                        # (ecp_local_gaussian_mat_symm_internal) so the gradient is
                        # consistent with the SCF energy surface.
                        if abs(screen) < 1.0e-8:
                            continue

                        PIx = Px - Ix
                        PIy = Py - Iy
                        PIz = Pz - Iz
                        PJx = Px - Jx
                        PJy = Py - Jy
                        PJz = Pz - Jz

                        Sx0 = calcS(lix, ljx, gamma, PIx, PJx)
                        Sy0 = calcS(liy, ljy, gamma, PIy, PJy)
                        Sz0 = calcS(liz, ljz, gamma, PIz, PJz)

                        base = ecp_coeff * dik * djk * Nik * Njk * Ni * Nj * screen
                        two_alpha = 2.0 * alphaik

                        dSx = two_alpha * calcS(lix + 1, ljx, gamma, PIx, PJx)
                        if lix > 0:
                            dSx -= lix * calcS(lix - 1, ljx, gamma, PIx, PJx)
                        dSy = two_alpha * calcS(liy + 1, ljy, gamma, PIy, PJy)
                        if liy > 0:
                            dSy -= liy * calcS(liy - 1, ljy, gamma, PIy, PJy)
                        dSz = two_alpha * calcS(liz + 1, ljz, gamma, PIz, PJz)
                        if liz > 0:
                            dSz -= liz * calcS(liz - 1, ljz, gamma, PIz, PJz)

                        gx += base * dSx * Sy0 * Sz0
                        gy += base * Sx0 * dSy * Sz0
                        gz += base * Sx0 * Sy0 * dSz

            dV[0, i, j] = gx
            dV[1, i, j] = gy
            dV[2, i, j] = gz

    return dV


@njit(cache=True, fastmath=True, error_model='numpy')
def _accum_powers(power_coeffs, lx, ly, lz, dx, dy, dz, alpha, mx, my, mz, series_order, weight):
    """Accumulate weight * poly * taylor * angular into power_coeffs[r_power]."""
    if weight == 0.0:
        return
    two_alpha = 2.0 * alpha
    for ax in range(lx + 1):
        cx = _comb_int_nb(lx, ax) * _pow_int_nb(-dx, lx - ax)
        if cx == 0.0:
            continue
        for ay in range(ly + 1):
            cy = _comb_int_nb(ly, ay) * _pow_int_nb(-dy, ly - ay)
            if cy == 0.0:
                continue
            for az in range(lz + 1):
                cz = _comb_int_nb(lz, az) * _pow_int_nb(-dz, lz - az)
                poly_coeff = cx * cy * cz
                if poly_coeff == 0.0:
                    continue
                poly_power = ax + ay + az
                for tx in range(series_order + 1):
                    x_factor = _pow_int_nb(dx, tx) / _factorial_int_nb(tx)
                    for ty in range(series_order + 1 - tx):
                        y_factor = _pow_int_nb(dy, ty) / _factorial_int_nb(ty)
                        for tz in range(series_order + 1 - tx - ty):
                            t = tx + ty + tz
                            exp_coeff = _pow_int_nb(two_alpha, t) * x_factor * y_factor
                            exp_coeff *= _pow_int_nb(dz, tz) / _factorial_int_nb(tz)
                            if exp_coeff == 0.0:
                                continue
                            angular = _sphere_monomial_integral_nb(mx + ax + tx, my + ay + ty, mz + az + tz)
                            if angular == 0.0:
                                continue
                            r_power = poly_power + t
                            power_coeffs[r_power] += weight * poly_coeff * exp_coeff * angular


@njit(cache=True, fastmath=True, error_model='numpy')
def _build_moment_and_grad_arrays_internal(
    bfs_coords, bfs_contr_prim_norms, bfs_lmn, bfs_nprim, bfs_coeffs, bfs_prim_norms, bfs_expnts,
    ecp_center, moment_powers, series_order, coeff_tol,
):
    """Original moment arrays AND their bra-center derivative moment arrays."""
    nao = bfs_coords.shape[0]
    nmoment = moment_powers.shape[0]
    max_nprim = bfs_coeffs.shape[1]
    max_l = 0
    for i in range(nao):
        lsum = bfs_lmn[i, 0] + bfs_lmn[i, 1] + bfs_lmn[i, 2]
        if lsum > max_l:
            max_l = lsum

    # derivative shifts l by +1, so allow one extra radial power
    max_r_power = max_l + 1 + series_order
    max_terms = max_nprim * (max_r_power + 1)
    if max_terms < 1:
        max_terms = 1

    mom_counts = np.zeros((nao, nmoment), dtype=np.int64)
    mom_alphas = np.zeros((nao, nmoment, max_terms), dtype=np.float64)
    mom_powers = np.zeros((nao, nmoment, max_terms), dtype=np.int64)
    mom_coeffs = np.zeros((nao, nmoment, max_terms), dtype=np.float64)

    d_counts = np.zeros((nao, 3, nmoment), dtype=np.int64)
    d_alphas = np.zeros((nao, 3, nmoment, max_terms), dtype=np.float64)
    d_powers = np.zeros((nao, 3, nmoment, max_terms), dtype=np.int64)
    d_coeffs = np.zeros((nao, 3, nmoment, max_terms), dtype=np.float64)

    pc_o = np.zeros(max_r_power + 1, dtype=np.float64)
    pc_x = np.zeros(max_r_power + 1, dtype=np.float64)
    pc_y = np.zeros(max_r_power + 1, dtype=np.float64)
    pc_z = np.zeros(max_r_power + 1, dtype=np.float64)

    for i in range(nao):
        lx = bfs_lmn[i, 0]
        ly = bfs_lmn[i, 1]
        lz = bfs_lmn[i, 2]
        dx = bfs_coords[i, 0] - ecp_center[0]
        dy = bfs_coords[i, 1] - ecp_center[1]
        dz = bfs_coords[i, 2] - ecp_center[2]
        d2 = dx * dx + dy * dy + dz * dz
        contr_norm = bfs_contr_prim_norms[i]

        for imoment in range(nmoment):
            mx = moment_powers[imoment, 0]
            my = moment_powers[imoment, 1]
            mz = moment_powers[imoment, 2]
            io = 0
            ix_ = 0
            iy_ = 0
            iz_ = 0

            for iprim in range(bfs_nprim[i]):
                alpha = bfs_expnts[i, iprim]
                prim_coeff = bfs_coeffs[i, iprim] * bfs_prim_norms[i, iprim] * contr_norm
                prefactor = prim_coeff * math.exp(-alpha * d2)
                if prefactor == 0.0:
                    continue

                for ir in range(max_r_power + 1):
                    pc_o[ir] = 0.0
                    pc_x[ir] = 0.0
                    pc_y[ir] = 0.0
                    pc_z[ir] = 0.0

                # original moment
                _accum_powers(pc_o, lx, ly, lz, dx, dy, dz, alpha, mx, my, mz, series_order, prefactor)
                # derivative moments: 2*alpha*(l+1) - l*(l-1) per direction
                _accum_powers(pc_x, lx + 1, ly, lz, dx, dy, dz, alpha, mx, my, mz, series_order, prefactor * 2.0 * alpha)
                if lx > 0:
                    _accum_powers(pc_x, lx - 1, ly, lz, dx, dy, dz, alpha, mx, my, mz, series_order, -prefactor * lx)
                _accum_powers(pc_y, lx, ly + 1, lz, dx, dy, dz, alpha, mx, my, mz, series_order, prefactor * 2.0 * alpha)
                if ly > 0:
                    _accum_powers(pc_y, lx, ly - 1, lz, dx, dy, dz, alpha, mx, my, mz, series_order, -prefactor * ly)
                _accum_powers(pc_z, lx, ly, lz + 1, dx, dy, dz, alpha, mx, my, mz, series_order, prefactor * 2.0 * alpha)
                if lz > 0:
                    _accum_powers(pc_z, lx, ly, lz - 1, dx, dy, dz, alpha, mx, my, mz, series_order, -prefactor * lz)

                for r_power in range(max_r_power + 1):
                    if abs(pc_o[r_power]) > coeff_tol:
                        mom_alphas[i, imoment, io] = alpha
                        mom_powers[i, imoment, io] = r_power
                        mom_coeffs[i, imoment, io] = pc_o[r_power]
                        io += 1
                    if abs(pc_x[r_power]) > coeff_tol:
                        d_alphas[i, 0, imoment, ix_] = alpha
                        d_powers[i, 0, imoment, ix_] = r_power
                        d_coeffs[i, 0, imoment, ix_] = pc_x[r_power]
                        ix_ += 1
                    if abs(pc_y[r_power]) > coeff_tol:
                        d_alphas[i, 1, imoment, iy_] = alpha
                        d_powers[i, 1, imoment, iy_] = r_power
                        d_coeffs[i, 1, imoment, iy_] = pc_y[r_power]
                        iy_ += 1
                    if abs(pc_z[r_power]) > coeff_tol:
                        d_alphas[i, 2, imoment, iz_] = alpha
                        d_powers[i, 2, imoment, iz_] = r_power
                        d_coeffs[i, 2, imoment, iz_] = pc_z[r_power]
                        iz_ += 1

            mom_counts[i, imoment] = io
            d_counts[i, 0, imoment] = ix_
            d_counts[i, 1, imoment] = iy_
            d_counts[i, 2, imoment] = iz_

    return (mom_counts, mom_alphas, mom_powers, mom_coeffs,
            d_counts, d_alphas, d_powers, d_coeffs)


def _grad_max_radial_power(mom_powers, mom_counts, d_powers, d_counts, channel_term_powers, power_shift):
    max_mom = 0
    for i in range(mom_counts.shape[0]):
        for im in range(mom_counts.shape[1]):
            for t in range(mom_counts[i, im]):
                if mom_powers[i, im, t] > max_mom:
                    max_mom = mom_powers[i, im, t]
    max_d = 0
    for i in range(d_counts.shape[0]):
        for dd in range(3):
            for im in range(d_counts.shape[2]):
                for t in range(d_counts[i, dd, im]):
                    if d_powers[i, dd, im, t] > max_d:
                        max_d = d_powers[i, dd, im, t]
    max_ecp = 0
    for power in np.ravel(channel_term_powers):
        if int(power) > max_ecp:
            max_ecp = int(power)
    return max(0, max_mom + max_d + max_ecp - power_shift + 2)


@njit(parallel=True, cache=True, fastmath=True, error_model='numpy')
def _assemble_projector_bra_grad_internal(
    mom_counts, mom_alphas, mom_powers, mom_coeffs,
    d_counts, d_alphas, d_powers, d_coeffs,
    kernel_counts, kernel_indices, kernel_coeffs,
    channel_term_counts, channel_term_coeffs, channel_term_powers, channel_term_expnts,
    gamma_half, power_shift,
):
    """dVbra[d, i, j] = sum_channels kernel * contract(dmom_i[d], mom_j) for one ECP center."""
    nao = mom_counts.shape[0]
    nchannel = channel_term_counts.shape[0]
    dV = np.zeros((3, nao, nao), dtype=np.float64)

    for i in prange(nao):
        for j in range(nao):
            for ddir in range(3):
                value = 0.0
                for ichannel in range(nchannel):
                    for ikernel in range(kernel_counts[ichannel]):
                        imoment = kernel_indices[ichannel, ikernel]
                        ni = d_counts[i, ddir, imoment]
                        nj = mom_counts[j, imoment]
                        if ni == 0 or nj == 0:
                            continue
                        moment_value = 0.0
                        for ti in range(ni):
                            alpha_i = d_alphas[i, ddir, imoment, ti]
                            power_i = d_powers[i, ddir, imoment, ti]
                            coeff_i = d_coeffs[i, ddir, imoment, ti]
                            for tj in range(nj):
                                alpha_j = mom_alphas[j, imoment, tj]
                                power_j = mom_powers[j, imoment, tj]
                                coeff_ij = coeff_i * mom_coeffs[j, imoment, tj]
                                alpha_ij = alpha_i + alpha_j
                                for iterm in range(channel_term_counts[ichannel]):
                                    radial_power = power_i + power_j + channel_term_powers[ichannel, iterm] - power_shift + 2
                                    exponent = alpha_ij + channel_term_expnts[ichannel, iterm]
                                    radial = gamma_half[radial_power] * exponent**(-0.5 * (radial_power + 1))
                                    moment_value += channel_term_coeffs[ichannel, iterm] * coeff_ij * radial
                        value += kernel_coeffs[ichannel, ikernel] * moment_value
                dV[ddir, i, j] = value

    return dV
