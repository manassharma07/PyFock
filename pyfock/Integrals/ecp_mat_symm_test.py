import math

import numpy as np
from numba import njit, prange

from .integral_helpers import calcS

try:
    from scipy.special import sph_harm_y as _sph_harm_y
except Exception:
    _sph_harm_y = None
    from scipy.special import sph_harm as _sph_harm


def ecp_mat_symm_test(basis, slice=None, n_radial=128, n_theta=18, n_phi=36, power_shift=2):
    """
    Compute the scalar semilocal ECP matrix in the Cartesian AO basis.

    ECP terms are interpreted as
        c r^(n - power_shift) exp(-alpha r^2).
    For the def2 ECP files this gives the usual c exp(-alpha r^2)
    terms because their tabulated radial powers are n = 2.
    """
    if not getattr(basis, 'has_ecp', False):
        if slice is None:
            return np.zeros((basis.bfs_nao, basis.bfs_nao))
        return np.zeros((int(slice[1]) - int(slice[0]), int(slice[3]) - int(slice[2])))

    arrays = basis_to_arrays(basis)
    ecp_data = ecp_to_arrays(basis, power_shift)

    if slice is None:
        slice = [0, basis.bfs_nao, 0, basis.bfs_nao]

    start_row = int(slice[0])
    end_row = int(slice[1])
    start_col = int(slice[2])
    end_col = int(slice[3])

    V = ecp_local_gaussian_mat_symm_internal(arrays['coords'], arrays['contr_norms'], arrays['lmn'], arrays['nprim'], arrays['coeffs'], arrays['prim_norms'], arrays['expnts'], ecp_data['coords'], ecp_data['local_counts'], ecp_data['local_coeffs'], ecp_data['local_expnts'], start_row, end_row, start_col, end_col)

    needs_quadrature = len(ecp_data['channels']) > 0 or np.any(ecp_data['local_quad_counts'] > 0)
    if needs_quadrature:
        V_quad = _ecp_projector_quadrature_mat(arrays, ecp_data, basis.bfs_nao, n_radial=n_radial, n_theta=n_theta, n_phi=n_phi, power_shift=power_shift)
        V = V + V_quad[start_row:end_row, start_col:end_col]

    return V


def basis_to_arrays(basis):
    bfs_coords = np.array(basis.bfs_coords, dtype=np.float64)
    bfs_contr_prim_norms = np.array(basis.bfs_contr_prim_norms, dtype=np.float64)
    bfs_lmn = np.array(basis.bfs_lmn, dtype=np.int64)
    bfs_nprim = np.array(basis.bfs_nprim, dtype=np.int64)
    bfs_radius_cutoff = np.array(basis.bfs_radius_cutoff, dtype=np.float64)

    maxnprim = max(basis.bfs_nprim)
    bfs_coeffs = np.zeros([basis.bfs_nao, maxnprim], dtype=np.float64)
    bfs_expnts = np.zeros([basis.bfs_nao, maxnprim], dtype=np.float64)
    bfs_prim_norms = np.zeros([basis.bfs_nao, maxnprim], dtype=np.float64)
    for i in range(basis.bfs_nao):
        for j in range(basis.bfs_nprim[i]):
            bfs_coeffs[i, j] = basis.bfs_coeffs[i][j]
            bfs_expnts[i, j] = basis.bfs_expnts[i][j]
            bfs_prim_norms[i, j] = basis.bfs_prim_norms[i][j]

    return {
        'coords': bfs_coords,
        'contr_norms': bfs_contr_prim_norms,
        'lmn': bfs_lmn,
        'nprim': bfs_nprim,
        'radius_cutoff': bfs_radius_cutoff,
        'coeffs': bfs_coeffs,
        'expnts': bfs_expnts,
        'prim_norms': bfs_prim_norms,
    }


def ecp_to_arrays(basis, power_shift):
    necp = len(basis.ecps)
    coords = np.zeros((necp, 3), dtype=np.float64)
    max_local = 1
    max_local_quad = 1
    for ecp in basis.ecps:
        max_local = max(max_local, sum(1 for term in ecp['local_terms'] if term[1] == power_shift))
        max_local_quad = max(max_local_quad, sum(1 for term in ecp['local_terms'] if term[1] != power_shift))

    local_counts = np.zeros(necp, dtype=np.int64)
    local_coeffs = np.zeros((necp, max_local), dtype=np.float64)
    local_expnts = np.zeros((necp, max_local), dtype=np.float64)
    local_quad_counts = np.zeros(necp, dtype=np.int64)
    local_quad_terms = np.zeros((necp, max_local_quad, 3), dtype=np.float64)
    channels = []

    for i, ecp in enumerate(basis.ecps):
        coords[i, :] = ecp['coords']
        ilocal = 0
        ilocal_quad = 0
        for coeff, power, exponent in ecp['local_terms']:
            if power == power_shift:
                local_coeffs[i, ilocal] = coeff
                local_expnts[i, ilocal] = exponent
                ilocal += 1
            else:
                local_quad_terms[i, ilocal_quad, 0] = coeff
                local_quad_terms[i, ilocal_quad, 1] = power
                local_quad_terms[i, ilocal_quad, 2] = exponent
                ilocal_quad += 1
        local_counts[i] = ilocal
        local_quad_counts[i] = ilocal_quad

        for l in sorted(ecp['projector_terms']):
            terms = np.array(ecp['projector_terms'][l], dtype=np.float64)
            if terms.size == 0:
                continue
            channels.append({
                'ecp_index': i,
                'l': l,
                'terms': terms,
            })

    return {
        'coords': coords,
        'local_counts': local_counts,
        'local_coeffs': local_coeffs,
        'local_expnts': local_expnts,
        'local_quad_counts': local_quad_counts,
        'local_quad_terms': local_quad_terms,
        'channels': channels,
    }


@njit(parallel=True, cache=True, fastmath=True, error_model='numpy')
def ecp_local_gaussian_mat_symm_internal(bfs_coords, bfs_contr_prim_norms, bfs_lmn, bfs_nprim, bfs_coeffs, bfs_prim_norms, bfs_expnts, ecp_coords, local_counts, local_coeffs, local_expnts, start_row, end_row, start_col, end_col):
    num_rows = end_row - start_row
    num_cols = end_col - start_col

    upper_tri = False
    lower_tri = False
    both_tri_symm = False
    both_tri_nonsymm = False
    if end_row <= start_col:
        upper_tri = True
    elif start_row >= end_col:
        lower_tri = True
    elif start_row == start_col and end_row == end_col:
        both_tri_symm = True
    else:
        both_tri_nonsymm = True

    V = np.zeros((num_rows, num_cols))

    for i in prange(start_row, end_row):
        I = bfs_coords[i]
        lmni = bfs_lmn[i]
        Ni = bfs_contr_prim_norms[i]
        for j in prange(start_col, end_col):
            if lower_tri or upper_tri or (both_tri_symm and j <= i) or both_tri_nonsymm:
                J = bfs_coords[j]
                lmnj = bfs_lmn[j]
                Nj = bfs_contr_prim_norms[j]
                result = 0.0

                for iecp in range(ecp_coords.shape[0]):
                    C = ecp_coords[iecp]
                    for iterm in range(local_counts[iecp]):
                        ecp_coeff = local_coeffs[iecp, iterm]
                        ecp_expnt = local_expnts[iecp, iterm]

                        for ik in range(bfs_nprim[i]):
                            alphaik = bfs_expnts[i, ik]
                            dik = bfs_coeffs[i, ik]
                            Nik = bfs_prim_norms[i, ik]
                            for jk in range(bfs_nprim[j]):
                                alphajk = bfs_expnts[j, jk]
                                djk = bfs_coeffs[j, jk]
                                Njk = bfs_prim_norms[j, jk]

                                gamma = alphaik + alphajk + ecp_expnt
                                P = (alphaik*I + alphajk*J + ecp_expnt*C)/gamma
                                exponent = -alphaik*np.dot(I, I) - alphajk*np.dot(J, J) - ecp_expnt*np.dot(C, C) + gamma*np.dot(P, P)
                                screenfactor = math.exp(exponent)
                                if abs(screenfactor) < 1.0e-8:
                                    continue

                                PI = P - I
                                PJ = P - J
                                Sx = calcS(lmni[0], lmnj[0], gamma, PI[0], PJ[0])
                                Sy = calcS(lmni[1], lmnj[1], gamma, PI[1], PJ[1])
                                Sz = calcS(lmni[2], lmnj[2], gamma, PI[2], PJ[2])

                                temp = ecp_coeff*dik*djk*Nik*Njk*Ni*Nj
                                result += temp*screenfactor*Sx*Sy*Sz

                V[i - start_row, j - start_col] = result

    if both_tri_symm:
        for i in prange(start_row, end_row):
            for j in prange(start_col, end_col):
                if j > i:
                    V[i - start_row, j - start_col] = V[j - start_col, i - start_row]

    return V


def _ecp_projector_quadrature_mat(
    arrays,
    ecp_data,
    nao,
    n_radial=128,
    n_theta=18,
    n_phi=36,
    power_shift=2,
):
    dirs, ang_weights = _angular_grid(n_theta, n_phi)
    radial_points, radial_weights = _gauss_chebyshev_radial(n_radial)

    max_l = -1
    for channel in ecp_data['channels']:
        max_l = max(max_l, channel['l'])
    ylm = _real_spherical_harmonics(max_l, dirs) if max_l >= 0 else None

    V = np.zeros((nao, nao), dtype=np.float64)
    for iecp in range(ecp_data['coords'].shape[0]):
        center = ecp_data['coords'][iecp]
        center_channels = [channel for channel in ecp_data['channels'] if channel['ecp_index'] == iecp]

        for ir in range(radial_points.shape[0]):
            r = radial_points[ir]
            wr = radial_weights[ir]
            if wr == 0.0:
                continue
            points = center + r*dirs
            ao = _eval_bfs_grid_internal(
                arrays['coords'],
                arrays['contr_norms'],
                arrays['nprim'],
                arrays['lmn'],
                arrays['coeffs'],
                arrays['prim_norms'],
                arrays['expnts'],
                arrays['radius_cutoff'],
                points,
            )

            radial_measure = wr*r*r

            local_value = _eval_local_quad_terms(ecp_data['local_quad_terms'][iecp], ecp_data['local_quad_counts'][iecp], r, power_shift)
            if local_value != 0.0:
                weighted_ao = ao*ang_weights[:, None]
                V += radial_measure*local_value*(ao.T @ weighted_ao)

            for channel in center_channels:
                channel_value = _eval_ecp_terms(channel['terms'], r, power_shift)
                if channel_value == 0.0:
                    continue
                l = channel['l']
                offset = l*l
                yblock = ylm[offset:offset + 2*l + 1, :].T
                weighted_y = yblock*ang_weights[:, None]
                proj = ao.T @ weighted_y
                V += radial_measure*channel_value*(proj @ proj.T)

    return 0.5*(V + V.T)


def _eval_ecp_terms(terms, r, power_shift):
    r2 = r*r
    value = 0.0
    for coeff, power, exponent in terms:
        radial_power = int(power) - power_shift
        if radial_power == 0:
            rpow = 1.0
        elif r == 0.0 and radial_power < 0:
            rpow = 0.0
        else:
            rpow = r**radial_power
        value += coeff*rpow*math.exp(-exponent*r2)
    return value


def _eval_local_quad_terms(terms, nterms, r, power_shift):
    if nterms == 0:
        return 0.0
    return _eval_ecp_terms(terms[:nterms], r, power_shift)


@njit(parallel=True, cache=True, fastmath=True, error_model='numpy')
def _eval_bfs_grid_internal(
    bfs_coords,
    bfs_contr_prim_norms,
    bfs_nprim,
    bfs_lmn,
    bfs_coeffs,
    bfs_prim_norms,
    bfs_expnts,
    bfs_radius_cutoff,
    coord,
):
    ncoord = coord.shape[0]
    nao = bfs_coords.shape[0]
    result = np.zeros((ncoord, nao))

    for k in prange(ncoord):
        coord_grid = coord[k]
        for i in range(nao):
            coord_bf = bfs_coords[i]
            x = coord_grid[0] - coord_bf[0]
            y = coord_grid[1] - coord_bf[1]
            z = coord_grid[2] - coord_bf[2]
            r2 = x*x + y*y + z*z
            if math.sqrt(r2) > bfs_radius_cutoff[i]:
                continue

            Ni = bfs_contr_prim_norms[i]
            lmni = bfs_lmn[i]
            poly = _pow_int(x, lmni[0]) * _pow_int(y, lmni[1]) * _pow_int(z, lmni[2])
            value = 0.0
            for ik in range(bfs_nprim[i]):
                value += bfs_coeffs[i, ik]*bfs_prim_norms[i, ik]*math.exp(-bfs_expnts[i, ik]*r2)
            result[k, i] = Ni*poly*value

    return result


@njit(cache=True, fastmath=True)
def _pow_int(x, n):
    value = 1.0
    for i in range(n):
        value *= x
    return value


def _angular_grid(n_theta, n_phi):
    cos_theta, weights_theta = np.polynomial.legendre.leggauss(n_theta)
    phi = 2.0*np.pi*(np.arange(n_phi, dtype=np.float64) + 0.5)/n_phi
    weights_phi = np.full(n_phi, 2.0*np.pi/n_phi, dtype=np.float64)

    dirs = np.zeros((n_theta*n_phi, 3), dtype=np.float64)
    weights = np.zeros(n_theta*n_phi, dtype=np.float64)
    idx = 0
    for itheta in range(n_theta):
        z = cos_theta[itheta]
        sintheta = math.sqrt(max(0.0, 1.0 - z*z))
        for iphi in range(n_phi):
            dirs[idx, 0] = sintheta*math.cos(phi[iphi])
            dirs[idx, 1] = sintheta*math.sin(phi[iphi])
            dirs[idx, 2] = z
            weights[idx] = weights_theta[itheta]*weights_phi[iphi]
            idx += 1
    return dirs, weights


def _gauss_chebyshev_radial(n_radial):
    r = np.zeros(n_radial, dtype=np.float64)
    w = np.zeros(n_radial, dtype=np.float64)
    step = 1.0/(n_radial + 1.0)
    log2 = math.log(2.0)
    x1 = 0.0
    for i in range(n_radial):
        x1 += math.pi*step
        x2 = math.sin(x1)
        x3 = math.sin(2.0*x1)
        x4 = x2*x2
        xi = (n_radial - 2.0*i - 1.0)*step + (1.0 + 2.0*x4/3.0)*x3/math.pi
        r[i] = 1.0 - math.log(1.0 + xi)/log2
        w[i] = 16.0*step*x4*x4/(3.0*log2*(1.0 + xi))
    return r, w


def _complex_sph_harm(m, l, theta, phi):
    if _sph_harm_y is not None:
        return _sph_harm_y(l, m, theta, phi)
    return _sph_harm(m, l, phi, theta)


def _real_spherical_harmonics(max_l, dirs):
    nang = dirs.shape[0]
    ylm = np.zeros(((max_l + 1)*(max_l + 1), nang), dtype=np.float64)
    z = np.clip(dirs[:, 2], -1.0, 1.0)
    theta = np.arccos(z)
    phi = np.mod(np.arctan2(dirs[:, 1], dirs[:, 0]), 2.0*np.pi)

    for l in range(max_l + 1):
        for m in range(-l, l + 1):
            idx = l*l + (m + l)
            if m < 0:
                ycomplex = _complex_sph_harm(-m, l, theta, phi)
                sign = -1.0 if ((-m) % 2) else 1.0
                ylm[idx, :] = math.sqrt(2.0)*sign*ycomplex.imag
            elif m == 0:
                ylm[idx, :] = _complex_sph_harm(0, l, theta, phi).real
            else:
                ycomplex = _complex_sph_harm(m, l, theta, phi)
                sign = -1.0 if (m % 2) else 1.0
                ylm[idx, :] = math.sqrt(2.0)*sign*ycomplex.real

    return ylm
