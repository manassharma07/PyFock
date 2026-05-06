import math

import numpy as np
from numba import njit, prange
from .integral_helpers import calcS




def ecp_mat_symm(basis, slice=None, series_order=20, coeff_tol=1.0e-16, power_shift=2):
    """
    Compute scalar semilocal ECP integrals without spatial quadrature.

    The local Gaussian ECP part is evaluated by the Gaussian product theorem.
    The projector part is evaluated analytically by expanding off-center
    Gaussian angular factors in a power series. Each term has a closed-form
    sphere monomial integral and a closed-form radial gamma integral.

    Parameters
    ----------
    basis : Basis
        PyFock basis object with parsed ECP data.
    slice : list, optional
        Matrix slice [row_start, row_end, col_start, col_end].
    series_order : int
        Highest order retained in exp(2 alpha r D.Omega). Centered terms are
        exact at order zero; off-center terms converge with this parameter.
    coeff_tol : float
        Drop moment-series coefficients below this absolute threshold.
    power_shift : int
        Interpret ECP powers as r**(power - power_shift). The def2 files use
        power_shift=2.
    """
    if not getattr(basis, 'has_ecp', False):
        if slice is None:
            return np.zeros((basis.bfs_nao, basis.bfs_nao))
        return np.zeros((int(slice[1]) - int(slice[0]), int(slice[3]) - int(slice[2])))

    arrays = basis_to_arrays(basis)
    ecp_data = ecp_to_arrays(basis, power_shift)

    if np.any(ecp_data['local_quad_counts'] > 0):
        raise NotImplementedError('Analytical ECP local terms currently require power == power_shift.')

    if slice is None:
        slice = [0, basis.bfs_nao, 0, basis.bfs_nao]

    start_row = int(slice[0])
    end_row = int(slice[1])
    start_col = int(slice[2])
    end_col = int(slice[3])

    V = ecp_local_gaussian_mat_symm_internal(arrays['coords'], arrays['contr_norms'], arrays['lmn'], arrays['nprim'], arrays['coeffs'], arrays['prim_norms'], arrays['expnts'], ecp_data['coords'], ecp_data['local_counts'], ecp_data['local_coeffs'], ecp_data['local_expnts'], start_row, end_row, start_col, end_col)

    if len(ecp_data['channels']) == 0:
        return V

    V_projector = _ecp_projector_series_mat(basis, ecp_data, series_order=series_order, coeff_tol=coeff_tol,power_shift=power_shift)
    V = V + V_projector[start_row:end_row, start_col:end_col]
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

def _ecp_projector_series_mat(basis, ecp_data, series_order=20, coeff_tol=1.0e-16, power_shift=2):
    nao = basis.bfs_nao
    V = np.zeros((nao, nao), dtype=np.float64)
    ao_primitives = _ao_primitives(basis)

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

        moment_cache = _build_moment_cache(
            ao_primitives,
            center,
            moment_powers,
            series_order=series_order,
            coeff_tol=coeff_tol,
        )
        term_counts, term_alphas, term_powers, term_coeffs = _moment_cache_to_arrays(moment_cache, moment_powers)
        channel_arrays = _channel_arrays(channels, kernel_by_l, moment_powers)
        gamma_half = _gamma_half_table(_max_radial_power(term_counts, term_powers, channel_arrays[5], power_shift))
        V += _assemble_projector_series_internal(
            term_counts,
            term_alphas,
            term_powers,
            term_coeffs,
            channel_arrays[0],
            channel_arrays[1],
            channel_arrays[2],
            channel_arrays[3],
            channel_arrays[4],
            channel_arrays[5],
            channel_arrays[6],
            gamma_half,
            power_shift,
        )

    return V


def _moment_cache_to_arrays(moment_cache, moment_powers):
    nao = len(moment_cache)
    nmoment = len(moment_powers)
    max_terms = 1
    for i in range(nao):
        for moment_power in moment_powers:
            nterms = 0
            for alpha, coeffs in moment_cache[i].get(moment_power, []):
                nterms += len(coeffs)
            max_terms = max(max_terms, nterms)

    term_counts = np.zeros((nao, nmoment), dtype=np.int64)
    term_alphas = np.zeros((nao, nmoment, max_terms), dtype=np.float64)
    term_powers = np.zeros((nao, nmoment, max_terms), dtype=np.int64)
    term_coeffs = np.zeros((nao, nmoment, max_terms), dtype=np.float64)
    for i in range(nao):
        for imoment, moment_power in enumerate(moment_powers):
            iterm = 0
            for alpha, coeffs in moment_cache[i].get(moment_power, []):
                for r_power, coeff in coeffs:
                    term_alphas[i, imoment, iterm] = alpha
                    term_powers[i, imoment, iterm] = r_power
                    term_coeffs[i, imoment, iterm] = coeff
                    iterm += 1
            term_counts[i, imoment] = iterm
    return term_counts, term_alphas, term_powers, term_coeffs


def _channel_arrays(channels, kernel_by_l, moment_powers):
    moment_index = {power: i for i, power in enumerate(moment_powers)}
    nchannel = len(channels)
    max_kernel = max(max(len(kernel_by_l[channel['l']]) for channel in channels), 1)
    max_terms = max(max(len(channel['terms']) for channel in channels), 1)

    kernel_counts = np.zeros(nchannel, dtype=np.int64)
    kernel_indices = np.zeros((nchannel, max_kernel), dtype=np.int64)
    kernel_coeffs = np.zeros((nchannel, max_kernel), dtype=np.float64)
    term_counts = np.zeros(nchannel, dtype=np.int64)
    term_coeffs = np.zeros((nchannel, max_terms), dtype=np.float64)
    term_powers = np.zeros((nchannel, max_terms), dtype=np.int64)
    term_expnts = np.zeros((nchannel, max_terms), dtype=np.float64)

    for ichannel, channel in enumerate(channels):
        for ikernel, (moment_power, kernel_coeff) in enumerate(kernel_by_l[channel['l']].items()):
            kernel_indices[ichannel, ikernel] = moment_index[moment_power]
            kernel_coeffs[ichannel, ikernel] = kernel_coeff
        kernel_counts[ichannel] = len(kernel_by_l[channel['l']])

        terms = channel['terms']
        term_counts[ichannel] = len(terms)
        for iterm, (coeff, power, exponent) in enumerate(terms):
            term_coeffs[ichannel, iterm] = coeff
            term_powers[ichannel, iterm] = int(power)
            term_expnts[ichannel, iterm] = exponent

    return kernel_counts, kernel_indices, kernel_coeffs, term_counts, term_coeffs, term_powers, term_expnts


def _max_radial_power(term_counts, term_powers, channel_term_powers, power_shift):
    max_power = 0
    for i in range(term_counts.shape[0]):
        for imoment in range(term_counts.shape[1]):
            for iterm in range(term_counts[i, imoment]):
                max_power = max(max_power, int(term_powers[i, imoment, iterm]))
    max_ecp_power = 0
    for power in np.ravel(channel_term_powers):
        max_ecp_power = max(max_ecp_power, int(power))
    return max(0, 2*max_power + max_ecp_power - power_shift + 2)


def _gamma_half_table(max_power):
    gamma_half = np.zeros(max_power + 1, dtype=np.float64)
    for power in range(max_power + 1):
        gamma_half[power] = 0.5*math.gamma(0.5*(power + 1))
    return gamma_half


def _ao_primitives(basis):
    ao_primitives = []
    for i in range(basis.bfs_nao):
        prims = []
        lmn = tuple(int(x) for x in basis.bfs_lmn[i])
        center = np.array(basis.bfs_coords[i], dtype=np.float64)
        contr_norm = basis.bfs_contr_prim_norms[i]
        for k in range(basis.bfs_nprim[i]):
            coeff = basis.bfs_coeffs[i][k]*basis.bfs_prim_norms[i][k]*contr_norm
            alpha = basis.bfs_expnts[i][k]
            prims.append((alpha, coeff, lmn, center))
        ao_primitives.append(prims)
    return ao_primitives


def _build_moment_cache(ao_primitives, center, moment_powers, series_order=20, coeff_tol=1.0e-16):
    moment_cache = []
    for prims in ao_primitives:
        ao_cache = {}
        for moment_power in moment_powers:
            entries = []
            for alpha, coeff, lmn, ao_center in prims:
                coeffs = _primitive_moment_coeffs(
                    alpha,
                    coeff,
                    lmn,
                    ao_center,
                    center,
                    moment_power,
                    series_order,
                    coeff_tol,
                )
                if len(coeffs) > 0:
                    entries.append((alpha, coeffs))
            if len(entries) > 0:
                ao_cache[moment_power] = entries
        moment_cache.append(ao_cache)
    return moment_cache


def _primitive_moment_coeffs(alpha, prim_coeff, lmn, ao_center, ecp_center, moment_power, series_order, coeff_tol):
    D = ao_center - ecp_center
    d2 = float(np.dot(D, D))
    gaussian_shift = math.exp(-alpha*d2)
    if gaussian_shift == 0.0:
        return []

    coeffs = {}
    poly_terms = _cartesian_polynomial_terms(lmn, D)
    prefactor = prim_coeff*gaussian_shift

    for poly_power, omega_power, poly_coeff in poly_terms:
        for tx in range(series_order + 1):
            for ty in range(series_order + 1 - tx):
                for tz in range(series_order + 1 - tx - ty):
                    t = tx + ty + tz
                    exp_coeff = _expansion_coeff(alpha, D, tx, ty, tz)
                    if exp_coeff == 0.0:
                        continue
                    ax = moment_power[0] + omega_power[0] + tx
                    ay = moment_power[1] + omega_power[1] + ty
                    az = moment_power[2] + omega_power[2] + tz
                    angular = _sphere_monomial_integral(ax, ay, az)
                    if angular == 0.0:
                        continue
                    r_power = poly_power + t
                    value = prefactor*poly_coeff*exp_coeff*angular
                    if abs(value) > coeff_tol:
                        coeffs[r_power] = coeffs.get(r_power, 0.0) + value

    return [(power, coeff) for power, coeff in sorted(coeffs.items()) if abs(coeff) > coeff_tol]


def _cartesian_polynomial_terms(lmn, D):
    lx, ly, lz = lmn
    terms = []
    for ax in range(lx + 1):
        cx = math.comb(lx, ax)*((-D[0])**(lx - ax))
        for ay in range(ly + 1):
            cy = math.comb(ly, ay)*((-D[1])**(ly - ay))
            for az in range(lz + 1):
                cz = math.comb(lz, az)*((-D[2])**(lz - az))
                coeff = cx*cy*cz
                if coeff != 0.0:
                    terms.append((ax + ay + az, (ax, ay, az), coeff))
    return terms


def _expansion_coeff(alpha, D, tx, ty, tz):
    t = tx + ty + tz
    coeff = (2.0*alpha)**t
    if tx > 0:
        coeff *= D[0]**tx
    if ty > 0:
        coeff *= D[1]**ty
    if tz > 0:
        coeff *= D[2]**tz
    coeff /= math.factorial(tx)*math.factorial(ty)*math.factorial(tz)
    return coeff


def _sphere_monomial_integral(ax, ay, az):
    if (ax % 2) or (ay % 2) or (az % 2):
        return 0.0
    nx = ax//2
    ny = ay//2
    nz = az//2
    numerator = _odd_double_factorial(2*nx - 1)
    numerator *= _odd_double_factorial(2*ny - 1)
    numerator *= _odd_double_factorial(2*nz - 1)
    denominator = _odd_double_factorial(2*(nx + ny + nz) + 1)
    return 4.0*math.pi*numerator/denominator


def _odd_double_factorial(n):
    if n <= 0:
        return 1.0
    value = 1.0
    for k in range(n, 0, -2):
        value *= k
    return value


def _projector_kernel_terms(l):
    legendre_coeffs = _legendre_coeffs(l)
    prefactor = (2*l + 1)/(4.0*math.pi)
    terms = {}
    for k, coeff in legendre_coeffs.items():
        for ax in range(k + 1):
            for ay in range(k + 1 - ax):
                az = k - ax - ay
                multinomial = math.factorial(k)/(math.factorial(ax)*math.factorial(ay)*math.factorial(az))
                power = (ax, ay, az)
                terms[power] = terms.get(power, 0.0) + prefactor*coeff*multinomial
    return {power: coeff for power, coeff in terms.items() if abs(coeff) > 0.0}


def _legendre_coeffs(l):
    if l == 0:
        return {0: 1.0}
    if l == 1:
        return {1: 1.0}
    if l == 2:
        return {2: 1.5, 0: -0.5}
    if l == 3:
        return {3: 2.5, 1: -1.5}
    if l == 4:
        return {4: 35.0/8.0, 2: -30.0/8.0, 0: 3.0/8.0}
    if l == 5:
        return {5: 63.0/8.0, 3: -70.0/8.0, 1: 15.0/8.0}
    raise NotImplementedError('Analytical ECP projectors are implemented up to l=5.')


def _moment_pair_ecp_integral(entries_i, entries_j, ecp_terms, power_shift):
    value = 0.0
    for alpha_i, coeffs_i in entries_i:
        for alpha_j, coeffs_j in entries_j:
            for ecp_coeff, ecp_power, ecp_expnt in ecp_terms:
                p = alpha_i + alpha_j + ecp_expnt
                radial_power_shifted = int(ecp_power) - power_shift + 2
                for power_i, coeff_i in coeffs_i:
                    for power_j, coeff_j in coeffs_j:
                        radial_power = power_i + power_j + radial_power_shifted
                        value += ecp_coeff*coeff_i*coeff_j*_radial_gamma_integral(radial_power, p)
    return value


def _radial_gamma_integral(power, exponent):
    return 0.5*math.gamma(0.5*(power + 1))*exponent**(-0.5*(power + 1))


@njit(parallel=True, cache=True, fastmath=True, error_model='numpy')
def _assemble_projector_series_internal(
    term_counts,
    term_alphas,
    term_powers,
    term_coeffs,
    kernel_counts,
    kernel_indices,
    kernel_coeffs,
    channel_term_counts,
    channel_term_coeffs,
    channel_term_powers,
    channel_term_expnts,
    gamma_half,
    power_shift,
):
    nao = term_counts.shape[0]
    nchannel = channel_term_counts.shape[0]
    V = np.zeros((nao, nao), dtype=np.float64)

    for i in prange(nao):
        for j in range(i + 1):
            value = 0.0
            for ichannel in range(nchannel):
                for ikernel in range(kernel_counts[ichannel]):
                    imoment = kernel_indices[ichannel, ikernel]
                    ni = term_counts[i, imoment]
                    nj = term_counts[j, imoment]
                    if ni == 0 or nj == 0:
                        continue

                    moment_value = 0.0
                    for ti in range(ni):
                        alpha_i = term_alphas[i, imoment, ti]
                        power_i = term_powers[i, imoment, ti]
                        coeff_i = term_coeffs[i, imoment, ti]
                        for tj in range(nj):
                            alpha_j = term_alphas[j, imoment, tj]
                            power_j = term_powers[j, imoment, tj]
                            coeff_ij = coeff_i*term_coeffs[j, imoment, tj]
                            alpha_ij = alpha_i + alpha_j
                            for iterm in range(channel_term_counts[ichannel]):
                                radial_power = power_i + power_j + channel_term_powers[ichannel, iterm] - power_shift + 2
                                exponent = alpha_ij + channel_term_expnts[ichannel, iterm]
                                radial = gamma_half[radial_power]*exponent**(-0.5*(radial_power + 1))
                                moment_value += channel_term_coeffs[ichannel, iterm]*coeff_ij*radial

                    value += kernel_coeffs[ichannel, ikernel]*moment_value

            V[i, j] = value
            if i != j:
                V[j, i] = value

    return V
