try:
    import cupy as cp
    from cupy import fuse
except Exception as e:
    cp = None
    def fuse(kernel_name):
        def decorator(func):
            return func
        return decorator
import numpy as np
from numba import njit


@njit(cache=True, fastmath=True, error_model="numpy", nogil=True, inline='always')
def _pz_rs_geq1(rs, gamma, beta1, beta2):
    """
    Compute ec and dec/drs for rs >= 1 using Eq. (C3) and (C4).
    """
    sqrt_rs = np.sqrt(rs)
    denom = 1.0 + beta1 * sqrt_rs + beta2 * rs
    ec = gamma / denom
    dec_drs = -gamma * (0.5 * beta1 / sqrt_rs + beta2) / (denom * denom)
    return ec, dec_drs


@njit(cache=True, fastmath=True, error_model="numpy", nogil=True, inline='always')
def lda_c_pz_mod_(rho):
    """
    Modified Perdew-Zunger parametrization of Ceperley-Alder data (spin-unpolarized).
    Corresponds to LDA_C_PZ_MOD with ID 10 in LibXC.

    Reference: J. P. Perdew and A. Zunger, Phys. Rev. B 23, 5048 (1981).

    The MOD version re-derives C and D in the rs < 1 region by enforcing
    continuity of BOTH ec and dec/drs at rs = 1 (the original PZ only
    enforces continuity of ec at rs = 1, leading to a kink in the potential).

    Parameters
    ----------
    rho : ndarray
        Electron density array (spin-unpolarized).

    Returns
    -------
    ec : ndarray
        Correlation energy density per particle.
    vc : ndarray
        Correlation potential.
    """
    rho = np.maximum(rho, 1e-12)

    gamma = -0.1423
    beta1 = 1.0529
    beta2 = 0.3334

    A = 0.0311
    B = -0.048

    ec_1, dec_drs_1 = _pz_rs_geq1(1.0, gamma, beta1, beta2)
    D = ec_1 - B
    C = dec_drs_1 - A - D

    rs = (3 / (4 * np.pi * rho)) ** (1 / 3)

    ec = np.empty_like(rho)
    vc = np.empty_like(rho)

    for i in range(len(rho)):
        r = rs[i]
        if r >= 1.0:
            sqrt_rs = np.sqrt(r)
            denom = 1.0 + beta1 * sqrt_rs + beta2 * r
            ec[i] = gamma / denom
            vc[i] = ec[i] * (1.0 + (7.0 / 6.0) * beta1 * sqrt_rs + (4.0 / 3.0) * beta2 * r) / denom
        else:
            log_rs = np.log(r)
            ec[i] = A * log_rs + B + C * r * log_rs + D * r
            dec_drs = A / r + C * log_rs + C + D
            vc[i] = ec[i] - r / 3.0 * dec_drs

    return ec, vc


@njit(cache=True, fastmath=True, error_model="numpy", nogil=True, inline='always')
def lda_c_pz_mod(rho):
    """
    Modified Perdew-Zunger correlation with NaN handling.
    Corresponds to LDA_C_PZ_MOD with ID 10 in LibXC.
    """
    ec, vc = lda_c_pz_mod_(rho)
    vc[np.isnan(vc)] = 0
    ec[np.isnan(ec)] = 0
    return ec, vc


def lda_c_pz_mod_cupy_(rho):
    """
    CuPy version of modified Perdew-Zunger correlation.
    Corresponds to LDA_C_PZ_MOD with ID 10 in LibXC.
    Reference: Phys. Rev. B 23, 5048 (1981).
    
    Re-derives C and D by enforcing continuity of ec and dec/drs at rs = 1.
    """
    rho = cp.maximum(rho, 1e-12)

    gamma = -0.1423
    beta1 = 1.0529
    beta2 = 0.3334
    A = 0.0311
    B = -0.048

    denom_1 = 1.0 + beta1 + beta2
    ec_1 = gamma / denom_1
    dec_drs_1 = -gamma * (0.5 * beta1 + beta2) / (denom_1 * denom_1)
    D = ec_1 - B
    C = dec_drs_1 - A - D

    rs = (3 / (4 * cp.pi * rho)) ** (1 / 3)

    sqrt_rs = cp.sqrt(rs)
    log_rs = cp.log(rs)


    denom = 1.0 + beta1 * sqrt_rs + beta2 * rs
    ec_high = gamma / denom
    vc_high = ec_high * (1.0 + (7.0 / 6.0) * beta1 * sqrt_rs + (4.0 / 3.0) * beta2 * rs) / denom

    ec_low = A * log_rs + B + C * rs * log_rs + D * rs
    dec_drs_low = A / rs + C * log_rs + C + D
    vc_low = ec_low - rs / 3.0 * dec_drs_low

    mask = rs >= 1.0
    ec = cp.where(mask, ec_high, ec_low)
    vc = cp.where(mask, vc_high, vc_low)

    return ec, vc


def lda_c_pz_mod_cupy(rho):
    """
    CuPy version of modified Perdew-Zunger correlation with NaN handling.
    Corresponds to LDA_C_PZ_MOD with ID 10 in LibXC.
    """
    ec, vc = lda_c_pz_mod_cupy_(rho)
    vc[cp.isnan(vc)] = 0
    ec[cp.isnan(ec)] = 0
    return ec, vc