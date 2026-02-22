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
def lda_c_pz_(rho):
    """
    Perdew-Zunger parametrization of Ceperley-Alder correlation (spin-unpolarized).
    Corresponds to LDA_C_PZ with ID 9 in LibXC.
    Reference: Phys. Rev. B 23, 5048 (1981).
    """
    rho = np.maximum(rho, 1e-12)

    # Parameters for the unpolarized case
    # rs >= 1
    gamma = -0.1423
    beta1 = 1.0529
    beta2 = 0.3334
    # rs < 1
    A = 0.0311
    B = -0.048
    C = 0.002
    D = -0.0116

    rs = (3 / (4 * np.pi * rho)) ** (1 / 3)

    ec = np.empty_like(rho)
    vc = np.empty_like(rho)

    for i in range(len(rho)):
        r = rs[i]
        if r >= 1.0:
            sqrt_rs = np.sqrt(r)
            denom = 1 + beta1 * sqrt_rs + beta2 * r
            ec[i] = gamma / denom
            vc[i] = ec[i] * (1 + (7.0 / 6.0) * beta1 * sqrt_rs + (4.0 / 3.0) * beta2 * r) / denom
        else:
            log_rs = np.log(r)
            ec[i] = A * log_rs + B + C * r * log_rs + D * r
            vc[i] = A * log_rs + (B - A / 3.0) + (2.0 / 3.0) * C * r * log_rs + (2 * D - C) / 3.0 * r

    return ec, vc


@njit(cache=True, fastmath=True, error_model="numpy", nogil=True, inline='always')
def lda_c_pz(rho):
    """
    Perdew-Zunger correlation functional with NaN handling.
    Corresponds to LDA_C_PZ with ID 9 in LibXC.
    Reference: Phys. Rev. B 23, 5048 (1981).
    """
    ec, vc = lda_c_pz_(rho)
    vc[np.isnan(vc)] = 0
    ec[np.isnan(ec)] = 0
    return ec, vc


def lda_c_pz_cupy_(rho):
    """
    CuPy version of Perdew-Zunger correlation.
    Corresponds to LDA_C_PZ with ID 9 in LibXC.
    Reference: Phys. Rev. B 23, 5048 (1981).
    """
    rho = cp.maximum(rho, 1e-12)

    gamma = -0.1423
    beta1 = 1.0529
    beta2 = 0.3334
    A = 0.0311
    B = -0.048
    C = 0.002
    D = -0.0116

    rs = (3 / (4 * cp.pi * rho)) ** (1 / 3)

    sqrt_rs = cp.sqrt(rs)
    log_rs = cp.log(rs)

    # rs >= 1 branch
    denom = 1 + beta1 * sqrt_rs + beta2 * rs
    ec_high = gamma / denom
    vc_high = ec_high * (1 + (7.0 / 6.0) * beta1 * sqrt_rs + (4.0 / 3.0) * beta2 * rs) / denom

    # rs < 1 branch
    ec_low = A * log_rs + B + C * rs * log_rs + D * rs
    vc_low = A * log_rs + (B - A / 3.0) + (2.0 / 3.0) * C * rs * log_rs + (2 * D - C) / 3.0 * rs

    mask = rs >= 1.0
    ec = cp.where(mask, ec_high, ec_low)
    vc = cp.where(mask, vc_high, vc_low)

    return ec, vc


def lda_c_pz_cupy(rho):
    """
    CuPy version of Perdew-Zunger correlation with NaN handling.
    Corresponds to LDA_C_PZ with ID 9 in LibXC.
    Reference: Phys. Rev. B 23, 5048 (1981).
    """
    ec, vc = lda_c_pz_cupy_(rho)
    vc[cp.isnan(vc)] = 0
    ec[cp.isnan(ec)] = 0
    return ec, vc