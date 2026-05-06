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
from pyfock.XC import lda_x, lda_x_cupy
from numba import njit


@njit(cache=True, fastmath=True, error_model="numpy", nogil=True, inline='always')
def pw91_x_temp(rho, sigma):
    """
    Intermediate computation for PW91 exchange enhancement factor.
    Reference: Phys. Rev. B 46, 6671 (1992).
    """
    # PW91 parameters
    a1 = 0.19645
    a2 = 7.7956
    a3 = 0.2743
    a4 = 0.1508
    a5 = 100.0
    a = 0.004

    sigma = np.maximum(sigma, 1e-30)
    norm_dn = np.sqrt(sigma)
    kf = (3 * np.pi**2 * rho) ** (1 / 3)
    divkf = 1.0 / kf
    s = norm_dn * divkf / (2 * rho)

    s2 = s * s
    s3 = s2 * s
    s4 = s2 * s2

    # Enhancement factor Fx(s)
    # Fx = (1 + a1*s*arcsinh(a2*s) + (a3 - a4*exp(-a5*s2))*s2) / (1 + a1*s*arcsinh(a2*s) + a*s4)
    asinh_val = np.arcsinh(a2 * s)
    num = 1 + a1 * s * asinh_val + (a3 - a4 * np.exp(-a5 * s2)) * s2
    den = 1 + a1 * s * asinh_val + a * s4
    Fx = num / den

    exunif = -3 * kf / (4 * np.pi)

    # Derivatives
    # d(arcsinh(a2*s))/ds = a2 / sqrt(1 + a2^2 * s^2)
    dasinh = a2 / np.sqrt(1 + a2**2 * s2)

    dnum_ds = a1 * asinh_val + a1 * s * dasinh + 2 * (a3 - a4 * np.exp(-a5 * s2)) * s + a4 * 2 * a5 * s * np.exp(-a5 * s2) * s2
    # Simplify last term: a4 * 2 * a5 * s^3 * exp(-a5*s2)
    dnum_ds = a1 * asinh_val + a1 * s * dasinh + 2 * a3 * s - 2 * a4 * np.exp(-a5 * s2) * s + 2 * a4 * a5 * s3 * np.exp(-a5 * s2)

    dden_ds = a1 * asinh_val + a1 * s * dasinh + 4 * a * s3

    dFx_ds = (dnum_ds * den - num * dden_ds) / den**2

    # sx = exunif * (Fx - 1)  since Slater exchange is added separately
    sx = exunif * (Fx - 1)

    dsdn = -4.0 / 3.0 * s
    dexunif = exunif / 3.0

    vx = sx + dexunif * (Fx - 1) + exunif * dFx_ds * dsdn

    vsigmax = exunif * dFx_ds * divkf / (2 * norm_dn)

    return sx * rho, vx, vsigmax


@njit(cache=True, fastmath=True, error_model="numpy", nogil=True, inline='always')
def gga_x_pw91(rho, sigma):
    """
    PW91 exchange functional (spin-unpolarized).
    Corresponds to GGA_X_PW91 with ID 109 in LibXC.
    Reference: Phys. Rev. B 46, 6671 (1992).

    Parameters
    ----------
    rho : ndarray
        Electron density.
    sigma : ndarray
        Squared gradient of density (∇ρ·∇ρ).

    Returns
    -------
    ex : ndarray
        Exchange energy density.
    vx : ndarray
        d(n*ex)/d(n).
    vsigma : ndarray
        d(n*ex)/d(sigma).
    """
    rho = np.maximum(rho, 1e-12)

    ex, vx = lda_x(rho)
    gex, gvx, vsigmax = pw91_x_temp(rho, sigma)

    ex += gex / rho
    vx += gvx
    vsigma = 0.5 * vsigmax

    vsigma[np.isnan(vsigma)] = 0
    vx[np.isnan(vx)] = 0
    ex[np.isnan(ex)] = 0

    return ex, vx, vsigma


@fuse(kernel_name='pw91_x_temp_cupy')
def pw91_x_temp_cupy(rho, sigma):
    """
    CuPy version of PW91 exchange intermediate computation.
    """
    a1 = 0.19645
    a2 = 7.7956
    a3 = 0.2743
    a4 = 0.1508
    a5 = 100.0
    a = 0.004

    sigma = cp.maximum(sigma, 1e-30)
    norm_dn = cp.sqrt(sigma)
    kf = (3 * cp.pi**2 * rho) ** (1 / 3)
    divkf = 1.0 / kf
    s = norm_dn * divkf / (2 * rho)

    s2 = s * s
    s3 = s2 * s
    s4 = s2 * s2

    asinh_val = cp.arcsinh(a2 * s)
    exp_val = cp.exp(-a5 * s2)
    num = 1 + a1 * s * asinh_val + (a3 - a4 * exp_val) * s2
    den = 1 + a1 * s * asinh_val + a * s4
    Fx = num / den

    exunif = -3 * kf / (4 * cp.pi)

    dasinh = a2 / cp.sqrt(1 + a2**2 * s2)
    dnum_ds = a1 * asinh_val + a1 * s * dasinh + 2 * a3 * s - 2 * a4 * exp_val * s + 2 * a4 * a5 * s3 * exp_val
    dden_ds = a1 * asinh_val + a1 * s * dasinh + 4 * a * s3
    dFx_ds = (dnum_ds * den - num * dden_ds) / den**2

    sx = exunif * (Fx - 1)

    dsdn = -4.0 / 3.0 * s
    dexunif = exunif / 3.0

    vx = sx + dexunif * (Fx - 1) + exunif * dFx_ds * dsdn
    vsigmax = exunif * dFx_ds * divkf / (2 * norm_dn)

    return sx * rho, vx, vsigmax


def gga_x_pw91_cupy(rho, sigma):
    """
    CuPy version of PW91 exchange functional.
    Corresponds to GGA_X_PW91 with ID 109 in LibXC.
    Reference: Phys. Rev. B 46, 6671 (1992).
    """
    rho = cp.maximum(rho, 1e-12)

    ex, vx = lda_x_cupy(rho)
    gex, gvx, vsigmax = pw91_x_temp_cupy(rho, sigma)

    ex += gex / rho
    vx += gvx
    vsigma = 0.5 * vsigmax

    vsigma[cp.isnan(vsigma)] = 0
    vx[cp.isnan(vx)] = 0
    ex[cp.isnan(ex)] = 0

    return ex, vx, vsigma
