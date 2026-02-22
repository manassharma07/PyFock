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
from pyfock.XC import lda_x
from numba import njit


@njit(cache=True, fastmath=True, error_model="numpy", nogil=True, inline='always')
def rpbe_x_temp(rho, sigma):
    """
    Intermediate computation for RPBE exchange.
    RPBE uses Fx = 1 + kappa * (1 - exp(-mu * s^2 / kappa)) instead of PBE's Fx.
    Reference: Phys. Rev. B 59, 7413 (1999) (Hammer, Hansen, Norskov).
    """
    mu = 0.2195149727645171
    kappa = 0.804

    norm_dn = np.sqrt(sigma)
    kf = (3 * np.pi**2 * rho) ** (1 / 3)
    divkf = 1.0 / kf
    s = norm_dn * divkf / (2 * rho)
    s2 = s ** 2

    exp_val = np.exp(-mu * s2 / kappa)
    Fx = kappa * (1 - exp_val)
    exunif = -3 * kf / (4 * np.pi)
    # Fx here is the enhancement beyond 1, since Slater exchange is added separately
    sx = exunif * Fx

    dsdn = -4.0 / 3.0 * s
    dFxds = 2 * mu * s * exp_val
    dexunif = exunif / 3.0
    exunifdFx = exunif * dFxds
    vx = sx + dexunif * Fx + exunifdFx * dsdn

    vsigmax = exunifdFx * divkf / (2 * norm_dn)

    return sx * rho, vx, vsigmax


@njit(cache=True, fastmath=True, error_model="numpy", nogil=True, inline='always')
def gga_x_rpbe(rho, sigma):
    """
    RPBE (Revised PBE) exchange functional (spin-unpolarized).
    Corresponds to GGA_X_RPBE with ID 117 in LibXC.
    Reference: Phys. Rev. B 59, 7413 (1999).

    Parameters
    ----------
    rho : ndarray
        Electron density.
    sigma : ndarray
        Squared gradient of density.

    Returns
    -------
    ex, vx, vsigma : ndarrays
    """
    rho = np.maximum(rho, 1e-12)

    ex, vx = lda_x(rho)
    gex, gvx, vsigmax = rpbe_x_temp(rho, sigma)

    ex += gex / rho
    vx += gvx
    vsigma = 0.5 * vsigmax

    vsigma[np.isnan(vsigma)] = 0
    vx[np.isnan(vx)] = 0
    ex[np.isnan(ex)] = 0

    return ex, vx, vsigma


@fuse(kernel_name='rpbe_x_temp_cupy')
def rpbe_x_temp_cupy(rho, sigma):
    """
    CuPy version of RPBE exchange intermediate computation.
    """
    mu = 0.2195149727645171
    kappa = 0.804

    norm_dn = cp.sqrt(sigma)
    kf = (3 * cp.pi**2 * rho) ** (1 / 3)
    divkf = 1.0 / kf
    s = norm_dn * divkf / (2 * rho)
    s2 = s ** 2

    exp_val = cp.exp(-mu * s2 / kappa)
    Fx = kappa * (1 - exp_val)
    exunif = -3 * kf / (4 * cp.pi)
    sx = exunif * Fx

    dsdn = -4.0 / 3.0 * s
    dFxds = 2 * mu * s * exp_val
    dexunif = exunif / 3.0
    exunifdFx = exunif * dFxds
    vx = sx + dexunif * Fx + exunifdFx * dsdn

    vsigmax = exunifdFx * divkf / (2 * norm_dn)

    return sx * rho, vx, vsigmax


def gga_x_rpbe_cupy(rho, sigma):
    """
    CuPy version of RPBE exchange functional.
    Corresponds to GGA_X_RPBE with ID 117 in LibXC.
    Reference: Phys. Rev. B 59, 7413 (1999).
    """
    rho = cp.maximum(rho, 1e-12)

    ex, vx = lda_x(rho)
    gex, gvx, vsigmax = rpbe_x_temp_cupy(rho, sigma)

    ex += gex / rho
    vx += gvx
    vsigma = 0.5 * vsigmax

    vsigma[cp.isnan(vsigma)] = 0
    vx[cp.isnan(vx)] = 0
    ex[cp.isnan(ex)] = 0

    return ex, vx, vsigma