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
def pbesol_x_temp(rho, sigma):
    """
    Intermediate computation for PBEsol exchange.
    Same as PBE exchange but with mu = 10/81 instead of mu_PBE.
    Reference: Phys. Rev. Lett. 100, 136406 (2008).
    """
    mu = 10.0 / 81.0  # PBEsol value
    kappa = 0.804

    norm_dn = np.sqrt(sigma)
    kf = (3 * np.pi**2 * rho) ** (1 / 3)
    divkf = 1.0 / kf
    s = norm_dn * divkf / (2 * rho)

    f1 = 1 + mu * s**2 / kappa
    Fx = kappa - kappa / f1
    exunif = -3 * kf / (4 * np.pi)
    sx = exunif * Fx

    dsdn = -4.0 / 3.0 * s
    dFxds = 2 * mu * s / f1**2
    dexunif = exunif / 3.0
    exunifdFx = exunif * dFxds
    vx = sx + dexunif * Fx + exunifdFx * dsdn

    vsigmax = exunifdFx * divkf / (2 * norm_dn)

    return sx * rho, vx, vsigmax


@njit(cache=True, fastmath=True, error_model="numpy", nogil=True, inline='always')
def gga_x_pbe_sol(rho, sigma):
    """
    PBEsol exchange functional (spin-unpolarized).
    Corresponds to GGA_X_PBE_SOL with ID 116 in LibXC.
    Reference: Phys. Rev. Lett. 100, 136406 (2008).

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
    gex, gvx, vsigmax = pbesol_x_temp(rho, sigma)

    ex += gex / rho
    vx += gvx
    vsigma = 0.5 * vsigmax

    vsigma[np.isnan(vsigma)] = 0
    vx[np.isnan(vx)] = 0
    ex[np.isnan(ex)] = 0

    return ex, vx, vsigma


@fuse(kernel_name='pbesol_x_temp_cupy')
def pbesol_x_temp_cupy(rho, sigma):
    """
    CuPy version of PBEsol exchange intermediate computation.
    """
    mu = 10.0 / 81.0
    kappa = 0.804

    norm_dn = cp.sqrt(sigma)
    kf = (3 * cp.pi**2 * rho) ** (1 / 3)
    divkf = 1.0 / kf
    s = norm_dn * divkf / (2 * rho)

    f1 = 1 + mu * s**2 / kappa
    Fx = kappa - kappa / f1
    exunif = -3 * kf / (4 * cp.pi)
    sx = exunif * Fx

    dsdn = -4.0 / 3.0 * s
    dFxds = 2 * mu * s / f1**2
    dexunif = exunif / 3.0
    exunifdFx = exunif * dFxds
    vx = sx + dexunif * Fx + exunifdFx * dsdn

    vsigmax = exunifdFx * divkf / (2 * norm_dn)

    return sx * rho, vx, vsigmax


def gga_x_pbe_sol_cupy(rho, sigma):
    """
    CuPy version of PBEsol exchange functional.
    Corresponds to GGA_X_PBE_SOL with ID 116 in LibXC.
    Reference: Phys. Rev. Lett. 100, 136406 (2008).
    """
    rho = cp.maximum(rho, 1e-12)

    ex, vx = lda_x(rho)
    gex, gvx, vsigmax = pbesol_x_temp_cupy(rho, sigma)

    ex += gex / rho
    vx += gvx
    vsigma = 0.5 * vsigmax

    vsigma[cp.isnan(vsigma)] = 0
    vx[cp.isnan(vx)] = 0
    ex[cp.isnan(ex)] = 0

    return ex, vx, vsigma