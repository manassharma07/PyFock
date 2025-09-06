try:
    import cupy as cp
    from cupy import fuse
except Exception as e:
    # Handle the case when Cupy is not installed
    cp = None
    # Define a dummy fuse decorator for CPU version
    def fuse(kernel_name):
        def decorator(func):
            return func 
        return decorator
import numpy as np
from pyfock.XC import lda_x

# The following implementation of the PBE exchange has been taken from this repository (https://github.com/wangenau/eminus/blob/main/eminus/xc/)
# pretty much as is. The repo has the Apache 2.0 license.

def gga_x_pbe(rho, sigma):
    """
    Compute the restricted PBE (Perdew–Burke–Ernzerhof) exchange energy density and potential
    using NumPy arrays for electron density and its gradient.

    Adapted from
    ----------
    Eminus project:
    https://github.com/wangenau/eminus/blob/main/eminus/xc/gga_x_pbe.py
    Licensed under the Apache License, Version 2.0.

    Reference
    ----------
    J. P. Perdew, K. Burke, and M. Ernzerhof,
    "Generalized Gradient Approximation Made Simple",
    Phys. Rev. Lett. 77, 3865 (1996).
    https://doi.org/10.1103/PhysRevLett.77.3865

    Parameters
    ----------
    rho : ndarray
        Electron density array.
    sigma : ndarray
        Gradient of the electron density, defined as ∇ρ·∇ρ.

    Returns
    -------
    ex : ndarray
        Exchange energy density.
    vx : ndarray
        Functional derivative of the exchange energy with respect to density.
    vsigma : ndarray
        Functional derivative of the exchange energy with respect to the density gradient term σ.
    """

    mu = 0.2195149727645171 # Functional parameter

    # rho_cutoff = 1e-12  # define rho_cutoff constant
    rho = np.maximum(rho, 1e-12)

    ex, vx = lda_x(rho)
    gex, gvx, vsigmax = pbe_x_temp(rho, sigma)

    ex += gex/rho
    vx += gvx
    vsigma = 0.5*vsigmax

    vsigma[np.isnan(vsigma)] = 0
    vx[np.isnan(vx)] = 0
    ex[np.isnan(ex)] = 0

    return ex, vx, vsigma

def pbe_x_temp(rho, sigma):
    """
    Intermediate computation for PBE exchange: enhancement factor, 
    exchange potential and derivative with respect to σ.

    Adapted from
    ----------
    Eminus project:
    https://github.com/wangenau/eminus/blob/main/eminus/xc/gga_x_pbe.py
    Licensed under the Apache License, Version 2.0.

    Reference
    ----------
    J. P. Perdew, K. Burke, and M. Ernzerhof,
    "Generalized Gradient Approximation Made Simple",
    Phys. Rev. Lett. 77, 3865 (1996).
    https://doi.org/10.1103/PhysRevLett.77.3865

    Parameters
    ----------
    rho : ndarray
        Electron density array.
    sigma : ndarray
        Gradient of the electron density, defined as ∇ρ·∇ρ.

    Returns
    -------
    gex : ndarray
        Gradient correction to the exchange energy density.
    gvx : ndarray
        Correction to the exchange potential (derivative with respect to density).
    vsigmax : ndarray
        Derivative of the exchange energy with respect to σ.
    """


    mu = 0.2195149727645171
    kappa = 0.804

    norm_dn = np.sqrt(sigma)
    kf = (3 * np.pi**2 * rho)**(1 / 3)
    # Handle divisions by zero
    divkf = 1 / kf
    # divkf = np.divide(1, kf,
    #                   out=np.zeros_like(kf), where=(kf > 0))
    # Handle divisions by zero
    s = norm_dn * divkf / (2 * rho)
    # s = np.divide(norm_dn * divkf, 2 * rho,
    #               out=np.zeros_like(rho), where=(rho > 0))
    f1 = 1 + mu * s**2 / kappa
    Fx = kappa - kappa / f1
    exunif = -3 * kf / (4 * np.pi)
    # In Fx a '1 + ' is missing, since n * exunif is the Slater exchange that is added later
    sx = exunif * Fx

    dsdn = -4 / 3 * s
    dFxds = 2 * mu * s / f1**2
    dexunif = exunif / 3
    exunifdFx = exunif * dFxds
    vx = sx + dexunif * Fx + exunifdFx * dsdn  # dFx/dn = dFx/ds * ds/dn

    # Handle divisions by zero
    vsigmax = exunifdFx * divkf / (2 * norm_dn)
    # vsigmax = np.divide(exunifdFx * divkf, 2 * norm_dn,
    #                     out=np.zeros_like(norm_dn), where=(norm_dn > 0))
    return sx * rho, np.array([vx]), vsigmax

@fuse(kernel_name='pbe_x_temp_cupy')
def pbe_x_temp_cupy(rho, sigma):
    """
    CuPy-accelerated version of `pbe_x_temp`. Computes the enhancement factor,
    exchange potential, and σ-derivative for the PBE exchange functional.

    Adapted from
    ----------
    Eminus project:
    https://github.com/wangenau/eminus/blob/main/eminus/xc/gga_x_pbe.py
    Licensed under the Apache License, Version 2.0.

    Reference
    ----------
    J. P. Perdew, K. Burke, and M. Ernzerhof,
    "Generalized Gradient Approximation Made Simple",
    Phys. Rev. Lett. 77, 3865 (1996).
    https://doi.org/10.1103/PhysRevLett.77.3865

    Parameters
    ----------
    rho : cp.ndarray
        Electron density array (CuPy).
    sigma : cp.ndarray
        Gradient of the electron density, defined as ∇ρ·∇ρ (CuPy).

    Returns
    -------
    gex : cp.ndarray
        Gradient correction to the exchange energy density.
    gvx : cp.ndarray
        Correction to the exchange potential (derivative with respect to density).
    vsigmax : cp.ndarray
        Derivative of the exchange energy with respect to σ.
    """


    mu = 0.2195149727645171
    kappa = 0.804

    norm_dn = cp.sqrt(sigma)
    kf = (3 * cp.pi**2 * rho)**(1 / 3)
    divkf = 1 / kf
    s = norm_dn * divkf / (2 * rho)
    f1 = 1 + mu * s**2 / kappa
    Fx = kappa - kappa / f1
    exunif = -3 * kf / (4 * cp.pi)
    # In Fx a '1 + ' is missing, since n * exunif is the Slater exchange that is added later
    sx = exunif * Fx

    dsdn = -4 / 3 * s
    dFxds = 2 * mu * s / f1**2
    dexunif = exunif / 3
    exunifdFx = exunif * dFxds
    vx = sx + dexunif * Fx + exunifdFx * dsdn  # dFx/dn = dFx/ds * ds/dn

    vsigmax = exunifdFx * divkf / (2 * norm_dn)

    return sx * rho, vx, vsigmax

# @fuse(kernel_name='gga_x_pbe_cupy')
def gga_x_pbe_cupy(rho, sigma):
    """
    Compute the restricted PBE exchange energy density and potential
    using CuPy for GPU acceleration.

    Adapted from
    ----------
    Eminus project:
    https://github.com/wangenau/eminus/blob/main/eminus/xc/gga_x_pbe.py
    Licensed under the Apache License, Version 2.0.

    Reference
    ----------
    J. P. Perdew, K. Burke, and M. Ernzerhof,
    "Generalized Gradient Approximation Made Simple",
    Phys. Rev. Lett. 77, 3865 (1996).
    https://doi.org/10.1103/PhysRevLett.77.3865

    Parameters
    ----------
    rho : cp.ndarray
        Electron density array (CuPy).
    sigma : cp.ndarray
        Gradient of the electron density, defined as ∇ρ·∇ρ (CuPy).

    Returns
    -------
    ex : cp.ndarray
        Exchange energy density.
    vx : cp.ndarray
        Functional derivative of the exchange energy with respect to density.
    vsigma : cp.ndarray
        Functional derivative of the exchange energy with respect to the density gradient term σ.
    """

    mu = 0.2195149727645171 # Functional parameter

    # rho_cutoff = 1e-12  # define rho_cutoff constant
    rho = cp.maximum(rho, 1e-12)

    ex, vx = lda_x(rho)
    gex, gvx, vsigmax = pbe_x_temp_cupy(rho, sigma)

    ex += gex/rho
    vx += gvx
    vsigma = 0.5*vsigmax

    vsigma[cp.isnan(vsigma)] = 0
    vx[cp.isnan(vx)] = 0
    ex[cp.isnan(ex)] = 0

    return ex, vx, vsigma