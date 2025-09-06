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

# The following implementation of the Vosko-Wilk-Nusair parametrization of the correlation functional
# has been taken from this repository (https://github.com/wangenau/eminus/blob/main/eminus/xc/lda_x.py)
# pretty much as is. The repo has the Apache 2.0 license.

def lda_c_vwn_(rho):
    """
    Compute the correlation energy and potential using the Vosko-Wilk-Nusair (VWN) LDA correlation functional.

    This function implements the spin-unpolarized LDA_C_VWN functional (LibXC ID 7) based on the 
    parametrization by Vosko, Wilk, and Nusair.

    Parameters
    ----------
    rho : ndarray
        Electron density array (assumed to be spin-unpolarized).

    Returns
    -------
    ec : ndarray
        Correlation energy density at each grid point.

    vc : ndarray
        Correlation potential (functional derivative of correlation energy with respect to density).

    Notes
    -----
    This is the raw implementation of the VWN correlation functional adapted from:
        - Phys. Rev. B 22, 3812 (1980)
    Code source: https://github.com/wangenau/eminus/blob/main/eminus/xc/lda_x.py
    Licensed under the Apache License, Version 2.0.

    """

    rho = np.maximum(rho, 1e-12)
    
    a = 0.0310907
    b = 3.72744
    c = 12.9352
    x0 = -0.10498
    pi34 = (3 / (4 * np.pi))**(1 / 3)
    rs = pi34 * np.power(rho, -1 / 3)
    q = np.sqrt(4 * c - b * b)
    f1 = 2 * b / q
    f2 = b * x0 / (x0 * x0 + b * x0 + c)
    f3 = 2 * (2 * x0 + b) / q
    rs12 = np.sqrt(rs)
    fx = rs + b * rs12 + c
    qx = np.arctan(q / (2 * rs12 + b))
    ec = a * (np.log(rs / fx) + f1 * qx - f2 * (np.log((rs12 - x0)**2 / fx) + f3 * qx))
    tx = 2 * rs12 + b
    tt = tx * tx + q * q
    vc = ec - rs12 * a / 6 * (2 / rs12 - tx / fx - 4 * b / tt -
                              f2 * (2 / (rs12 - x0) - tx / fx - 4 * (2 * x0 + b) / tt))
    return ec, vc

def lda_c_vwn(rho):
    """
    Wrapper function for `lda_c_vwn_` providing the LDA correlation energy and potential using VWN parametrization.

    Parameters
    ----------
    rho : ndarray
        Electron density array (spin-unpolarized).

    Returns
    -------
    ec : ndarray
        Correlation energy density.

    vc : ndarray
        Correlation potential.

    Notes
    -----
    This function is equivalent to `lda_c_vwn_` but kept as a public-facing interface in PyFock.
    """
    ec, vc = lda_c_vwn_(rho)
    return ec, vc

@fuse(kernel_name='lda_c_vwn_cupy_')
def lda_c_vwn_cupy_(rho):
    """
    GPU-accelerated implementation of the LDA_C_VWN correlation functional using CuPy.

    This is a CuPy version of the Vosko-Wilk-Nusair LDA correlation functional (LibXC ID 7) for spin-unpolarized densities.

    Parameters
    ----------
    rho : cupy.ndarray
        Electron density array (spin-unpolarized), allocated on GPU.

    Returns
    -------
    ec : cupy.ndarray
        Correlation energy density at each grid point.

    vc : cupy.ndarray
        Correlation potential (functional derivative of correlation energy with respect to density).

    Notes
    -----
    Based on:
        - Vosko, Wilk, and Nusair, Phys. Rev. B 22, 3812 (1980)
    Adapted from: https://github.com/wangenau/eminus/blob/main/eminus/xc/lda_x.py
    Licensed under the Apache License, Version 2.0.

    """

    rho = cp.maximum(rho, 1e-12)
    
    a = 0.0310907
    b = 3.72744
    c = 12.9352
    x0 = -0.10498
    pi34 = (3 / (4 * cp.pi))**(1 / 3)
    rs = pi34 * cp.power(rho, -1 / 3)
    q = cp.sqrt(4 * c - b * b)
    f1 = 2 * b / q
    f2 = b * x0 / (x0 * x0 + b * x0 + c)
    f3 = 2 * (2 * x0 + b) / q
    rs12 = cp.sqrt(rs)
    fx = rs + b * rs12 + c
    qx = cp.arctan(q / (2 * rs12 + b))
    ec = a * (cp.log(rs / fx) + f1 * qx - f2 * (cp.log((rs12 - x0)**2 / fx) + f3 * qx))
    tx = 2 * rs12 + b
    tt = tx * tx + q * q
    vc = ec - rs12 * a / 6 * (2 / rs12 - tx / fx - 4 * b / tt -
                              f2 * (2 / (rs12 - x0) - tx / fx - 4 * (2 * x0 + b) / tt))
    return ec, vc

def lda_c_vwn_cupy(rho):
    """
    Safe wrapper for the GPU-based VWN LDA correlation functional.

    This function calls `lda_c_vwn_cupy_`, and replaces any NaNs in the result with zeros 
    to ensure numerical stability in downstream calculations.

    Parameters
    ----------
    rho : cupy.ndarray
        Electron density array on GPU.

    Returns
    -------
    ec : cupy.ndarray
        Correlation energy density with NaNs replaced by 0.

    vc : cupy.ndarray
        Correlation potential with NaNs replaced by 0.

    Notes
    -----
    This is a numerically safe version of `lda_c_vwn_cupy_` intended for production use.
    """
    ec, vc = lda_c_vwn_cupy_(rho)
    vc[cp.isnan(vc)] = 0
    ec[cp.isnan(ec)] = 0
    return ec, vc