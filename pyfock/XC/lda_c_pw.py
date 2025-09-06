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

# The following implementation of the Perdew-Wang parametrization of the correlation functional
# has been taken from this repository (https://github.com/wangenau/eminus/blob/main/eminus/xc/lda_c_pw.py)
# pretty much as is. The repo has the Apache 2.0 license.

def lda_c_pw_(rho):
    """
    Compute the LDA correlation energy and potential using the Perdew-Wang (PW92) parametrization.

    This is the spin-unpolarized version of the LDA_C_PW correlation functional (LibXC ID 12).

    Parameters
    ----------
    rho : ndarray
        Electron density array (assumed to be spin-unpolarized). Should be non-negative.

    Returns
    -------
    ec : ndarray
        Correlation energy density at each grid point.

    vc : ndarray
        Correlation potential (functional derivative of correlation energy with respect to density).

    Notes
    -----
    Reference:
        - J.P. Perdew and Y. Wang, Phys. Rev. B 45, 13244 (1992).

    Implementation adapted from:
        https://github.com/wangenau/eminus/blob/main/eminus/xc/lda_c_pw.py
        Licensed under the Apache License, Version 2.0.

    """

    rho = np.maximum(rho, 1e-12)

    A = 0.031091 
    a1 = 0.2137 
    b1 = 7.5957 
    b2 = 3.5876 
    b3 = 1.6382 
    b4 = 0.49294

    rs = (3 / (4 * np.pi * rho))**(1 / 3)
    rs12 = np.sqrt(rs)
    rs32 = rs * rs12
    rs2 = rs**2

    om = 2 * A * (b1 * rs12 + b2 * rs + b3 * rs32 + b4 * rs2)
    olog = np.log(1 + 1 / om)
    ec = -2 * A * (1 + a1 * rs) * olog

    dom = 2 * A * (0.5 * b1 * rs12 + b2 * rs + 1.5 * b3 * rs32 + 2 * b4 * rs2)
    vc = -2 * A * (1 + 2 / 3 * a1 * rs) * olog - 2 / 3 * A * (1 + a1 * rs) * dom / (om * (om + 1))

    return ec, vc

def lda_c_pw(rho):
    """
    Numerically stable wrapper for the Perdew-Wang (PW92) LDA correlation functional.

    This calls `lda_c_pw_`, then replaces any NaNs in the result with zeros to ensure 
    downstream stability in SCF or post-processing steps.

    Parameters
    ----------
    rho : ndarray
        Electron density array (non-negative, spin-unpolarized).

    Returns
    -------
    ec : ndarray
        Correlation energy density with NaNs replaced by 0.

    vc : ndarray
        Correlation potential with NaNs replaced by 0.

    Notes
    -----
    This is the CPU-based version. Use `lda_c_pw_cupy` for GPU acceleration.
    """
    ec, vc = lda_c_pw_(rho)
    vc[np.isnan(vc)] = 0
    ec[np.isnan(ec)] = 0
    return ec, vc


@fuse(kernel_name='lda_c_pw_cupy_')
def lda_c_pw_cupy_(rho):
    """
    GPU-accelerated implementation of the LDA_C_PW correlation functional using CuPy.

    This function computes the Perdew-Wang LDA correlation energy and potential for spin-unpolarized systems.

    Parameters
    ----------
    rho : cupy.ndarray
        Electron density array on GPU (non-negative, spin-unpolarized).

    Returns
    -------
    ec : cupy.ndarray
        Correlation energy density at each grid point.

    vc : cupy.ndarray
        Correlation potential (functional derivative of correlation energy with respect to density).

    Notes
    -----
    Reference:
        - J.P. Perdew and Y. Wang, Phys. Rev. B 45, 13244 (1992).

    Implementation adapted from:
        https://github.com/wangenau/eminus/blob/main/eminus/xc/lda_c_pw.py
        Licensed under the Apache License, Version 2.0.

    This version uses CuPy's `@fuse` decorator to enable kernel fusion for performance optimization.

    """

    rho = cp.maximum(rho, 1e-12)

    A = 0.031091 
    a1 = 0.2137 
    b1 = 7.5957 
    b2 = 3.5876 
    b3 = 1.6382 
    b4 = 0.49294

    rs = (3 / (4 * cp.pi * rho))**(1 / 3)
    rs12 = cp.sqrt(rs)
    rs32 = rs * rs12
    rs2 = rs**2

    om = 2 * A * (b1 * rs12 + b2 * rs + b3 * rs32 + b4 * rs2)
    olog = cp.log(1 + 1 / om)
    ec = -2 * A * (1 + a1 * rs) * olog

    dom = 2 * A * (0.5 * b1 * rs12 + b2 * rs + 1.5 * b3 * rs32 + 2 * b4 * rs2)
    vc = -2 * A * (1 + 2 / 3 * a1 * rs) * olog - 2 / 3 * A * (1 + a1 * rs) * dom / (om * (om + 1))

    return ec, vc

def lda_c_pw_cupy(rho):
    """
    Numerically stable GPU wrapper for the Perdew-Wang LDA correlation functional.

    This function calls `lda_c_pw_cupy_` and replaces any NaNs in the resulting arrays
    with zeros to ensure stability during molecular dynamics or SCF procedures.

    Parameters
    ----------
    rho : cupy.ndarray
        Electron density array on GPU (non-negative, spin-unpolarized).

    Returns
    -------
    ec : cupy.ndarray
        Correlation energy density with NaNs replaced by 0.

    vc : cupy.ndarray
        Correlation potential with NaNs replaced by 0.

    Notes
    -----
    Use this function for production GPU workflows where numerical robustness is critical.
    """
    ec, vc = lda_c_pw_cupy_(rho)
    vc[cp.isnan(vc)] = 0
    ec[cp.isnan(ec)] = 0
    return ec, vc

