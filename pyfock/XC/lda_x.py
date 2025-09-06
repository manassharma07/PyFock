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

# The following implementation of the Slater exchange has been taken from this repository (https://github.com/wangenau/eminus/blob/main/eminus/xc/lda_x.py)
# pretty much as is. The repo has the Apache 2.0 license.

def lda_x(rho):
    """
    Compute the LDA exchange energy and potential using the Slater exchange functional (spin-unpolarized).

    This is the standard Local Density Approximation (LDA) exchange functional corresponding to 
    `LDA_X` with ID 1 in the LibXC functional library.

    Parameters
    ----------
    rho : ndarray
        Electron density array (assumed to be spin-paired / spin-unpolarized).

    Returns
    -------
    ex : ndarray
        Exchange energy density at each point.

    vx : ndarray
        Exchange potential (functional derivative of exchange energy with respect to density).

    Notes
    -----
    This implementation is based on:
        - Phys. Rev. 81, 385 (1951) — the original Slater exchange.
    Code adapted from the `eminus` repository:
        https://github.com/wangenau/eminus/blob/main/eminus/xc/lda_x.py
        Licensed under the Apache License, Version 2.0.

    """
    
    pi34 = (3 / (4 * np.pi))**(1 / 3)
    f = -3 / 4 * (3 / (2 * np.pi))**(2 / 3)
    rs = pi34 * np.power(rho, -1 / 3)
    ex = f / rs
    vx = 4 / 3 * ex
    # return {'zk':ex, 'vrho':vx}
    return ex, vx

@fuse(kernel_name='lda_x_cupy')
def lda_x_cupy(rho):
    """
    GPU-accelerated version of the LDA exchange functional using CuPy.

    This is the same as `lda_x` but leverages CuPy for GPU computation. It corresponds to 
    `LDA_X` with ID 1 in LibXC for spin-unpolarized electron density.

    Parameters
    ----------
    rho : cupy.ndarray
        Electron density array (spin-unpolarized), on GPU.

    Returns
    -------
    ex : cupy.ndarray
        Exchange energy density at each point.

    vx : cupy.ndarray
        Exchange potential (functional derivative of exchange energy with respect to density).

    Notes
    -----
    Based on the Slater exchange:
        - Phys. Rev. 81, 385 (1951)
    Adapted from: https://github.com/wangenau/eminus/blob/main/eminus/xc/lda_x.py
    Licensed under the Apache License, Version 2.0.

    This version is fused using CuPy's `@fuse` decorator for better performance.

    """
    
    pi34 = (3 / (4 * cp.pi))**(1 / 3)
    f = -3 / 4 * (3 / (2 * cp.pi))**(2 / 3)
    rs = pi34 * cp.power(rho, -1 / 3)
    ex = f / rs
    vx = 4 / 3 * ex
    # return {'zk':ex, 'vrho':vx}
    return ex, vx