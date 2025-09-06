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

def lda_c_pw_mod_(rho):
    # From https://github.com/wangenau/eminus
    # Corresponds to the functional with the label LDA_C_PW_MOD and ID 13 in Libxc.
    # Reference: Phys. Rev. B 45, 13244.

    rho = np.maximum(rho, 1e-12)

    A = 0.0310907 
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

def lda_c_pw_mod(rho):
    ec, vc = lda_c_pw_mod_(rho)
    vc[np.isnan(vc)] = 0
    ec[np.isnan(ec)] = 0
    return ec, vc


@fuse(kernel_name='lda_c_pw_mod_cupy_')
def lda_c_pw_mod_cupy_(rho):
    # From https://github.com/wangenau/eminus
    # Corresponds to the functional with the label LDA_C_PW_MOD and ID 13 in Libxc.
    # Reference: Phys. Rev. B 45, 13244.

    rho = cp.maximum(rho, 1e-12)

    A = 0.0310907
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

def lda_c_pw_mod_cupy(rho):
    ec, vc = lda_c_pw_mod_cupy_(rho)
    vc[cp.isnan(vc)] = 0
    ec[cp.isnan(ec)] = 0
    return ec, vc

