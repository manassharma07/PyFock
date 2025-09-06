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
# from pyfock.XC import lda_c_pw_mod, lda_c_pw_mod_cupy
from pyfock.XC.lda_c_pw_mod import lda_c_pw_mod_, lda_c_pw_mod_cupy_

# The following implementation of the PBE correlation has been taken from this repository (https://github.com/wangenau/eminus/blob/main/eminus/xc/gga_c_pbe.py)
# pretty much as is. The repo has the Apache 2.0 license.


def gga_c_pbe_cupy_(rho, sigma):
    # Taken from: https://github.com/wangenau/eminus/blob/main/eminus/xc/gga_c_pbe.py
    # Perdew-Burke-Ernzerhof parametrization of the correlation functional (spin-paired).
    # Corresponds to the functional with the label GGA_C_PBE and ID 130 in Libxc.
    # Reference: Phys. Rev. Lett. 78, 1396.

    rho = np.maximum(rho, 1e-12)

    beta = 0.06672455060314922
    gamma = (1 - np.log(2)) / np.pi**2

    pi34 = (3 / (4 * np.pi))**(1 / 3)
    rs = pi34 * rho**(-1 / 3)
    norm_dn = np.sqrt(sigma)
    ec, vc = lda_c_pw_mod_(rho)

    kf = (9 / 4 * np.pi)**(1 / 3) / rs
    ks = np.sqrt(4 * kf / np.pi)
    divt = 2 * ks * rho
    t = norm_dn / divt
    expec = np.exp(-ec / gamma)
    A = beta / (gamma * (expec - 1))
    t2 = t**2
    At2 = A * t2
    A2t4 = At2**2
    divsum = 1 + At2 + A2t4
    div = (1 + At2) / divsum
    nolog = 1 + beta / gamma * t2 * div
    gec = gamma * np.log(nolog)

    factor = A2t4 * (2 + At2) / divsum**2
    dgec = beta * t2 / nolog * (-7 / 3 * div - factor * (A * expec * (vc - ec) / beta - 7 / 3))
    gvc = gec + dgec

    vsigmac = beta / (divt * ks) * (div - factor) / nolog

    ec += gec
    vc += gvc
    vsigma = 0.5*vsigmac
    

    return ec, vc, vsigma

def gga_c_pbe(rho, sigma):
    # Taken from: https://github.com/wangenau/eminus/blob/main/eminus/xc/gga_c_pbe.py
    # Perdew-Burke-Ernzerhof parametrization of the correlation functional (spin-paired).
    # Corresponds to the functional with the label GGA_C_PBE and ID 130 in Libxc.
    # Reference: Phys. Rev. Lett. 78, 1396.
    
    ec, vc, vsigma = gga_c_pbe_(rho, sigma)
    
    vsigma[np.isnan(vsigma)] = 0
    vc[np.isnan(vc)] = 0
    ec[np.isnan(ec)] = 0

    return ec, vc, vsigma

@fuse(kernel_name='pbe_c_pbe_cupy_')
def gga_c_pbe_cupy_(rho, sigma):
    # Taken from: https://github.com/wangenau/eminus/blob/main/eminus/xc/gga_c_pbe.py
    # Perdew-Burke-Ernzerhof parametrization of the correlation functional (spin-paired).
    # Corresponds to the functional with the label GGA_C_PBE and ID 130 in Libxc.
    # Reference: Phys. Rev. Lett. 78, 1396.

    rho = cp.maximum(rho, 1e-12)

    beta = 0.06672455060314922
    gamma = (1 - cp.log(2)) / cp.pi**2

    pi34 = (3 / (4 * cp.pi))**(1 / 3)
    rs = pi34 * rho**(-1 / 3)
    norm_dn = cp.sqrt(sigma)
    ec, vc = lda_c_pw_mod_cupy_(rho)

    kf = (9 / 4 * cp.pi)**(1 / 3) / rs
    ks = cp.sqrt(4 * kf / cp.pi)
    divt = 2 * ks * rho
    t = norm_dn / divt
    expec = cp.exp(-ec / gamma)
    A = beta / (gamma * (expec - 1))
    t2 = t**2
    At2 = A * t2
    A2t4 = At2**2
    divsum = 1 + At2 + A2t4
    div = (1 + At2) / divsum
    nolog = 1 + beta / gamma * t2 * div
    gec = gamma * cp.log(nolog)

    factor = A2t4 * (2 + At2) / divsum**2
    dgec = beta * t2 / nolog * (-7 / 3 * div - factor * (A * expec * (vc - ec) / beta - 7 / 3))
    gvc = gec + dgec

    vsigmac = beta / (divt * ks) * (div - factor) / nolog

    ec += gec
    vc += gvc
    vsigma = 0.5*vsigmac
    

    return ec, vc, vsigma

def gga_c_pbe_cupy(rho, sigma):
    # Taken from: https://github.com/wangenau/eminus/blob/main/eminus/xc/gga_c_pbe.py
    # Perdew-Burke-Ernzerhof parametrization of the correlation functional (spin-paired).
    # Corresponds to the functional with the label GGA_C_PBE and ID 130 in Libxc.
    # Reference: Phys. Rev. Lett. 78, 1396.
    
    ec, vc, vsigma = gga_c_pbe_cupy_(rho, sigma)
    
    vsigma[cp.isnan(vsigma)] = 0
    vc[cp.isnan(vc)] = 0
    ec[cp.isnan(ec)] = 0

    return ec, vc, vsigma