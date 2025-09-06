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


def gga_c_lyp_e(rho, sigma):
    # Corresponds to 106 id in Libxc
    # Adapted from https://github.com/dylan-jayatilaka/tonto/blob/master/foofiles/dft_functional.foo
    # Return the values of the Lee-Yang-Parr energy density and potential
    rho = np.maximum(rho, 1e-12)
    # Constants
    a = 0.04918
    b = 0.132
    c = 0.2533
    d = 0.349
    const = (3/10) * (3 * np.pi ** 2) ** (2/3)
    fac = 2 ** (11/3) * const

    rho_13 = rho ** (1 / 3)
    rho_m13 = 1 / rho_13
    rho1 = 1/2 * rho
    gg = 1/4 * sigma
    gamma_inv = 1 / (1 + d * rho_m13)
    a_b_omega = a * b * np.exp(-c * rho_m13) * gamma_inv * rho_m13 ** 11
    delta = (c + d * gamma_inv) * rho_m13
    ec = - a * gamma_inv + 1/2 * a_b_omega * rho1 * gg * (6 + 14 * delta) * (1/9) - a_b_omega * fac * rho1 ** (11/3)

    return ec


def gga_c_lyp_v(rho, sigma):
    # Corresponds to 106 id in Libxc
    # Adapted from https://github.com/dylan-jayatilaka/tonto/blob/master/foofiles/dft_functional.foo
    # Return the derivatives of the LYP correlation functional.
    rho = np.maximum(rho, 1e-12)
    const = (3 / 10) * (3 * np.pi ** 2) ** (2 / 3)
    two_13 = 2 ** (1 / 3)
    two_m13 = 1 / two_13
    two_113 = 16 * two_m13
    two_m113 = 1 / two_113
    a = 0.04918
    b = 0.132
    c = 0.2533
    d = 0.349
    e = two_113 * const
    ab9 = a * b / 9

    rho1 = 0.5 * rho
    aa = 1/4 * sigma
    rho_13 = rho1 ** (1 / 3)
    rho_m13 = 1 / rho_13
    rho_83 = rho1 ** 3 * rho_m13
    rho_m83 = 1 / rho_83
    p_third = two_m13 * rho_m13
    gamma_inv = 1 / (1 + d * p_third)
    mu = d * gamma_inv * p_third
    abw9_pa = two_m113 * ab9 * np.exp(-c * p_third) * rho_m83 * gamma_inv
    delta = c * p_third + mu
    vc = -a * gamma_inv * (1 + mu / 3) \
                + abw9_pa * aa * (7 / 3 * (mu ** 2 + delta ** 2) - 13 * delta - 5) \
                - abw9_pa * e * (3 * delta + 9) * rho_83
    fac = abw9_pa * rho1 * (6 + 14 * delta)
    vsigma = 0.25*fac # Final (works)

    return vc, vsigma

def gga_c_lyp(rho, sigma):
    # Corresponds to 131 id in Libxc
    ec = gga_c_lyp_e(rho, sigma)
    vc, vsigma = gga_c_lyp_v(rho, sigma)
    return ec, vc, vsigma

@fuse(kernel_name='gga_c_lyp_e_cupy')
def gga_c_lyp_e_cupy(rho, sigma):
    # Corresponds to 106 id in Libxc
    # Adapted from https://github.com/dylan-jayatilaka/tonto/blob/master/foofiles/dft_functional.foo
    # Return the values of the Lee-Yang-Parr energy density and potential
    # rho = cp.maximum(rho, 1e-12)
    # Constants
    a = 0.04918
    b = 0.132
    c = 0.2533
    d = 0.349
    const = (3/10) * (3 * cp.pi ** 2) ** (2/3)
    fac = 2 ** (11/3) * const

    rho_13 = rho ** (1 / 3)
    rho_m13 = 1 / rho_13
    rho1 = 1/2 * rho
    gg = 1/4 * sigma
    gamma_inv = 1 / (1 + d * rho_m13)
    a_b_omega = a * b * cp.exp(-c * rho_m13) * gamma_inv * rho_m13 ** 11
    delta = (c + d * gamma_inv) * rho_m13
    ec = - a * gamma_inv + 1/2 * a_b_omega * rho1 * gg * (6 + 14 * delta) * (1/9) - a_b_omega * fac * rho1 ** (11/3)

    return ec

@fuse(kernel_name='gga_c_lyp_v_cupy')
def gga_c_lyp_v_cupy(rho, sigma):
    # Corresponds to 106 id in Libxc
    # Adapted from https://github.com/dylan-jayatilaka/tonto/blob/master/foofiles/dft_functional.foo
    # Return the derivatives of the LYP correlation functional.
    # rho = cp.maximum(rho, 1e-12)
    const = (3 / 10) * (3 * cp.pi ** 2) ** (2 / 3)
    two_13 = 2 ** (1 / 3)
    two_m13 = 1 / two_13
    two_113 = 16 * two_m13
    two_m113 = 1 / two_113
    a = 0.04918
    b = 0.132
    c = 0.2533
    d = 0.349
    e = two_113 * const
    ab9 = a * b / 9

    rho1 = 0.5 * rho
    aa = 1/4 * sigma
    rho_13 = rho1 ** (1 / 3)
    rho_m13 = 1 / rho_13
    rho_83 = rho1 ** 3 * rho_m13
    rho_m83 = 1 / rho_83
    p_third = two_m13 * rho_m13
    gamma_inv = 1 / (1 + d * p_third)
    mu = d * gamma_inv * p_third
    abw9_pa = two_m113 * ab9 * cp.exp(-c * p_third) * rho_m83 * gamma_inv
    delta = c * p_third + mu
    vc = -a * gamma_inv * (1 + mu / 3) \
                + abw9_pa * aa * (7 / 3 * (mu ** 2 + delta ** 2) - 13 * delta - 5) \
                - abw9_pa * e * (3 * delta + 9) * rho_83
    fac = abw9_pa * rho1 * (6 + 14 * delta)
    vsigma = 0.25*fac # Final (works)

    return vc, vsigma

def gga_c_lyp_cupy(rho, sigma):
    # Corresponds to 131 id in Libxc
    ec = gga_c_lyp_e_cupy(rho, sigma)
    vc, vsigma = gga_c_lyp_v_cupy(rho, sigma)
    vsigma[cp.isnan(vsigma)] = 0
    vc[cp.isnan(vc)] = 0
    ec[cp.isnan(ec)] = 0
    return ec, vc, vsigma