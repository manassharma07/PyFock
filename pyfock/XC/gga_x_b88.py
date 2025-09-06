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

def gga_x_b88_e(rho, sigma):
    # Adapted from https://github.com/dylan-jayatilaka/tonto/blob/master/foofiles/dft_functional.foo
    # The restricted Becke 88 exchange energy density.
    # Corresponds to 106 id in Libxc

    # rho_cutoff = 1e-12  # define rho_cutoff constant
    rho = np.maximum(rho, 1e-12)

    beta = 0.0042  # beta parameter
    beta6 = 6 * beta
    const = (3 / 2) * ((3 / (4 * np.pi)) ** (1 / 3))
    two_m13 = 2 ** (-1 / 3)
    two_m43 = 1 / 2 * two_m13


    rho_13 = rho ** (1 / 3)
    rho_43 = two_m43 * rho * rho_13
    gg = 1 / 2 * np.sqrt(sigma)
    x = gg / rho_43
    x2 = x ** 2
    log_term = np.log(x + np.sqrt(1 + x2))
    ex = - two_m13 * rho_13 * (const + beta * x2 / (1 + beta6 * x * log_term))

    return ex

def gga_x_b88_v(rho, sigma):
    # Adapted from https://github.com/dylan-jayatilaka/tonto/blob/master/foofiles/dft_functional.foo
    # The restricted Becke 88 exchange potential. These equations are
    # essentially the same as in the appendix of JCP 98(7) 5612-5626.
    # Note that (A5) is in error because the gamma variables should be square
    # rooted. Note that (A6) is in error because the power of rho_alpha should
    # be 1/3 not 4/3.
    # Corresponds to 106 id in Libxc

    rho = np.maximum(rho, 1e-12)
    sigma = np.maximum(sigma, 1e-12)

    beta = 0.0042  # beta parameter
    beta2 = 2 * beta
    beta6 = 6 * beta
    bbta6 = 6 * beta * beta
    const = 3 / 2 * (3 / (4 * np.pi)) ** (1 / 3)

    rho1 = 0.5 * rho
    rho_13 = rho1 ** (1 / 3)
    rho_43 = rho1 * rho_13
    gg = 0.5 * np.sqrt(sigma)
    x = gg / rho_43
    x2 = x * x
    sq = np.sqrt(1 + x2)
    as_ = np.log(x + sq) 
    d = 1 / (1 + beta6 * x * as_)
    d2 = d * d
    g0 = const + beta * x2 * d
    g1 = (beta2 * x + bbta6 * x2 * (as_ - x / sq)) * d2
    gg = 0.5 * g1 / gg  # a factor 1/2 x 2 which cancel, here
    vx = -4 / 3 * rho_13 * (g0 - x * g1)
    vsigma = -0.5*gg # This works (verified)

    return vx, vsigma


def gga_x_b88(rho, sigma):
    # Corresponds to 106 id in Libxc
    ex = gga_x_b88_e(rho, sigma)
    vx, vsigma = gga_x_b88_v(rho, sigma)
    return ex, vx, vsigma

@fuse(kernel_name='gga_x_b88_e_cupy')
def gga_x_b88_e_cupy(rho, sigma):
    # Adapted from https://github.com/dylan-jayatilaka/tonto/blob/master/foofiles/dft_functional.foo
    # The restricted Becke 88 exchange energy density.
    # Corresponds to 106 id in Libxc

    # rho_cutoff = 1e-12  # define rho_cutoff constant
    # rho = cp.maximum(rho, 1e-12)

    beta = 0.0042  # beta parameter
    beta6 = 6 * beta
    const = (3 / 2) * ((3 / (4 * cp.pi)) ** (1 / 3))
    two_m13 = 2 ** (-1 / 3)
    two_m43 = 1 / 2 * two_m13


    rho_13 = rho ** (1 / 3)
    rho_43 = two_m43 * rho * rho_13
    gg = 1 / 2 * cp.sqrt(sigma)
    x = gg / rho_43
    x2 = x ** 2
    log_term = cp.log(x + cp.sqrt(1 + x2))
    ex = - two_m13 * rho_13 * (const + beta * x2 / (1 + beta6 * x * log_term))

    return ex

@fuse(kernel_name='gga_x_b88_v_cupy')
def gga_x_b88_v_cupy(rho, sigma):
    # Adapted from https://github.com/dylan-jayatilaka/tonto/blob/master/foofiles/dft_functional.foo
    # The restricted Becke 88 exchange potential. These equations are
    # essentially the same as in the appendix of JCP 98(7) 5612-5626.
    # Note that (A5) is in error because the gamma variables should be square
    # rooted. Note that (A6) is in error because the power of rho_alpha should
    # be 1/3 not 4/3.
    # Corresponds to 106 id in Libxc

    # rho = cp.maximum(rho, 1e-12)
    # sigma = cp.maximum(sigma, 1e-12)

    beta = 0.0042  # beta parameter
    beta2 = 2 * beta
    beta6 = 6 * beta
    bbta6 = 6 * beta * beta
    const = 3 / 2 * (3 / (4 * cp.pi)) ** (1 / 3)

    rho1 = 0.5 * rho
    rho_13 = rho1 ** (1 / 3)
    rho_43 = rho1 * rho_13
    gg = 0.5 * cp.sqrt(sigma)
    # print(cp.min(gg))
    x = gg / rho_43
    x2 = x * x
    # print(cp.min(x2))
    sq = cp.sqrt(1 + x2)
    # print(cp.isnan(cp.sum(sq)))
    as_ = cp.log(x + sq) 
    # sq = cp.sqrt(1 + x2)
    # as_ = cp.log(cp.maximum(x + sq, 1e-12))
    d = 1 / (1 + beta6 * x * as_)
    d2 = d * d
    g0 = const + beta * x2 * d
    g1 = (beta2 * x + bbta6 * x2 * (as_ - x / sq)) * d2
    gg = 0.5 * g1 / gg  # a factor 1/2 x 2 which cancel, here
    vx = -4 / 3 * rho_13 * (g0 - x * g1)
    vsigma = -0.5*gg # This works (verified)

    return vx, vsigma


def gga_x_b88_cupy(rho, sigma):
    # Corresponds to 106 id in Libxc
    ex = gga_x_b88_e_cupy(rho, sigma)
    vx, vsigma = gga_x_b88_v_cupy(rho, sigma)
    vsigma[cp.isnan(vsigma)] = 0
    vx[cp.isnan(vx)] = 0
    ex[cp.isnan(ex)] = 0
    return ex, vx, vsigma