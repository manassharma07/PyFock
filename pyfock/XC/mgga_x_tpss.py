try:
    import cupy as cp
except Exception:
    cp = None
import numpy as np


def _tpss_x_energy(rho, sigma, tau, xp):
    rho = xp.maximum(rho, 1e-12)
    sigma = xp.maximum(sigma, 0.0)
    tau = xp.maximum(tau, 1e-14)

    b = 0.40
    c = 1.59096
    e = 1.537
    kappa = 0.804
    mu = 0.21951

    pi = xp.pi
    kf2 = (3.0 * pi**2) ** (2.0 / 3.0) * rho ** (2.0 / 3.0)
    p = sigma / (4.0 * kf2 * rho**2 + 1e-30)
    tau_w = sigma / (8.0 * rho + 1e-30)
    z = xp.minimum(tau_w / tau, 1.0)
    tau_unif = 0.3 * (3.0 * pi**2) ** (2.0 / 3.0) * rho ** (5.0 / 3.0)
    alpha = xp.maximum((tau - tau_w) / (tau_unif + 1e-30), 0.0)

    qb = (9.0 / 20.0) * (alpha - 1.0) / xp.sqrt(1.0 + b * alpha * (alpha - 1.0) + 1e-30) + 2.0 * p / 3.0
    z2 = z * z
    root_arg = 0.5 * ((3.0 * z / 5.0) ** 2 + p * p)
    x_num = (
        (10.0 / 81.0 + c * z2 / (1.0 + z2) ** 2) * p
        + 146.0 / 2025.0 * qb * qb
        - 73.0 / 405.0 * qb * xp.sqrt(root_arg)
        + (10.0 / 81.0) ** 2 * p * p / kappa
        + 2.0 * xp.sqrt(e) * (10.0 / 81.0) * (3.0 * z / 5.0) ** 2
        + e * mu * p**3
    )
    x = x_num / (1.0 + xp.sqrt(e) * p) ** 2
    fx = 1.0 + kappa - kappa / (1.0 + x / kappa)
    ex_unif = -0.75 * (3.0 / pi) ** (1.0 / 3.0) * rho ** (1.0 / 3.0)
    return ex_unif * fx


def _finite_difference_vxc(energy_func, rho, sigma, tau, xp):
    eps = energy_func(rho, sigma, tau)
    f0 = rho * eps

    hr = 1e-5 * xp.maximum(xp.abs(rho), 1e-3)
    hs = 1e-5 * xp.maximum(xp.abs(sigma), 1e-6)
    ht = 1e-5 * xp.maximum(xp.abs(tau), 1e-6)

    fp = (rho + hr) * energy_func(rho + hr, sigma, tau)
    rm = xp.maximum(rho - hr, 1e-12)
    fm = rm * energy_func(rm, sigma, tau)
    vrho = (fp - fm) / ((rho + hr) - rm)

    sp = sigma + hs
    sm = xp.maximum(sigma - hs, 0.0)
    vsigma = ((rho * energy_func(rho, sp, tau)) - (rho * energy_func(rho, sm, tau))) / (sp - sm + 1e-30)

    tp = tau + ht
    tm = xp.maximum(tau - ht, 1e-14)
    vtau = ((rho * energy_func(rho, sigma, tp)) - (rho * energy_func(rho, sigma, tm))) / (tp - tm)

    eps = xp.where(xp.isfinite(eps), eps, 0.0)
    vrho = xp.where(xp.isfinite(vrho), vrho, 0.0)
    vsigma = xp.where(xp.isfinite(vsigma), vsigma, 0.0)
    vtau = xp.where(xp.isfinite(vtau), vtau, 0.0)
    return eps, vrho, vsigma, vtau


def mgga_x_tpss(rho, sigma, tau):
    rho = np.asarray(rho)
    sigma = np.asarray(sigma)
    tau = np.asarray(tau)
    return _finite_difference_vxc(lambda r, s, t: _tpss_x_energy(r, s, t, np), rho, sigma, tau, np)


def mgga_x_tpss_cupy(rho, sigma, tau):
    if cp is None:
        raise ImportError("CuPy is required for mgga_x_tpss_cupy")
    return _finite_difference_vxc(lambda r, s, t: _tpss_x_energy(r, s, t, cp), rho, sigma, tau, cp)
