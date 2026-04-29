try:
    import cupy as cp
except Exception:
    cp = None
import numpy as np


def _pw92_ec(rs, polarized, xp):
    if polarized:
        A = 0.01554535
        a1 = 0.20548
        b1 = 14.1189
        b2 = 6.1977
        b3 = 3.3662
        b4 = 0.62517
    else:
        A = 0.0310907
        a1 = 0.21370
        b1 = 7.5957
        b2 = 3.5876
        b3 = 1.6382
        b4 = 0.49294
    sqrt_rs = xp.sqrt(rs)
    q = 2.0 * A * (b1 * sqrt_rs + b2 * rs + b3 * rs * sqrt_rs + b4 * rs * rs)
    return -2.0 * A * (1.0 + a1 * rs) * xp.log1p(1.0 / (q + 1e-30))


def _pbe_c_energy_spin(rho, sigma, zeta, xp):
    rho = xp.maximum(rho, 1e-12)
    sigma = xp.maximum(sigma, 0.0)
    beta = 0.06672455060314922
    gamma = (1.0 - xp.log(2.0)) / xp.pi**2

    rs = (3.0 / (4.0 * xp.pi * rho)) ** (1.0 / 3.0)
    ec_lda = _pw92_ec(rs, zeta > 0.5, xp)
    phi = ((1.0 + zeta) ** (2.0 / 3.0) + (1.0 - zeta) ** (2.0 / 3.0)) / 2.0
    kf = (3.0 * xp.pi**2 * rho) ** (1.0 / 3.0)
    ks = xp.sqrt(4.0 * kf / xp.pi)
    t = xp.sqrt(sigma) / (2.0 * phi * ks * rho + 1e-30)

    phi3 = phi**3
    arg = -ec_lda / (gamma * phi3)
    arg = xp.minimum(arg, 50.0)
    A = beta / (gamma * (xp.exp(arg) - 1.0) + 1e-30)
    t2 = t * t
    At2 = A * t2
    h_arg = 1.0 + (beta / gamma) * t2 * (1.0 + At2) / (1.0 + At2 + At2 * At2)
    return ec_lda + gamma * phi3 * xp.log(h_arg)


def _tpss_c_energy(rho, sigma, tau, xp):
    rho = xp.maximum(rho, 1e-12)
    sigma = xp.maximum(sigma, 0.0)
    tau = xp.maximum(tau, 1e-14)

    C = 0.53
    d = 2.8
    tau_w_over_tau = xp.minimum(sigma / (8.0 * rho * tau + 1e-30), 1.0)
    tau2 = tau_w_over_tau * tau_w_over_tau

    ec_pbe = _pbe_c_energy_spin(rho, sigma, 0.0, xp)
    ec_pbe_spin = _pbe_c_energy_spin(0.5 * rho, 0.25 * sigma, 1.0, xp)
    ec_tilde = xp.maximum(ec_pbe_spin, ec_pbe)
    ec_revpkzb = ec_pbe * (1.0 + C * tau2) - (1.0 + C) * tau2 * ec_tilde
    return ec_revpkzb * (1.0 + d * ec_revpkzb * tau_w_over_tau**3)


def _finite_difference_vxc(energy_func, rho, sigma, tau, xp):
    eps = energy_func(rho, sigma, tau)

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


def mgga_c_tpss(rho, sigma, tau):
    rho = np.asarray(rho)
    sigma = np.asarray(sigma)
    tau = np.asarray(tau)
    return _finite_difference_vxc(lambda r, s, t: _tpss_c_energy(r, s, t, np), rho, sigma, tau, np)


def mgga_c_tpss_cupy(rho, sigma, tau):
    if cp is None:
        raise ImportError("CuPy is required for mgga_c_tpss_cupy")
    return _finite_difference_vxc(lambda r, s, t: _tpss_c_energy(r, s, t, cp), rho, sigma, tau, cp)
