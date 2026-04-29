try:
    import cupy as cp
except Exception:
    cp = None
import numpy as np


_M06L_X_A = (
    0.3987756, 0.2548219, 0.3923994, -2.103655, -6.302147, 10.97615,
    30.97273, -23.18489, -56.73480, 21.60364, 34.21814, -9.049762,
)
_M06L_X_D = (
    0.6012244, 0.004748822, -0.008635108,
    -0.000009308062, 0.00004482811, 0.0,
)

_M06L_C_CSS = (5.349466e-01, 5.396620e-01, -3.161217e+01, 5.149592e+01, -2.919613e+01)
_M06L_C_CAB = (6.042374e-01, 1.776783e+02, -2.513252e+02, 7.635173e+01, -1.255699e+01)
_M06L_C_DSS = (4.650534e-01, 1.617589e-01, 1.833657e-01, 4.692100e-04, -4.990573e-03, 0.0)
_M06L_C_DAB = (3.957626e-01, -5.614546e-01, 1.403963e-02, 9.831442e-04, -3.577176e-03, 0.0)


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


def _m06_l_x_energy(rho, sigma, tau, xp):
    rho = xp.maximum(rho, 1e-12)
    sigma = xp.maximum(sigma, 0.0)
    tau = xp.maximum(tau, 1e-14)

    cbrt2 = 2.0 ** (1.0 / 3.0)
    cbrt2_2 = cbrt2 * cbrt2
    cbrt6_2 = 6.0 ** (2.0 / 3.0)
    pi2_23 = (xp.pi * xp.pi) ** (2.0 / 3.0)

    rho13 = rho ** (1.0 / 3.0)
    rho23 = rho13 * rho13
    rho83 = rho23 * rho * rho
    rho53 = rho23 * rho

    x = sigma * cbrt2_2 / rho83
    z = tau * cbrt2_2 / rho53
    cf = cbrt6_2 * pi2_23
    t = 0.3 * cf

    f_pw86 = 1.804 - 0.646416 / (0.804 + 0.0091464571985215458336 * 6.0 ** (1.0 / 3.0) / pi2_23 * x)
    w = (t - z) / (t + z)

    poly = xp.zeros_like(rho)
    wpow = xp.ones_like(rho)
    for coeff in _M06L_X_A:
        poly = poly + coeff * wpow
        wpow = wpow * w

    u = 1.0 + 0.00186726 * x + 0.00373452 * z - 0.001120356 * cf
    y = 2.0 * z - 0.6 * cf
    d0, d1, d2, d3, d4, d5 = _M06L_X_D
    h1 = d0 / u
    h2 = (d1 * sigma * cbrt2_2 / rho83 + d2 * y) / (u * u)
    h3 = (
        d4 * sigma * cbrt2_2 / rho83 * y
        + 2.0 * d3 * sigma * sigma * cbrt2 / (rho ** (16.0 / 3.0))
        + d5 * y * y
    ) / (u * u * u)

    return -0.75 * (3.0 / xp.pi) ** (1.0 / 3.0) * rho13 * (f_pw86 * poly + h1 + h2 + h3)


def _pw92_mod_unpolarized(rs, xp):
    sqrt_rs = xp.sqrt(rs)
    rs32 = rs * sqrt_rs
    rs2 = rs * rs
    q = 3.79785 * sqrt_rs + 0.8969 * rs + 0.204775 * rs32 + 0.123235 * rs2 / 4.0
    return -0.0621814 * (1.0 + 0.053425 * rs) * xp.log(1.0 + 16.081979498692535067 / q)


def _pw92_mod_parallel(rs, xp):
    sqrt_rs = xp.sqrt(rs)
    rs32 = rs * sqrt_rs
    rs2 = rs * rs
    q = 7.05945 * sqrt_rs + 1.549425 * rs + 0.420775 * rs32 + 0.1562925 * rs2 / 4.0
    return -0.0310907 * (1.0 + 0.05137 * rs) * xp.log(1.0 + 32.163958997385070134 / q)


def _m06_l_c_energy(rho, sigma, tau, xp):
    rho = xp.maximum(rho, 1e-12)
    sigma = xp.maximum(sigma, 0.0)
    tau = xp.maximum(tau, 1e-14)

    gamma_ss = 0.06
    gamma_ab = 0.0031
    alpha_ss = 0.00515088
    alpha_ab = 0.00304966
    fermi_d = 1e-10

    cbrt2 = 2.0 ** (1.0 / 3.0)
    cbrt2_2 = cbrt2 * cbrt2
    cbrt6_2 = 6.0 ** (2.0 / 3.0)
    pi2_23 = (xp.pi * xp.pi) ** (2.0 / 3.0)

    rho13 = rho ** (1.0 / 3.0)
    rho23 = rho13 * rho13
    rho83 = rho23 * rho * rho
    rho163 = rho ** (16.0 / 3.0)
    x = sigma * cbrt2_2 / rho83
    z = tau * cbrt2_2 / (rho23 * rho)
    cf = cbrt6_2 * pi2_23

    rs_ss = (96.0 / (xp.pi * rho)) ** (1.0 / 3.0)
    rs_ab = (48.0 / (xp.pi * rho)) ** (1.0 / 3.0)
    e_ss_lda = 0.5 * _pw92_mod_parallel(rs_ss, xp)
    e_ab_lda = _pw92_mod_unpolarized(rs_ab, xp) - 2.0 * e_ss_lda

    def gamma_series(coeffs, gamma, scale):
        u = 1.0 + scale * gamma * x
        y = scale * gamma * x / u
        return coeffs[0] + coeffs[1] * y + coeffs[2] * y * y + coeffs[3] * y**3 + coeffs[4] * y**4

    same_spin = gamma_series(_M06L_C_CSS, gamma_ss, 1.0)
    opposite_spin = gamma_series(_M06L_C_CAB, gamma_ab, 2.0)

    tau_w_over_tau = sigma / (8.0 * rho * tau)
    fermi = 1.0 - xp.exp(-8.0 * tau * tau * cbrt2 / (rho ** (10.0 / 3.0) * fermi_d * fermi_d))
    self_interaction = (1.0 - tau_w_over_tau) * fermi

    same_lda_term = 2.0 * e_ss_lda * same_spin * self_interaction
    opposite_lda_term = e_ab_lda * opposite_spin

    def kinetic_series(coeffs, alpha, scale):
        u = 1.0 + alpha * (scale * x + 2.0 * scale * z - 0.6 * scale * cf)
        y = 2.0 * scale * z - 0.6 * scale * cf
        return (
            coeffs[0] / u
            + (scale * coeffs[1] * x + coeffs[2] * y) / (u * u)
            + (scale * coeffs[4] * x * y + 2.0 * scale * scale * coeffs[3] * sigma * sigma * cbrt2 / rho163 + coeffs[5] * y * y)
            / (u * u * u)
        )

    same_kinetic = 2.0 * e_ss_lda * kinetic_series(_M06L_C_DSS, alpha_ss, 1.0) * (1.0 - tau_w_over_tau)
    opposite_kinetic = e_ab_lda * kinetic_series(_M06L_C_DAB, alpha_ab, 2.0)

    return same_lda_term + opposite_lda_term + same_kinetic + opposite_kinetic


def mgga_x_m06_l(rho, sigma, tau):
    rho = np.asarray(rho)
    sigma = np.asarray(sigma)
    tau = np.asarray(tau)
    return _finite_difference_vxc(lambda r, s, t: _m06_l_x_energy(r, s, t, np), rho, sigma, tau, np)


def mgga_x_m06_l_cupy(rho, sigma, tau):
    if cp is None:
        raise ImportError("CuPy is required for mgga_x_m06_l_cupy")
    return _finite_difference_vxc(lambda r, s, t: _m06_l_x_energy(r, s, t, cp), rho, sigma, tau, cp)


def mgga_c_m06_l(rho, sigma, tau):
    rho = np.asarray(rho)
    sigma = np.asarray(sigma)
    tau = np.asarray(tau)
    return _finite_difference_vxc(lambda r, s, t: _m06_l_c_energy(r, s, t, np), rho, sigma, tau, np)


def mgga_c_m06_l_cupy(rho, sigma, tau):
    if cp is None:
        raise ImportError("CuPy is required for mgga_c_m06_l_cupy")
    return _finite_difference_vxc(lambda r, s, t: _m06_l_c_energy(r, s, t, cp), rho, sigma, tau, cp)
