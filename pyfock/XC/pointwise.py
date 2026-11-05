"""
Point-wise (scalar) Numba versions of the native functionals that are implemented as
array expressions over NumPy or CuPy: the meta-GGAs (TPSS, M06-L, r2SCAN, TASK) and PW91
correlation. They are what :mod:`pyfock.Integrals.eval_xc_3` evaluates inside its parallel
kernel, one grid point at a time, so that no array temporaries are created.

The energy expressions are transcriptions of the array versions in ``mgga_x_tpss.py``,
``mgga_c_tpss.py``, ``mgga_m06_l.py``, ``mgga_r2scan_task.py`` and ``gga_c_pw91.py``
(``xp.`` -> ``np.``, ``xp.where`` -> a scalar select) and the derivatives use the same
finite-difference and complex-step schemes with the same step sizes, so the two agree to
rounding (``tests/test_eval_xc_3.py`` checks it). Any change to one of the array
versions must be carried over here.
"""
import math
import numpy as np
from numba import njit

from .mgga_r2scan_task import _R2SCAN_X_PARAMS, _TASK_C, _TASK_D, _TASK_H0X, _TASK_ANU, _TASK_BNU
from .mgga_m06_l import _M06L_X_A, _M06L_X_D, _M06L_C_CSS, _M06L_C_CAB, _M06L_C_DSS, _M06L_C_DAB

__all__ = ['POINTWISE_IDS', 'mgga_x_tpss_point', 'mgga_c_tpss_point', 'mgga_x_m06_l_point', 'mgga_c_m06_l_point',
           'mgga_x_r2scan_point', 'mgga_c_r2scan_point', 'mgga_x_task_point', 'gga_c_pw91_point',
           'eval_pointwise_mgga', 'eval_pointwise_gga']

# No fastmath here: the finite differences and the isfinite guards must see the exact
# IEEE results, and error_model='numpy' turns division by zero into inf/nan as NumPy does.
_NB = dict(cache=True, nogil=True, error_model='numpy')
_NBI = dict(cache=True, nogil=True, error_model='numpy', inline='always')

_PW91_RHO_CUTOFF = 1e-12
_PW91_COMPLEX_STEP = 1e-30


@njit(**_NBI)
def _where(cond, a, b):
    return a if cond else b


@njit(**_NBI)
def _cbrt(x):
    return x ** (1.0 / 3.0)


@njit(**_NBI)
def _pw3(cond, a, b):
    return a if cond else b


@njit(**_NBI)
def _pw5(cond1, a, cond2, b, c):
    if cond1:
        return a
    if cond2:
        return b
    return c


# ---------------------------------------------------------------------------------------
# r2SCAN and TASK (mgga_r2scan_task.py)
# ---------------------------------------------------------------------------------------
@njit(**_NB)
def _r2scan_x_energy(rho, sigma, tau):
    c1, c2, d, k1, eta, dp2 = _R2SCAN_X_PARAMS
    rho = np.maximum(rho, 1e-12)
    sigma = np.maximum(sigma, 1e-30)
    tau = np.maximum(tau, 1e-14)

    pi = np.pi
    cbrt2 = 2.0 ** (1.0 / 3.0)
    cbrt3 = 3.0 ** (1.0 / 3.0)
    cbrt6 = 6.0 ** (1.0 / 3.0)
    cbrtpi = pi ** (1.0 / 3.0)

    t7 = cbrt3 / cbrtpi
    t20 = _cbrt(rho)
    t22 = 20.0 / 27.0 + 5.0 * eta / 3.0
    t24 = cbrt6 * cbrt6
    t25 = pi * pi
    t26 = _cbrt(t25)
    t29 = t24 / (t26 * t25)
    t30 = sigma * sigma
    t38 = cbrt2 / (t20 * rho**5)
    t45 = np.exp(-t29 * t30 * t38 / (288.0 * dp2**4))
    t49 = (-0.162742215233874 * t22 * t45 + 10.0 / 81.0) * cbrt6
    t52 = t49 / (t26 * t26)
    t53 = cbrt2 * cbrt2
    t58 = sigma * t53 / (t20 * t20 * rho * rho)
    t61 = k1 + t52 * t58 / 24.0
    t65 = k1 * (1.0 - k1 / t61)
    t71 = tau * t53 / (t20 * t20 * rho) - t58 / 8.0
    t78 = 0.3 * t24 * t26 * t26 + eta * sigma * t53 / (8.0 * t20 * t20 * rho * rho)
    t80 = t71 / t78
    t81 = t80 <= 0.0
    t89 = t80 <= 2.5
    t90 = 2.5 < t80

    t83 = _pw3(0.0 < t80, 0.0, t80)
    t88 = np.exp(-c1 * t83 / (1.0 - t83))
    t91 = _pw3(t90, 2.5, t80)
    t93 = t91 * t91
    t95 = t93 * t91
    t97 = t93 * t93
    t99 = t97 * t91
    t101 = t97 * t93
    t106 = _pw3(t90, t80, 2.5)
    t110 = np.exp(c2 / (1.0 - t106))
    t112 = _pw5(
        t81,
        t88,
        t89,
        1.0
        - 0.667 * t91
        - 0.4445555 * t93
        - 0.663086601049 * t95
        + 1.45129704449 * t97
        - 0.887998041597 * t99
        + 0.234528941479 * t101
        - 0.023185843322 * t97 * t95,
        -d * t110)
    t115 = t112 * (0.174 - t65) + t65 + 1.0

    t125 = t24 / t26 * np.sqrt(sigma) * cbrt2 / (t20 * rho)
    t126 = np.sqrt(t125)
    t131 = 1.0 - np.exp(-9.8958 * np.sqrt(3.0) / t126)
    return -0.75 * t7 * t20 * t115 * t131


@njit(**_NB)
def _r2scan_c_energy(rho, sigma, tau):
    eta = 0.001
    rho = np.maximum(rho, 1e-12)
    sigma = np.maximum(sigma, 0.0)
    tau = np.maximum(tau, 1e-14)

    pi = np.pi
    cbrt2 = 2.0 ** (1.0 / 3.0)
    cbrt3 = 3.0 ** (1.0 / 3.0)
    cbrt4 = 4.0 ** (1.0 / 3.0)
    cbrt6 = 6.0 ** (1.0 / 3.0)

    t4 = _cbrt(1.0 / pi)
    t5 = cbrt3 * t4
    t7 = cbrt4 * cbrt4
    t8 = _cbrt(rho)
    t10 = t7 / t8
    t11 = t5 * t10
    t13 = 1.0 + 0.053425 * t11
    t14 = np.sqrt(t11)
    t16 = 0.8969 * t11
    t17 = t11 ** 1.5
    t18 = 0.204775 * t17
    t21 = cbrt3 * cbrt3 * t4 * t4
    t22 = t8 * t8
    t25 = t21 * cbrt4 / t22
    t26 = 0.123235 * t25
    t27 = 3.79785 * t14 + t16 + t18 + t26
    t31 = np.log(1.0 + 16.081979498692535067 / t27)
    t33 = 0.0621814 * t13 * t31
    t60 = 1.0 - np.log(2.0)
    t61 = pi * pi
    t63 = t60 / t61
    t69 = 1.0 / t60
    t74 = np.exp(t33 * t69 * t61)
    t75 = t74 - 1.0
    t77 = 1.0 + 0.025 * t11
    t79 = 1.0 + 0.04445 * t11
    t81 = t77 / t79
    t82 = rho * rho
    t84 = 1.0 / t8 / t82
    t93 = 1.0 / t75
    t95 = cbrt4 * t69 * t93 * cbrt3 * cbrt3 / t4
    t101 = t69
    t102 = t93
    t103 = 2.0
    t104 = t103 * t14
    t106 = 0.03138525 * t11
    t107 = 1.0 + 0.022225 * t104 + t106
    t108 = t107 * t107
    t114 = 1.0 / t108
    t116 = t103 / t14
    t118 = 0.04445 * t116 + 0.125541
    t122 = 1.898925 * t104 + t16 + t18 + t26
    t126 = np.log(1.0 + 16.081979498692535067 / t122)
    t128 = t122 * t122
    t130 = t13 / t128
    t132 = np.sqrt(t11)
    t135 = 3.79785 * t116 + 3.5876 + 1.22865 * t132 + 0.24647 * t11
    t136 = 1.0 / (1.0 + 16.081979498692535067 / t122)
    t137 = t135 * t136
    t141 = 2.58925 * t104 + 0.905775 * t11 + 0.1100325 * t17 + 0.1241775 * t25
    t144 = 1.0 + 29.608749977793437516 / t141
    t145 = np.log(t144)
    t149 = t141 * t141
    t154 = 5.1785 * t116 + 3.6231 + 0.660195 * t132 + 0.248355 * t11
    t157 = t154 / (t149 * t144)
    t160 = 0.0285764 * t114 * t118 + 0.01328816518 * t126 - t130 * t137
    t165 = 1.0 + 0.04445 * t14 + t106
    t166 = 1.0 / t165
    t172 = 5.0 * t5 * t10 * t160 - 45.0 * eta * (-0.0285764 * t166 + t33)
    t174 = t101 * t102 * t172
    t176 = _cbrt(t61)
    t177 = t176 * t176
    t179 = cbrt6 / t177
    t180 = cbrt2 * cbrt2
    t181 = t179 * t180
    t183 = 1.0 / t22 / t82
    t184 = sigma * t183
    t185 = cbrt6 * cbrt6
    t188 = t185 / (t176 * t61)
    t189 = sigma * sigma
    t194 = 1.0 / t8 / (t82 * t82 * rho)
    t198 = np.exp(-0.20444604078896369094 * t188 * cbrt2 * t189 * t194)
    t200 = t181 * t184 * t198
    t203 = 1.0 + 0.027439371595564631661 * t81 * sigma * t84 * cbrt2 * t95 + 0.043341108700271342816 * t174 * t200
    t204 = t203 ** 0.25
    t211 = t63 * np.log(t75 * (1.0 - 1.0 / t204) + 1.0)
    t216 = tau / (t22 * rho) - t184 / 8.0
    t223 = 0.15 * t185 * t177 * cbrt2 + eta * sigma * t183 / 8.0
    t225 = t216 / t223
    t226 = t225 <= 0.0
    t234 = t225 <= 2.5
    t235 = 2.5 < t225
    t228 = _pw3(0.0 < t225, 0.0, t225)
    t233 = np.exp(-0.64 * t228 / (1.0 - t228))
    t236 = _pw3(t235, 2.5, t225)
    t238 = t236 * t236
    t240 = t238 * t236
    t242 = t238 * t238
    t244 = t242 * t236
    t246 = t242 * t238
    t251 = _pw3(t235, t225, 2.5)
    t255 = np.exp(1.5 / (1.0 - t251))
    t257 = _pw5(
        t226,
        t233,
        t234,
        1.0
        - 0.64 * t236
        - 0.4352 * t238
        - 1.535685604549 * t240
        + 3.061560252175 * t242
        - 1.915710236206 * t244
        + 0.516884468372 * t246
        - 0.051848879792 * t242 * t240,
        -0.7 * t255)
    t261 = np.exp(t166) - 1.0
    t266 = 1.0 + 0.021337642104376358333 * t179 * t180 * sigma * t183
    t269 = 1.0 - 1.0 / (t266 ** 0.25)
    t272 = np.log(t261 * t269 + 1.0)
    t276 = -0.0285764 * t166 + 0.0285764 * t272 + t33 - t211
    return -t33 + t211 + t257 * t276


@njit(**_NB)
def _task_x_energy(rho, sigma, tau):
    rho = np.maximum(rho, 1e-12)
    sigma = np.maximum(sigma, 1e-30)
    tau = np.maximum(tau, 1e-14)

    pi = np.pi
    cbrt2 = 2.0 ** (1.0 / 3.0)
    cbrt3 = 3.0 ** (1.0 / 3.0)
    cbrt6 = 6.0 ** (1.0 / 3.0)
    cbrtpi = pi ** (1.0 / 3.0)
    task_bnu = _TASK_BNU
    task_anu = _TASK_ANU

    t7 = cbrt3 / cbrtpi
    t19 = _cbrt(rho)
    t23 = _cbrt(pi * pi)
    t24 = t23 * t23
    t28 = cbrt2 * cbrt2
    t30 = rho * rho
    t31 = t19 * t19
    t32 = t31 * t30
    t36 = cbrt6 / t24 * sigma * t28 / t32 / 24.0
    t37 = 0.0 < t36
    t38 = _pw3(t37, t36, 0.0)
    t42 = np.exp(-_TASK_C / (t38 ** 0.25))
    t44 = _pw3(t37, 1.0 - t42, 0.0)
    t46 = tau * tau
    t47 = t46 * t46
    t48 = t47 * cbrt3
    t49, t50, t51, t52, t53 = task_bnu
    t54 = t49 + t50 + t51 + t52 + t53
    t55 = rho * tau
    t63 = 0.0 < (0.9999999999 * t55 - 0.125 * sigma) / (rho * tau)
    t65 = 8.0 * t55 - sigma
    t69 = _pw3(t63, t65 / (8.0 * rho * tau), 1e-10)
    t70 = t69 * t69
    t71 = t70 * t70
    t72 = t54 * t71
    t75 = cbrtpi * pi
    t76 = t50 / 2.0
    t77 = 3.5 * t52
    t78 = 7.0 * t53
    t80 = t75 * (t49 + t76 - t51 - t77 - t78)
    t81 = t31 * rho
    t82 = t46 * tau
    t84 = t70 * t69
    t89 = t19 * t30 * rho
    t90 = cbrtpi * cbrtpi
    t91 = t90 * pi * pi
    t92 = t89 * t91
    t93 = cbrt3 * cbrt3
    t94 = t92 * t93
    t97 = t49 - 5.0 * t51 / 3.0 + 35.0 * t53 / 3.0
    t99 = t97 * t46 * t70
    t102 = t30 * t30
    t103 = t102 * rho
    t104 = pi**4
    t106 = t49 - t76 - t51 + t77 - t78
    t107 = t103 * t104 * t106
    t109 = tau * cbrt3 * t69
    t113 = t31 * t102 * t30
    t116 = t113 * cbrtpi * t104 * pi
    t117 = t49 - t50 + t51 - t52 + t53
    t120 = (
        108000.0 * t80 * t81 * t82 * t84
        + 29160.0 * t107 * t109
        + 6561.0 * t116 * t117
        + 30000.0 * t48 * t72
        + 48600.0 * t94 * t99
    )
    t121 = t81 * t75
    t124 = 9.0 * t121 + 10.0 * t109
    t129 = 1.0 - t120 / (t124**4)
    t130, t131, t132 = task_anu
    t134 = t91 * (t130 - t131 + t132)
    t138 = cbrt3 * t75
    t140 = t130 - 3.0 * t132
    t143 = 24.0 * t138 * t140 * t32
    t145 = t130 + t131 + t132
    t146 = sigma * t93 * t145
    t149 = 144.0 * t134 * t19 * t103 + (t143 + t146) * sigma
    t153 = 12.0 * t75 * t32 + cbrt3 * sigma
    t157 = t149 / (t153 * t153) - _TASK_H0X
    t161 = _TASK_H0X * t44 + t129 * t157 * (t44 ** _TASK_D)
    return -0.75 * t7 * t19 * t161


# ---------------------------------------------------------------------------------------
# TPSS (mgga_x_tpss.py, mgga_c_tpss.py)
# ---------------------------------------------------------------------------------------
@njit(**_NB)
def _tpss_x_energy(rho, sigma, tau):
    rho = np.maximum(rho, 1e-12)
    sigma = np.maximum(sigma, 0.0)
    tau = np.maximum(tau, 1e-14)

    b = 0.40
    c = 1.59096
    e = 1.537
    kappa = 0.804
    mu = 0.21951

    pi = np.pi
    kf2 = (3.0 * pi**2) ** (2.0 / 3.0) * rho ** (2.0 / 3.0)
    p = sigma / (4.0 * kf2 * rho**2 + 1e-30)
    tau_w = sigma / (8.0 * rho + 1e-30)
    z = np.minimum(tau_w / tau, 1.0)
    tau_unif = 0.3 * (3.0 * pi**2) ** (2.0 / 3.0) * rho ** (5.0 / 3.0)
    alpha = np.maximum((tau - tau_w) / (tau_unif + 1e-30), 0.0)

    qb = (9.0 / 20.0) * (alpha - 1.0) / np.sqrt(1.0 + b * alpha * (alpha - 1.0) + 1e-30) + 2.0 * p / 3.0
    z2 = z * z
    root_arg = 0.5 * ((3.0 * z / 5.0) ** 2 + p * p)
    x_num = (
        (10.0 / 81.0 + c * z2 / (1.0 + z2) ** 2) * p
        + 146.0 / 2025.0 * qb * qb
        - 73.0 / 405.0 * qb * np.sqrt(root_arg)
        + (10.0 / 81.0) ** 2 * p * p / kappa
        + 2.0 * np.sqrt(e) * (10.0 / 81.0) * (3.0 * z / 5.0) ** 2
        + e * mu * p**3
    )
    x = x_num / (1.0 + np.sqrt(e) * p) ** 2
    fx = 1.0 + kappa - kappa / (1.0 + x / kappa)
    ex_unif = -0.75 * (3.0 / pi) ** (1.0 / 3.0) * rho ** (1.0 / 3.0)
    return ex_unif * fx


@njit(**_NB)
def _pw92_ec(rs, polarized):
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
    sqrt_rs = np.sqrt(rs)
    q = 2.0 * A * (b1 * sqrt_rs + b2 * rs + b3 * rs * sqrt_rs + b4 * rs * rs)
    return -2.0 * A * (1.0 + a1 * rs) * np.log1p(1.0 / (q + 1e-30))


@njit(**_NB)
def _pbe_c_energy_spin(rho, sigma, zeta):
    rho = np.maximum(rho, 1e-12)
    sigma = np.maximum(sigma, 0.0)
    beta = 0.06672455060314922
    gamma = (1.0 - np.log(2.0)) / np.pi**2

    rs = (3.0 / (4.0 * np.pi * rho)) ** (1.0 / 3.0)
    ec_lda = _pw92_ec(rs, zeta > 0.5)
    phi = ((1.0 + zeta) ** (2.0 / 3.0) + (1.0 - zeta) ** (2.0 / 3.0)) / 2.0
    kf = (3.0 * np.pi**2 * rho) ** (1.0 / 3.0)
    ks = np.sqrt(4.0 * kf / np.pi)
    t = np.sqrt(sigma) / (2.0 * phi * ks * rho + 1e-30)

    phi3 = phi**3
    arg = -ec_lda / (gamma * phi3)
    arg = np.minimum(arg, 50.0)
    A = beta / (gamma * (np.exp(arg) - 1.0) + 1e-30)
    t2 = t * t
    At2 = A * t2
    h_arg = 1.0 + (beta / gamma) * t2 * (1.0 + At2) / (1.0 + At2 + At2 * At2)
    return ec_lda + gamma * phi3 * np.log(h_arg)


@njit(**_NB)
def _tpss_c_energy(rho, sigma, tau):
    rho = np.maximum(rho, 1e-12)
    sigma = np.maximum(sigma, 0.0)
    tau = np.maximum(tau, 1e-14)

    C = 0.53
    d = 2.8
    tau_w_over_tau = np.minimum(sigma / (8.0 * rho * tau + 1e-30), 1.0)
    tau2 = tau_w_over_tau * tau_w_over_tau

    ec_pbe = _pbe_c_energy_spin(rho, sigma, 0.0)
    ec_pbe_spin = _pbe_c_energy_spin(0.5 * rho, 0.25 * sigma, 1.0)
    ec_tilde = np.maximum(ec_pbe_spin, ec_pbe)
    ec_revpkzb = ec_pbe * (1.0 + C * tau2) - (1.0 + C) * tau2 * ec_tilde
    return ec_revpkzb * (1.0 + d * ec_revpkzb * tau_w_over_tau**3)


# ---------------------------------------------------------------------------------------
# M06-L (mgga_m06_l.py)
# ---------------------------------------------------------------------------------------
@njit(**_NB)
def _m06_l_x_energy(rho, sigma, tau):
    rho = np.maximum(rho, 1e-12)
    sigma = np.maximum(sigma, 0.0)
    tau = np.maximum(tau, 1e-14)

    cbrt2 = 2.0 ** (1.0 / 3.0)
    cbrt2_2 = cbrt2 * cbrt2
    cbrt6_2 = 6.0 ** (2.0 / 3.0)
    pi2_23 = (np.pi * np.pi) ** (2.0 / 3.0)

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

    poly = 0.0
    wpow = 1.0
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

    return -0.75 * (3.0 / np.pi) ** (1.0 / 3.0) * rho13 * (f_pw86 * poly + h1 + h2 + h3)


@njit(**_NB)
def _pw92_mod_unpolarized(rs):
    sqrt_rs = np.sqrt(rs)
    rs32 = rs * sqrt_rs
    rs2 = rs * rs
    q = 3.79785 * sqrt_rs + 0.8969 * rs + 0.204775 * rs32 + 0.123235 * rs2 / 4.0
    return -0.0621814 * (1.0 + 0.053425 * rs) * np.log(1.0 + 16.081979498692535067 / q)


@njit(**_NB)
def _pw92_mod_parallel(rs):
    sqrt_rs = np.sqrt(rs)
    rs32 = rs * sqrt_rs
    rs2 = rs * rs
    q = 7.05945 * sqrt_rs + 1.549425 * rs + 0.420775 * rs32 + 0.1562925 * rs2 / 4.0
    return -0.0310907 * (1.0 + 0.05137 * rs) * np.log(1.0 + 32.163958997385070134 / q)


@njit(**_NB)
def _m06_l_c_energy(rho, sigma, tau):
    rho = np.maximum(rho, 1e-12)
    sigma = np.maximum(sigma, 0.0)
    tau = np.maximum(tau, 1e-14)

    gamma_ss = 0.06
    gamma_ab = 0.0031
    alpha_ss = 0.00515088
    alpha_ab = 0.00304966
    fermi_d = 1e-10

    cbrt2 = 2.0 ** (1.0 / 3.0)
    cbrt2_2 = cbrt2 * cbrt2
    cbrt6_2 = 6.0 ** (2.0 / 3.0)
    pi2_23 = (np.pi * np.pi) ** (2.0 / 3.0)

    rho13 = rho ** (1.0 / 3.0)
    rho23 = rho13 * rho13
    rho83 = rho23 * rho * rho
    rho163 = rho ** (16.0 / 3.0)
    x = sigma * cbrt2_2 / rho83
    z = tau * cbrt2_2 / (rho23 * rho)
    cf = cbrt6_2 * pi2_23

    rs_ss = (96.0 / (np.pi * rho)) ** (1.0 / 3.0)
    rs_ab = (48.0 / (np.pi * rho)) ** (1.0 / 3.0)
    e_ss_lda = 0.5 * _pw92_mod_parallel(rs_ss)
    e_ab_lda = _pw92_mod_unpolarized(rs_ab) - 2.0 * e_ss_lda

    def gamma_series(coeffs, gamma, scale):
        u = 1.0 + scale * gamma * x
        y = scale * gamma * x / u
        return coeffs[0] + coeffs[1] * y + coeffs[2] * y * y + coeffs[3] * y**3 + coeffs[4] * y**4

    same_spin = gamma_series(_M06L_C_CSS, gamma_ss, 1.0)
    opposite_spin = gamma_series(_M06L_C_CAB, gamma_ab, 2.0)

    tau_w_over_tau = sigma / (8.0 * rho * tau)
    fermi = 1.0 - np.exp(-8.0 * tau * tau * cbrt2 / (rho ** (10.0 / 3.0) * fermi_d * fermi_d))
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


# ---------------------------------------------------------------------------------------
# PW91 correlation (gga_c_pw91.py): complex-step derivatives
# ---------------------------------------------------------------------------------------
@njit(**_NB)
def _gga_c_pw91_energy_density(rho, sigma):
    """LibXC-equivalent unpolarized PW91 correlation energy density."""
    sigma = np.asarray(sigma)

    third = 1.0 / 3.0
    cbrt2 = 2.0**third
    cbrt3 = 3.0**third
    cbrt4 = 4.0**third
    cbrt9 = 9.0**third
    invpi = 1.0 / np.pi

    t3 = invpi**third
    t4 = cbrt3 * t3
    t6 = cbrt4 * cbrt4
    t7 = rho**third
    t10 = t4 * t6 / t7
    t12 = 1.0 + 0.053425 * t10
    t13 = np.sqrt(t10)
    t16 = t10**1.5
    t18 = cbrt3 * cbrt3
    t19 = t3 * t3
    t20 = t18 * t19
    t21 = t7 * t7
    t24 = t20 * cbrt4 / t21
    t26 = 3.79785 * t13 + 0.8969 * t10 + 0.204775 * t16 + 0.123235 * t24
    t29 = 1.0 + 16.081824322151104822 / t26
    t32 = 0.062182 * t12 * np.log(t29)
    ec_lda = -t32

    pi2_cbrt = (np.pi * np.pi) ** third
    pi2_cbrt2 = pi2_cbrt * pi2_cbrt
    t61 = t18 * pi2_cbrt2
    t66 = 1.0 / pi2_cbrt
    t67 = t18 * t66
    rho2 = rho * rho
    t70 = 1.0 / (t7 * rho2)
    t72 = sigma * t70 * cbrt2
    t75 = 1.0 / t3
    t76 = t75 * cbrt4
    t77 = t18 * t76
    t83 = 1.0 / pi2_cbrt2
    t87 = np.exp(-128.97460341341234505 * ec_lda * cbrt3 * t83)
    t88 = t87 - 1.0
    t89 = 1.0 / t88
    t90 = t66 * t89
    sigma2 = sigma * sigma
    rho4 = rho2 * rho2
    t94 = 1.0 / (t21 * rho4)
    t95 = sigma2 * t94
    t97 = cbrt2 * cbrt2
    t101 = 1.0 / t19
    t102 = t101 * t6
    t103 = t97 * t102
    t106 = t72 * t77 / 96.0 + 0.0027166129655589868296 * t90 * t95 * t103
    t107 = cbrt3 * t66
    t109 = t107 * t89 * sigma
    t110 = t70 * cbrt2
    t112 = t75 * cbrt4
    t116 = t18 * t83
    t118 = 1.0 / (t88 * t88)
    t120 = t116 * t118 * sigma2
    t121 = t94 * t97
    t123 = t101 * t6
    t124 = t121 * t123
    t127 = (
        1.0
        + 0.086931614897887578546 * t109 * t110 * t112
        + 0.0075571056687546295931 * t120 * t124
    )
    t132 = 1.0 + 2.7818116767324025134 * t67 * t106 / t127
    h0 = 0.0025844881434903430496 * t61 * np.log(t132)

    t137 = invpi * pi2_cbrt
    t140 = 2.568 + 5.8165 * t10 + 0.00184725 * t24
    t143 = 1000.0 + 2180.75 * t10 + 118.0 * t24
    t146 = t140 / t143 - 0.0018535714285714285714
    t149 = t137 * t146 * sigma
    t152 = cbrt9 * cbrt9
    t156 = 1.0 / (t21 * rho2)
    t158 = sigma * cbrt2
    t162 = np.exp(-25.0 / 18.0 * invpi * cbrt4 * t152 * t3 * t156 * t158)
    h1 = t149 * (t110 * t76 * t162) / 2.0

    return ec_lda + h0 + h1


# ---------------------------------------------------------------------------------------
# Derivatives
# ---------------------------------------------------------------------------------------

@njit(**_NB)
def mgga_x_r2scan_point(rho, sigma, tau):
    """eps, vrho, vsigma, vtau of r2SCAN exchange (LibXC 497) at one point (finite differences as in the array version)."""
    eps = _r2scan_x_energy(rho, sigma, tau)
    hr = 1e-5 * max(abs(rho), 1e-3)
    hs = 1e-7 * max(abs(sigma), 1e-6)
    ht = 1e-5 * max(abs(tau), 1e-6)
    fp = (rho + hr) * _r2scan_x_energy(rho + hr, sigma, tau)
    rm = max(rho - hr, 1e-12)
    fm = rm * _r2scan_x_energy(rm, sigma, tau)
    vrho = (fp - fm) / ((rho + hr) - rm)
    sp = sigma + hs
    sm = max(sigma - hs, 0.0)
    vsigma = ((rho * _r2scan_x_energy(rho, sp, tau)) - (rho * _r2scan_x_energy(rho, sm, tau))) / (sp - sm + 1e-30)
    tp = tau + ht
    tm = max(tau - ht, 1e-14)
    vtau = ((rho * _r2scan_x_energy(rho, sigma, tp)) - (rho * _r2scan_x_energy(rho, sigma, tm))) / (tp - tm)
    if not math.isfinite(eps):
        eps = 0.0
    if not math.isfinite(vrho):
        vrho = 0.0
    if not math.isfinite(vsigma):
        vsigma = 0.0
    if not math.isfinite(vtau):
        vtau = 0.0
    return eps, vrho, vsigma, vtau


@njit(**_NB)
def mgga_c_r2scan_point(rho, sigma, tau):
    """eps, vrho, vsigma, vtau of r2SCAN correlation (498) at one point (finite differences as in the array version)."""
    eps = _r2scan_c_energy(rho, sigma, tau)
    hr = 1e-5 * max(abs(rho), 1e-3)
    hs = 1e-7 * max(abs(sigma), 1e-6)
    ht = 1e-5 * max(abs(tau), 1e-6)
    fp = (rho + hr) * _r2scan_c_energy(rho + hr, sigma, tau)
    rm = max(rho - hr, 1e-12)
    fm = rm * _r2scan_c_energy(rm, sigma, tau)
    vrho = (fp - fm) / ((rho + hr) - rm)
    sp = sigma + hs
    sm = max(sigma - hs, 0.0)
    vsigma = ((rho * _r2scan_c_energy(rho, sp, tau)) - (rho * _r2scan_c_energy(rho, sm, tau))) / (sp - sm + 1e-30)
    tp = tau + ht
    tm = max(tau - ht, 1e-14)
    vtau = ((rho * _r2scan_c_energy(rho, sigma, tp)) - (rho * _r2scan_c_energy(rho, sigma, tm))) / (tp - tm)
    if not math.isfinite(eps):
        eps = 0.0
    if not math.isfinite(vrho):
        vrho = 0.0
    if not math.isfinite(vsigma):
        vsigma = 0.0
    if not math.isfinite(vtau):
        vtau = 0.0
    return eps, vrho, vsigma, vtau


@njit(**_NB)
def mgga_x_task_point(rho, sigma, tau):
    """eps, vrho, vsigma, vtau of TASK exchange (707) at one point (finite differences as in the array version)."""
    eps = _task_x_energy(rho, sigma, tau)
    hr = 1e-5 * max(abs(rho), 1e-3)
    hs = 1e-7 * max(abs(sigma), 1e-6)
    ht = 1e-5 * max(abs(tau), 1e-6)
    fp = (rho + hr) * _task_x_energy(rho + hr, sigma, tau)
    rm = max(rho - hr, 1e-12)
    fm = rm * _task_x_energy(rm, sigma, tau)
    vrho = (fp - fm) / ((rho + hr) - rm)
    sp = sigma + hs
    sm = max(sigma - hs, 0.0)
    vsigma = ((rho * _task_x_energy(rho, sp, tau)) - (rho * _task_x_energy(rho, sm, tau))) / (sp - sm + 1e-30)
    tp = tau + ht
    tm = max(tau - ht, 1e-14)
    vtau = ((rho * _task_x_energy(rho, sigma, tp)) - (rho * _task_x_energy(rho, sigma, tm))) / (tp - tm)
    if not math.isfinite(eps):
        eps = 0.0
    if not math.isfinite(vrho):
        vrho = 0.0
    if not math.isfinite(vsigma):
        vsigma = 0.0
    if not math.isfinite(vtau):
        vtau = 0.0
    return eps, vrho, vsigma, vtau


@njit(**_NB)
def mgga_x_tpss_point(rho, sigma, tau):
    """eps, vrho, vsigma, vtau of TPSS exchange (202) at one point (finite differences as in the array version)."""
    eps = _tpss_x_energy(rho, sigma, tau)
    hr = 1e-5 * max(abs(rho), 1e-3)
    hs = 1e-5 * max(abs(sigma), 1e-6)
    ht = 1e-5 * max(abs(tau), 1e-6)
    fp = (rho + hr) * _tpss_x_energy(rho + hr, sigma, tau)
    rm = max(rho - hr, 1e-12)
    fm = rm * _tpss_x_energy(rm, sigma, tau)
    vrho = (fp - fm) / ((rho + hr) - rm)
    sp = sigma + hs
    sm = max(sigma - hs, 0.0)
    vsigma = ((rho * _tpss_x_energy(rho, sp, tau)) - (rho * _tpss_x_energy(rho, sm, tau))) / (sp - sm + 1e-30)
    tp = tau + ht
    tm = max(tau - ht, 1e-14)
    vtau = ((rho * _tpss_x_energy(rho, sigma, tp)) - (rho * _tpss_x_energy(rho, sigma, tm))) / (tp - tm)
    if not math.isfinite(eps):
        eps = 0.0
    if not math.isfinite(vrho):
        vrho = 0.0
    if not math.isfinite(vsigma):
        vsigma = 0.0
    if not math.isfinite(vtau):
        vtau = 0.0
    return eps, vrho, vsigma, vtau


@njit(**_NB)
def mgga_c_tpss_point(rho, sigma, tau):
    """eps, vrho, vsigma, vtau of TPSS correlation (231) at one point (finite differences as in the array version)."""
    eps = _tpss_c_energy(rho, sigma, tau)
    hr = 1e-5 * max(abs(rho), 1e-3)
    hs = 1e-5 * max(abs(sigma), 1e-6)
    ht = 1e-5 * max(abs(tau), 1e-6)
    fp = (rho + hr) * _tpss_c_energy(rho + hr, sigma, tau)
    rm = max(rho - hr, 1e-12)
    fm = rm * _tpss_c_energy(rm, sigma, tau)
    vrho = (fp - fm) / ((rho + hr) - rm)
    sp = sigma + hs
    sm = max(sigma - hs, 0.0)
    vsigma = ((rho * _tpss_c_energy(rho, sp, tau)) - (rho * _tpss_c_energy(rho, sm, tau))) / (sp - sm + 1e-30)
    tp = tau + ht
    tm = max(tau - ht, 1e-14)
    vtau = ((rho * _tpss_c_energy(rho, sigma, tp)) - (rho * _tpss_c_energy(rho, sigma, tm))) / (tp - tm)
    if not math.isfinite(eps):
        eps = 0.0
    if not math.isfinite(vrho):
        vrho = 0.0
    if not math.isfinite(vsigma):
        vsigma = 0.0
    if not math.isfinite(vtau):
        vtau = 0.0
    return eps, vrho, vsigma, vtau


@njit(**_NB)
def mgga_x_m06_l_point(rho, sigma, tau):
    """eps, vrho, vsigma, vtau of M06-L exchange (203) at one point (finite differences as in the array version)."""
    eps = _m06_l_x_energy(rho, sigma, tau)
    hr = 1e-5 * max(abs(rho), 1e-3)
    hs = 1e-5 * max(abs(sigma), 1e-6)
    ht = 1e-5 * max(abs(tau), 1e-6)
    fp = (rho + hr) * _m06_l_x_energy(rho + hr, sigma, tau)
    rm = max(rho - hr, 1e-12)
    fm = rm * _m06_l_x_energy(rm, sigma, tau)
    vrho = (fp - fm) / ((rho + hr) - rm)
    sp = sigma + hs
    sm = max(sigma - hs, 0.0)
    vsigma = ((rho * _m06_l_x_energy(rho, sp, tau)) - (rho * _m06_l_x_energy(rho, sm, tau))) / (sp - sm + 1e-30)
    tp = tau + ht
    tm = max(tau - ht, 1e-14)
    vtau = ((rho * _m06_l_x_energy(rho, sigma, tp)) - (rho * _m06_l_x_energy(rho, sigma, tm))) / (tp - tm)
    if not math.isfinite(eps):
        eps = 0.0
    if not math.isfinite(vrho):
        vrho = 0.0
    if not math.isfinite(vsigma):
        vsigma = 0.0
    if not math.isfinite(vtau):
        vtau = 0.0
    return eps, vrho, vsigma, vtau


@njit(**_NB)
def mgga_c_m06_l_point(rho, sigma, tau):
    """eps, vrho, vsigma, vtau of M06-L correlation (233) at one point (finite differences as in the array version)."""
    eps = _m06_l_c_energy(rho, sigma, tau)
    hr = 1e-5 * max(abs(rho), 1e-3)
    hs = 1e-5 * max(abs(sigma), 1e-6)
    ht = 1e-5 * max(abs(tau), 1e-6)
    fp = (rho + hr) * _m06_l_c_energy(rho + hr, sigma, tau)
    rm = max(rho - hr, 1e-12)
    fm = rm * _m06_l_c_energy(rm, sigma, tau)
    vrho = (fp - fm) / ((rho + hr) - rm)
    sp = sigma + hs
    sm = max(sigma - hs, 0.0)
    vsigma = ((rho * _m06_l_c_energy(rho, sp, tau)) - (rho * _m06_l_c_energy(rho, sm, tau))) / (sp - sm + 1e-30)
    tp = tau + ht
    tm = max(tau - ht, 1e-14)
    vtau = ((rho * _m06_l_c_energy(rho, sigma, tp)) - (rho * _m06_l_c_energy(rho, sigma, tm))) / (tp - tm)
    if not math.isfinite(eps):
        eps = 0.0
    if not math.isfinite(vrho):
        vrho = 0.0
    if not math.isfinite(vsigma):
        vsigma = 0.0
    if not math.isfinite(vtau):
        vtau = 0.0
    return eps, vrho, vsigma, vtau


@njit(**_NB)
def gga_c_pw91_point(rho, sigma):
    """ec, vrho, vsigma of PW91 correlation (134) at one point (complex-step derivatives)."""
    rho = max(rho, _PW91_RHO_CUTOFF)
    h = _PW91_COMPLEX_STEP
    ec = _gga_c_pw91_energy_density(rho, sigma)
    vrho = ((rho + 1j * h) * _gga_c_pw91_energy_density(rho + 1j * h, sigma)).imag / h
    vsigma = (rho * _gga_c_pw91_energy_density(rho, sigma + 1j * h)).imag / h
    if not math.isfinite(ec):
        ec = 0.0
    if not math.isfinite(vrho):
        vrho = 0.0
    if not math.isfinite(vsigma):
        vsigma = 0.0
    return ec, vrho, vsigma


POINTWISE_IDS = frozenset((134, 202, 203, 231, 233, 497, 498, 707))
"""LibXC IDs of the functionals this module evaluates."""


@njit(**_NB)
def eval_pointwise_mgga(fid, c, rho, sigma, tau, e, vrho, vsigma, vtau):
    """Add ``c`` times the meta-GGA ``fid`` (202, 203, 231, 233, 497, 498, 707) at every point."""
    for i in range(rho.shape[0]):
        r = rho[i]
        s = sigma[i]
        t = tau[i]
        if fid == 497:
            a, b, d, f = mgga_x_r2scan_point(r, s, t)
        elif fid == 498:
            a, b, d, f = mgga_c_r2scan_point(r, s, t)
        elif fid == 202:
            a, b, d, f = mgga_x_tpss_point(r, s, t)
        elif fid == 231:
            a, b, d, f = mgga_c_tpss_point(r, s, t)
        elif fid == 203:
            a, b, d, f = mgga_x_m06_l_point(r, s, t)
        elif fid == 233:
            a, b, d, f = mgga_c_m06_l_point(r, s, t)
        elif fid == 707:
            a, b, d, f = mgga_x_task_point(r, s, t)
        else:
            a = 0.0
            b = 0.0
            d = 0.0
            f = 0.0
        e[i] += c * a
        vrho[i] += c * b
        vsigma[i] += c * d
        vtau[i] += c * f


@njit(**_NB)
def eval_pointwise_gga(fid, c, rho, sigma, e, vrho, vsigma):
    """Add ``c`` times the GGA ``fid`` (134) at every point."""
    for i in range(rho.shape[0]):
        if fid == 134:
            a, b, d = gga_c_pw91_point(rho[i], sigma[i])
        else:
            a = 0.0
            b = 0.0
            d = 0.0
        e[i] += c * a
        vrho[i] += c * b
        vsigma[i] += c * d
