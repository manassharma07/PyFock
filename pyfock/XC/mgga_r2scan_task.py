try:
    import cupy as cp
except Exception:
    cp = None
import numpy as np


_R2SCAN_X_PARAMS = (0.667, 0.8, 1.24, 0.065, 0.001, 0.361)
_TASK_C = 4.9479
_TASK_D = 10.0
_TASK_H0X = 1.174
_TASK_ANU = (0.938719, -0.076371, -0.0150899)
_TASK_BNU = (-0.628591, -2.10315, -0.5, 0.103153, 0.128591)


def _cbrt(x, xp):
    return x ** (1.0 / 3.0)


def _pw3(cond, a, b, xp):
    return xp.where(cond, a, b)


def _pw5(cond1, a, cond2, b, c, xp):
    return xp.where(cond1, a, xp.where(cond2, b, c))


def _finite_difference_vxc(energy_func, rho, sigma, tau, xp):
    eps = energy_func(rho, sigma, tau)

    hr = 1e-5 * xp.maximum(xp.abs(rho), 1e-3)
    hs = 1e-7 * xp.maximum(xp.abs(sigma), 1e-6)
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


def _r2scan_x_energy(rho, sigma, tau, xp):
    c1, c2, d, k1, eta, dp2 = _R2SCAN_X_PARAMS
    rho = xp.maximum(rho, 1e-12)
    sigma = xp.maximum(sigma, 1e-30)
    tau = xp.maximum(tau, 1e-14)

    pi = xp.pi
    cbrt2 = 2.0 ** (1.0 / 3.0)
    cbrt3 = 3.0 ** (1.0 / 3.0)
    cbrt6 = 6.0 ** (1.0 / 3.0)
    cbrtpi = pi ** (1.0 / 3.0)

    t7 = cbrt3 / cbrtpi
    t20 = _cbrt(rho, xp)
    t22 = 20.0 / 27.0 + 5.0 * eta / 3.0
    t24 = cbrt6 * cbrt6
    t25 = pi * pi
    t26 = _cbrt(t25, xp)
    t29 = t24 / (t26 * t25)
    t30 = sigma * sigma
    t38 = cbrt2 / (t20 * rho**5)
    t45 = xp.exp(-t29 * t30 * t38 / (288.0 * dp2**4))
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

    t83 = _pw3(0.0 < t80, 0.0, t80, xp)
    t88 = xp.exp(-c1 * t83 / (1.0 - t83))
    t91 = _pw3(t90, 2.5, t80, xp)
    t93 = t91 * t91
    t95 = t93 * t91
    t97 = t93 * t93
    t99 = t97 * t91
    t101 = t97 * t93
    t106 = _pw3(t90, t80, 2.5, xp)
    t110 = xp.exp(c2 / (1.0 - t106))
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
        -d * t110,
        xp,
    )
    t115 = t112 * (0.174 - t65) + t65 + 1.0

    t125 = t24 / t26 * xp.sqrt(sigma) * cbrt2 / (t20 * rho)
    t126 = xp.sqrt(t125)
    t131 = 1.0 - xp.exp(-9.8958 * xp.sqrt(3.0) / t126)
    return -0.75 * t7 * t20 * t115 * t131


def _r2scan_c_energy(rho, sigma, tau, xp):
    eta = 0.001
    rho = xp.maximum(rho, 1e-12)
    sigma = xp.maximum(sigma, 0.0)
    tau = xp.maximum(tau, 1e-14)

    pi = xp.pi
    cbrt2 = 2.0 ** (1.0 / 3.0)
    cbrt3 = 3.0 ** (1.0 / 3.0)
    cbrt4 = 4.0 ** (1.0 / 3.0)
    cbrt6 = 6.0 ** (1.0 / 3.0)

    t4 = _cbrt(1.0 / pi, xp)
    t5 = cbrt3 * t4
    t7 = cbrt4 * cbrt4
    t8 = _cbrt(rho, xp)
    t10 = t7 / t8
    t11 = t5 * t10
    t13 = 1.0 + 0.053425 * t11
    t14 = xp.sqrt(t11)
    t16 = 0.8969 * t11
    t17 = t11 ** 1.5
    t18 = 0.204775 * t17
    t21 = cbrt3 * cbrt3 * t4 * t4
    t22 = t8 * t8
    t25 = t21 * cbrt4 / t22
    t26 = 0.123235 * t25
    t27 = 3.79785 * t14 + t16 + t18 + t26
    t31 = xp.log(1.0 + 16.081979498692535067 / t27)
    t33 = 0.0621814 * t13 * t31
    t60 = 1.0 - xp.log(2.0)
    t61 = pi * pi
    t63 = t60 / t61
    t69 = 1.0 / t60
    t74 = xp.exp(t33 * t69 * t61)
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
    t126 = xp.log(1.0 + 16.081979498692535067 / t122)
    t128 = t122 * t122
    t130 = t13 / t128
    t132 = xp.sqrt(t11)
    t135 = 3.79785 * t116 + 3.5876 + 1.22865 * t132 + 0.24647 * t11
    t136 = 1.0 / (1.0 + 16.081979498692535067 / t122)
    t137 = t135 * t136
    t141 = 2.58925 * t104 + 0.905775 * t11 + 0.1100325 * t17 + 0.1241775 * t25
    t144 = 1.0 + 29.608749977793437516 / t141
    t145 = xp.log(t144)
    t149 = t141 * t141
    t154 = 5.1785 * t116 + 3.6231 + 0.660195 * t132 + 0.248355 * t11
    t157 = t154 / (t149 * t144)
    t160 = 0.0285764 * t114 * t118 + 0.01328816518 * t126 - t130 * t137
    t165 = 1.0 + 0.04445 * t14 + t106
    t166 = 1.0 / t165
    t172 = 5.0 * t5 * t10 * t160 - 45.0 * eta * (-0.0285764 * t166 + t33)
    t174 = t101 * t102 * t172
    t176 = _cbrt(t61, xp)
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
    t198 = xp.exp(-0.20444604078896369094 * t188 * cbrt2 * t189 * t194)
    t200 = t181 * t184 * t198
    t203 = 1.0 + 0.027439371595564631661 * t81 * sigma * t84 * cbrt2 * t95 + 0.043341108700271342816 * t174 * t200
    t204 = t203 ** 0.25
    t211 = t63 * xp.log(t75 * (1.0 - 1.0 / t204) + 1.0)
    t216 = tau / (t22 * rho) - t184 / 8.0
    t223 = 0.15 * t185 * t177 * cbrt2 + eta * sigma * t183 / 8.0
    t225 = t216 / t223
    t226 = t225 <= 0.0
    t234 = t225 <= 2.5
    t235 = 2.5 < t225
    t228 = _pw3(0.0 < t225, 0.0, t225, xp)
    t233 = xp.exp(-0.64 * t228 / (1.0 - t228))
    t236 = _pw3(t235, 2.5, t225, xp)
    t238 = t236 * t236
    t240 = t238 * t236
    t242 = t238 * t238
    t244 = t242 * t236
    t246 = t242 * t238
    t251 = _pw3(t235, t225, 2.5, xp)
    t255 = xp.exp(1.5 / (1.0 - t251))
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
        -0.7 * t255,
        xp,
    )
    t261 = xp.exp(t166) - 1.0
    t266 = 1.0 + 0.021337642104376358333 * t179 * t180 * sigma * t183
    t269 = 1.0 - 1.0 / (t266 ** 0.25)
    t272 = xp.log(t261 * t269 + 1.0)
    t276 = -0.0285764 * t166 + 0.0285764 * t272 + t33 - t211
    return -t33 + t211 + t257 * t276


def _task_x_energy(rho, sigma, tau, xp):
    rho = xp.maximum(rho, 1e-12)
    sigma = xp.maximum(sigma, 1e-30)
    tau = xp.maximum(tau, 1e-14)

    pi = xp.pi
    cbrt2 = 2.0 ** (1.0 / 3.0)
    cbrt3 = 3.0 ** (1.0 / 3.0)
    cbrt6 = 6.0 ** (1.0 / 3.0)
    cbrtpi = pi ** (1.0 / 3.0)
    task_bnu = _TASK_BNU
    task_anu = _TASK_ANU

    t7 = cbrt3 / cbrtpi
    t19 = _cbrt(rho, xp)
    t23 = _cbrt(pi * pi, xp)
    t24 = t23 * t23
    t28 = cbrt2 * cbrt2
    t30 = rho * rho
    t31 = t19 * t19
    t32 = t31 * t30
    t36 = cbrt6 / t24 * sigma * t28 / t32 / 24.0
    t37 = 0.0 < t36
    t38 = _pw3(t37, t36, 0.0, xp)
    t42 = xp.exp(-_TASK_C / (t38 ** 0.25))
    t44 = _pw3(t37, 1.0 - t42, 0.0, xp)
    t46 = tau * tau
    t47 = t46 * t46
    t48 = t47 * cbrt3
    t49, t50, t51, t52, t53 = task_bnu
    t54 = t49 + t50 + t51 + t52 + t53
    t55 = rho * tau
    t63 = 0.0 < (0.9999999999 * t55 - 0.125 * sigma) / (rho * tau)
    t65 = 8.0 * t55 - sigma
    t69 = _pw3(t63, t65 / (8.0 * rho * tau), 1e-10, xp)
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
    t161 = _TASK_H0X * t44 + t129 * t157 * xp.power(t44, _TASK_D)
    return -0.75 * t7 * t19 * t161


def mgga_x_r2scan(rho, sigma, tau):
    rho = np.asarray(rho)
    sigma = np.asarray(sigma)
    tau = np.asarray(tau)
    return _finite_difference_vxc(lambda r, s, t: _r2scan_x_energy(r, s, t, np), rho, sigma, tau, np)


def mgga_x_r2scan_cupy(rho, sigma, tau):
    if cp is None:
        raise ImportError("CuPy is required for mgga_x_r2scan_cupy")
    return _finite_difference_vxc(lambda r, s, t: _r2scan_x_energy(r, s, t, cp), rho, sigma, tau, cp)


def mgga_c_r2scan(rho, sigma, tau):
    rho = np.asarray(rho)
    sigma = np.asarray(sigma)
    tau = np.asarray(tau)
    return _finite_difference_vxc(lambda r, s, t: _r2scan_c_energy(r, s, t, np), rho, sigma, tau, np)


def mgga_c_r2scan_cupy(rho, sigma, tau):
    if cp is None:
        raise ImportError("CuPy is required for mgga_c_r2scan_cupy")
    return _finite_difference_vxc(lambda r, s, t: _r2scan_c_energy(r, s, t, cp), rho, sigma, tau, cp)


def mgga_x_task(rho, sigma, tau):
    rho = np.asarray(rho)
    sigma = np.asarray(sigma)
    tau = np.asarray(tau)
    return _finite_difference_vxc(lambda r, s, t: _task_x_energy(r, s, t, np), rho, sigma, tau, np)


def mgga_x_task_cupy(rho, sigma, tau):
    if cp is None:
        raise ImportError("CuPy is required for mgga_x_task_cupy")
    return _finite_difference_vxc(lambda r, s, t: _task_x_energy(r, s, t, cp), rho, sigma, tau, cp)
