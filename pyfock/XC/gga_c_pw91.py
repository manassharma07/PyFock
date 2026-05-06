try:
    import cupy as cp
except Exception:
    cp = None

import numpy as np


_RHO_CUTOFF = 1e-12
_COMPLEX_STEP = 1e-30


def _sanitize_rho(rho, xp):
    if xp is np:
        rho = np.asarray(rho)
        if np.iscomplexobj(rho):
            return np.where(
                np.real(rho) > _RHO_CUTOFF,
                rho,
                _RHO_CUTOFF + 1j * np.imag(rho),
            )
        return np.maximum(rho, _RHO_CUTOFF)

    rho = xp.asarray(rho)
    if rho.dtype.kind == "c":
        return xp.where(
            xp.real(rho) > _RHO_CUTOFF,
            rho,
            _RHO_CUTOFF + 1j * xp.imag(rho),
        )
    return xp.maximum(rho, _RHO_CUTOFF)


def _finite_or_zero(values, xp):
    if xp is np:
        return np.where(np.isfinite(values), values, 0.0)
    return xp.where(xp.isfinite(values), values, 0.0)


def _gga_c_pw91_energy_density(rho, sigma, xp):
    """LibXC-equivalent unpolarized PW91 correlation energy density."""
    rho = _sanitize_rho(rho, xp)
    sigma = xp.asarray(sigma)

    third = 1.0 / 3.0
    cbrt2 = 2.0**third
    cbrt3 = 3.0**third
    cbrt4 = 4.0**third
    cbrt9 = 9.0**third
    invpi = 1.0 / xp.pi

    t3 = invpi**third
    t4 = cbrt3 * t3
    t6 = cbrt4 * cbrt4
    t7 = rho**third
    t10 = t4 * t6 / t7
    t12 = 1.0 + 0.053425 * t10
    t13 = xp.sqrt(t10)
    t16 = t10**1.5
    t18 = cbrt3 * cbrt3
    t19 = t3 * t3
    t20 = t18 * t19
    t21 = t7 * t7
    t24 = t20 * cbrt4 / t21
    t26 = 3.79785 * t13 + 0.8969 * t10 + 0.204775 * t16 + 0.123235 * t24
    t29 = 1.0 + 16.081824322151104822 / t26
    t32 = 0.062182 * t12 * xp.log(t29)
    ec_lda = -t32

    pi2_cbrt = (xp.pi * xp.pi) ** third
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
    t87 = xp.exp(-128.97460341341234505 * ec_lda * cbrt3 * t83)
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
    h0 = 0.0025844881434903430496 * t61 * xp.log(t132)

    t137 = invpi * pi2_cbrt
    t140 = 2.568 + 5.8165 * t10 + 0.00184725 * t24
    t143 = 1000.0 + 2180.75 * t10 + 118.0 * t24
    t146 = t140 / t143 - 0.0018535714285714285714
    t149 = t137 * t146 * sigma
    t152 = cbrt9 * cbrt9
    t156 = 1.0 / (t21 * rho2)
    t158 = sigma * cbrt2
    t162 = xp.exp(-25.0 / 18.0 * invpi * cbrt4 * t152 * t3 * t156 * t158)
    h1 = t149 * (t110 * t76 * t162) / 2.0

    return ec_lda + h0 + h1


def gga_c_pw91(rho, sigma):
    """
    Perdew-Wang 91 correlation functional (spin-unpolarized).

    Returns the same ``zk``, ``vrho``, and ``vsigma`` convention as LibXC.
    """
    rho = np.maximum(np.asarray(rho), _RHO_CUTOFF)
    sigma = np.asarray(sigma)

    ec = np.real(_gga_c_pw91_energy_density(rho, sigma, np))

    h = _COMPLEX_STEP
    rho_complex = rho.astype(np.complex128) + 1j * h
    sigma_complex = sigma.astype(np.complex128) + 1j * h

    vrho = np.imag(
        rho_complex * _gga_c_pw91_energy_density(rho_complex, sigma, np)
    ) / h
    vsigma = np.imag(
        rho * _gga_c_pw91_energy_density(rho, sigma_complex, np)
    ) / h

    return (
        _finite_or_zero(ec, np),
        _finite_or_zero(vrho, np),
        _finite_or_zero(vsigma, np),
    )


def gga_c_pw91_cupy(rho, sigma):
    """
    CuPy version of the Perdew-Wang 91 correlation functional.
    """
    if cp is None:
        raise ImportError("CuPy is required for gga_c_pw91_cupy")

    rho = cp.maximum(cp.asarray(rho), _RHO_CUTOFF)
    sigma = cp.asarray(sigma)

    ec = cp.real(_gga_c_pw91_energy_density(rho, sigma, cp))

    h = _COMPLEX_STEP
    rho_complex = rho.astype(cp.complex128) + 1j * h
    sigma_complex = sigma.astype(cp.complex128) + 1j * h

    vrho = cp.imag(
        rho_complex * _gga_c_pw91_energy_density(rho_complex, sigma, cp)
    ) / h
    vsigma = cp.imag(
        rho * _gga_c_pw91_energy_density(rho, sigma_complex, cp)
    ) / h

    return (
        _finite_or_zero(ec, cp),
        _finite_or_zero(vrho, cp),
        _finite_or_zero(vsigma, cp),
    )
