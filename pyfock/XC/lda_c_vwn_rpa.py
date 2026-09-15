import numpy as np
try:
    import cupy as cp
    from cupy import fuse
except Exception:
    cp = None
    def fuse(kernel_name):
        def decorator(func):
            return func
        return decorator

# Vosko-Wilk-Nusair correlation, parametrization of the RPA correlation energies
# (LibXC ID 8, LDA_C_VWN_RPA). Same functional form as lda_c_vwn (ID 7, the
# Ceperley-Alder fit); only the four parameters differ. This is the variant used
# in the original B3LYP of Gaussian and in LibXC's / PySCF's HYB_GGA_XC_B3LYP.


def lda_c_vwn_rpa_(rho):
    """
    Spin-unpolarized VWN correlation energy per particle and potential with the
    RPA parameters (LibXC ID 8). Returns ``(ec, vc)``.
    """
    rho = np.maximum(rho, 1e-12)

    a = 0.0310907
    b = 13.0720
    c = 42.7198
    x0 = -0.409286
    pi34 = (3 / (4 * np.pi))**(1 / 3)
    rs = pi34 * np.power(rho, -1 / 3)
    q = np.sqrt(4 * c - b * b)
    f1 = 2 * b / q
    f2 = b * x0 / (x0 * x0 + b * x0 + c)
    f3 = 2 * (2 * x0 + b) / q
    rs12 = np.sqrt(rs)
    fx = rs + b * rs12 + c
    qx = np.arctan(q / (2 * rs12 + b))
    ec = a * (np.log(rs / fx) + f1 * qx - f2 * (np.log((rs12 - x0)**2 / fx) + f3 * qx))
    tx = 2 * rs12 + b
    tt = tx * tx + q * q
    vc = ec - rs12 * a / 6 * (2 / rs12 - tx / fx - 4 * b / tt -
                              f2 * (2 / (rs12 - x0) - tx / fx - 4 * (2 * x0 + b) / tt))
    return ec, vc


def lda_c_vwn_rpa(rho):
    """LDA_C_VWN_RPA (LibXC ID 8): correlation energy per particle and potential."""
    return lda_c_vwn_rpa_(rho)


@fuse(kernel_name='lda_c_vwn_rpa_cupy_')
def lda_c_vwn_rpa_cupy_(rho):
    rho = cp.maximum(rho, 1e-12)
    # cupy.fuse infers the type of bare Python scalars inside the kernel and may
    # evaluate these constants in single precision; np.float64 pins them (see
    # the CPU implementation above, which is the reference).
    a = np.float64(0.0310907)
    b = np.float64(13.0720)
    c = np.float64(42.7198)
    x0 = np.float64(-0.409286)
    pi34 = np.float64((3 / (4 * np.pi))**(1 / 3))
    rs = pi34 * cp.power(rho, -1 / 3)
    q = np.float64(np.sqrt(4 * c - b * b))
    f1 = np.float64(2 * b / q)
    f2 = np.float64(b * x0 / (x0 * x0 + b * x0 + c))
    f3 = np.float64(2 * (2 * x0 + b) / q)
    rs12 = cp.sqrt(rs)
    fx = rs + b * rs12 + c
    qx = cp.arctan(q / (2 * rs12 + b))
    ec = a * (cp.log(rs / fx) + f1 * qx - f2 * (cp.log((rs12 - x0)**2 / fx) + f3 * qx))
    tx = 2 * rs12 + b
    tt = tx * tx + q * q
    vc = ec - rs12 * a / 6 * (2 / rs12 - tx / fx - 4 * b / tt -
                              f2 * (2 / (rs12 - x0) - tx / fx - 4 * (2 * x0 + b) / tt))
    return ec, vc


def lda_c_vwn_rpa_cupy(rho):
    """GPU version of LDA_C_VWN_RPA; NaNs (from rho = 0) are replaced by zeros."""
    ec, vc = lda_c_vwn_rpa_cupy_(rho)
    vc[cp.isnan(vc)] = 0
    ec[cp.isnan(ec)] = 0
    return ec, vc
