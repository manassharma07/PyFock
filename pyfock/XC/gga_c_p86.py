try:
    import cupy as cp
    from cupy import fuse
except Exception as e:
    cp = None
    def fuse(kernel_name):
        def decorator(func):
            return func
        return decorator
import numpy as np
from pyfock.XC.lda_c_pz import lda_c_pz_, lda_c_pz_cupy_
from numba import njit


@njit(cache=True, fastmath=True, error_model="numpy", nogil=True, inline='always')
def gga_c_p86_(rho, sigma):
    """
    Perdew 86 GGA correlation functional (spin-unpolarized).
    Corresponds to GGA_C_P86 with ID 132 in LibXC.

    Reference: J. P. Perdew, Phys. Rev. B 33, 8822 (1986).
              Erratum: Phys. Rev. B 34, 7406 (1986).

    Eq. (8): E_c = int n*ec_LDA dr + int d^{-1} exp(-Phi) C(n) |grad n|^2 / n^{4/3} dr

    Eq. (6): C(n) = 0.001667 + (0.002568 + alpha*rs + beta*rs^2)
                                / (1 + gamma*rs + delta*rs^2 + 1e4*beta*rs^3)
             alpha=0.023266, beta=7.389e-6, gamma=8.723, delta=0.472

    Eq. (9): Phi = 1.745 * f_tilde * C(inf)/C(n) * |grad n| / n^{7/6}
             f_tilde = 0.11, C(inf) = 0.001667

    Eq. (4): d = 2^{1/3} * [(1+zeta)/2)^{5/3} + ((1-zeta)/2)^{5/3}]^{1/2}
             For zeta=0 (unpolarized): d = 1

    Parameters
    ----------
    rho : ndarray
        Electron density array.
    sigma : ndarray
        Squared gradient of density (nabla rho . nabla rho).

    Returns
    -------
    ec : ndarray
        Correlation energy density per particle.
    vc : ndarray
        Correlation potential d(n*ec)/dn.
    vsigma : ndarray
        d(n*ec)/d(sigma).
    """
    rho = np.maximum(rho, 1e-12)

    ec_lda, vc_lda = lda_c_pz_(rho)

    pi34 = (3.0 / (4.0 * np.pi)) ** (1.0 / 3.0)
    rs = pi34 * rho ** (-1.0 / 3.0)

    # C(rs) from Eq. (6)
    Cinf = 0.001667
    alpha_c = 0.023266
    beta_c = 7.389e-6
    gamma_c = 8.723
    delta_c = 0.472

    rs2 = rs * rs
    rs3 = rs * rs2

    num_C = 0.002568 + alpha_c * rs + beta_c * rs2
    den_C = 1.0 + gamma_c * rs + delta_c * rs2 + 1e4 * beta_c * rs3
    Crs = Cinf + num_C / den_C

    d_factor = 1.0

    norm_dn = np.sqrt(sigma)

    f_tilde = 0.11
    prefac_Phi = 1.745 * f_tilde  # = 0.19195

    rho76 = rho ** (7.0 / 6.0)
    rho43 = rho ** (4.0 / 3.0)

    Phi = prefac_Phi * Cinf / Crs * norm_dn / rho76

    exp_phi = np.exp(-Phi)

    

    delta_ec = Crs * sigma * exp_phi / (d_factor * rho43 * rho)

    # Total ec
    ec = ec_lda + delta_ec

    
    drs_dn = -rs / (3.0 * rho)

    dnum_drs = alpha_c + 2.0 * beta_c * rs
    dden_drs = gamma_c + 2.0 * delta_c * rs + 3.0e4 * beta_c * rs2
    dCrs_drs = (dnum_drs * den_C - num_C * dden_drs) / (den_C * den_C)
    dCrs_dn = dCrs_drs * drs_dn

    dPhi_dn = Phi * (-dCrs_dn / Crs - 7.0 / 6.0 / rho)

    F = delta_ec * rho  
    dF_dn = F * (dCrs_dn / Crs - dPhi_dn - 4.0 / 3.0 / rho)

    vc = vc_lda + delta_ec + dF_dn


    vsigmac = Crs * exp_phi / (d_factor * rho43) * (1.0 - Phi / 2.0)

    vsigma = 0.5 * vsigmac

    return ec, vc, vsigma


@njit(cache=True, fastmath=True, error_model="numpy", nogil=True, inline='always')
def gga_c_p86(rho, sigma):
    """
    Perdew 86 correlation functional with NaN handling.
    Corresponds to GGA_C_P86 with ID 132 in LibXC.
    Reference: J. P. Perdew, Phys. Rev. B 33, 8822 (1986).
    """
    ec, vc, vsigma = gga_c_p86_(rho, sigma)
    vsigma[np.isnan(vsigma)] = 0
    vc[np.isnan(vc)] = 0
    ec[np.isnan(ec)] = 0
    return ec, vc, vsigma


def gga_c_p86_cupy_(rho, sigma):
    """
    CuPy version of Perdew 86 correlation.
    Corresponds to GGA_C_P86 with ID 132 in LibXC.
    Reference: J. P. Perdew, Phys. Rev. B 33, 8822 (1986).
    """
    rho = cp.maximum(rho, 1e-12)

    ec_lda, vc_lda = lda_c_pz_cupy_(rho)

    pi34 = (3.0 / (4.0 * cp.pi)) ** (1.0 / 3.0)
    rs = pi34 * rho ** (-1.0 / 3.0)

    Cinf = 0.001667
    alpha_c = 0.023266
    beta_c = 7.389e-6
    gamma_c = 8.723
    delta_c = 0.472

    rs2 = rs * rs
    rs3 = rs * rs2

    num_C = 0.002568 + alpha_c * rs + beta_c * rs2
    den_C = 1.0 + gamma_c * rs + delta_c * rs2 + 1e4 * beta_c * rs3
    Crs = Cinf + num_C / den_C

    d_factor = 1.0

    norm_dn = cp.sqrt(sigma)

    f_tilde = 0.11
    prefac_Phi = 1.745 * f_tilde

    rho76 = rho ** (7.0 / 6.0)
    rho43 = rho ** (4.0 / 3.0)

    Phi = prefac_Phi * Cinf / Crs * norm_dn / rho76

    exp_phi = cp.exp(-Phi)

    delta_ec = Crs * sigma * exp_phi / (d_factor * rho43 * rho)

    ec = ec_lda + delta_ec

    # Derivatives
    drs_dn = -rs / (3.0 * rho)

    dnum_drs = alpha_c + 2.0 * beta_c * rs
    dden_drs = gamma_c + 2.0 * delta_c * rs + 3.0e4 * beta_c * rs2
    dCrs_drs = (dnum_drs * den_C - num_C * dden_drs) / (den_C * den_C)
    dCrs_dn = dCrs_drs * drs_dn

    dPhi_dn = Phi * (-dCrs_dn / Crs - 7.0 / 6.0 / rho)

    F = delta_ec * rho
    dF_dn = F * (dCrs_dn / Crs - dPhi_dn - 4.0 / 3.0 / rho)

    vc = vc_lda + delta_ec + dF_dn

    vsigmac = Crs * exp_phi / (d_factor * rho43) * (1.0 - Phi / 2.0)
    vsigma = 0.5 * vsigmac

    return ec, vc, vsigma


def gga_c_p86_cupy(rho, sigma):
    """
    CuPy version of Perdew 86 correlation with NaN handling.
    Corresponds to GGA_C_P86 with ID 132 in LibXC.
    """
    ec, vc, vsigma = gga_c_p86_cupy_(rho, sigma)
    vsigma[cp.isnan(vsigma)] = 0
    vc[cp.isnan(vc)] = 0
    ec[cp.isnan(ec)] = 0
    return ec, vc, vsigma