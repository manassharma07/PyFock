from pyfock.XC import lda_x, lda_x_cupy, lda_c_vwn, lda_c_vwn_cupy, lda_c_pw, lda_c_pw_cupy, lda_c_pw_mod, lda_c_pw_mod_cupy
from pyfock.XC import gga_x_pbe, gga_x_pbe_cupy, gga_c_pbe, gga_c_pbe_cupy, gga_x_b88, gga_x_b88_cupy, gga_c_lyp, gga_c_lyp_cupy
#LibXC IDs of implemented functionals

# 1   - LDA_X
# 7   - LDA_C_VWN
# 12  - LDA_C_PW
# 13  - LDA_C_PW_MOD
# 101 - GGA_X_PBE
# 106 - GGA_X_B88
# 130 - GGA_C_PBE
# 131 - GGA_C_LYP

def check_implemented(funcid):
    """
    Check if the given functional ID is implemented in PyFock.

    Parameters
    ----------
    funcid : int
        LibXC-style functional ID (e.g., 1 for LDA_X, 101 for GGA_X_PBE).

    Raises
    ------
    SystemExit
        If the functional ID is not supported by PyFock, prints an error and exits.

    Notes
    -----
    Implemented functional IDs in PyFock:
        - 1   : LDA_X
        - 7   : LDA_C_VWN
        - 12  : LDA_C_PW
        - 13  : LDA_C_PW_MOD
        - 101 : GGA_X_PBE
        - 106 : GGA_X_B88
        - 131 : GGA_C_LYP
    For unsupported functionals, you can use LibXC directly.
    """
    if funcid not in [1, 7, 12, 13, 101, 106, 131]:
        print('ERROR: The specified functional is not implemented in PyFock. You need to use LibXC to calculate the functional values.')
        exit()

def func_compute(funcid, rho, sigma=None, use_gpu=True):
    """
    Compute exchange-correlation energy and potential using the specified functional.

    Parameters
    ----------
    funcid : int
        Identifier for the XC functional. Matches LibXC IDs:
            - 1   : LDA_X
            - 7   : LDA_C_VWN
            - 12  : LDA_C_PW
            - 13  : LDA_C_PW_MOD
            - 101 : GGA_X_PBE
            - 106 : GGA_X_B88
            - 130 : GGA_C_PBE
            - 131 : GGA_C_LYP

    rho : ndarray
        Electron density array. Should be shape-compatible with the expected input of the XC functional.

    sigma : ndarray, optional
        Gradient of the density (∇ρ·∇ρ), required for GGA functionals.

    use_gpu : bool, default=True
        Whether to use CuPy (GPU) versions of the functionals. If False, falls back to CPU (NumPy) versions.

    Returns
    -------
    energy_density : ndarray
        The exchange-correlation energy density at each grid point.

    potential : ndarray or tuple
        The exchange-correlation potential, and for GGA, possibly its derivative with respect to sigma.

    Raises
    ------
    SystemExit
        If the functional is not implemented in PyFock.

    Notes
    -----
    For unsupported functionals (i.e., other than those listed above), use LibXC.
    """
    if use_gpu:
        if funcid==1:
            return lda_x_cupy(rho)
        elif funcid==7:
            return lda_c_vwn_cupy(rho)
        elif funcid==12:
            return lda_c_pw_cupy(rho)
        elif funcid==13:
            return lda_c_pw_mod_cupy(rho)
        elif funcid==101:
            return gga_x_pbe_cupy(rho, sigma)
        elif funcid==106:
            return gga_x_b88_cupy(rho, sigma)
        elif funcid==130:
            return gga_c_pbe_cupy(rho, sigma)
        elif funcid==131:
            return gga_c_lyp_cupy(rho, sigma)
        else:
            print('The specified functional is not implemented in PyFock. You need to use LibXC to calculate the functional values.')
            exit()
    else:
        if funcid==1:
            return lda_x(rho)
        elif funcid==7:
            return lda_c_vwn(rho)
        elif funcid==12:
            return lda_c_pw(rho)
        elif funcid==13:
            return lda_c_pw_mod(rho)
        elif funcid==101:
            return gga_x_pbe(rho, sigma)
        elif funcid==106:
            return gga_x_b88(rho, sigma)
        elif funcid==130:
            return gga_c_pbe(rho, sigma)
        elif funcid==131:
            return gga_c_lyp(rho, sigma)
        else:
            print('The specified functional is not implemented in PyFock. You need to use LibXC to calculate the functional values.')
            exit()
        


