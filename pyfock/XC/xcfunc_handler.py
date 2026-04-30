from pyfock.XC import lda_x, lda_x_cupy, lda_c_vwn, lda_c_vwn_cupy, lda_c_pw, lda_c_pw_cupy, lda_c_pw_mod, lda_c_pw_mod_cupy
from pyfock.XC import lda_c_pz, lda_c_pz_cupy, lda_c_pz_mod, lda_c_pz_mod_cupy
from pyfock.XC import gga_x_pbe, gga_x_pbe_cupy, gga_c_pbe, gga_c_pbe_cupy, gga_x_b88, gga_x_b88_cupy, gga_c_lyp, gga_c_lyp_cupy
from pyfock.XC import gga_x_pbe_sol, gga_x_pbe_sol_cupy, gga_c_pbe_sol, gga_c_pbe_sol_cupy
from pyfock.XC import gga_x_rpbe, gga_x_rpbe_cupy
from pyfock.XC import gga_x_pw91, gga_x_pw91_cupy, gga_c_pw91, gga_c_pw91_cupy
from pyfock.XC import gga_c_p86, gga_c_p86_cupy
from pyfock.XC import mgga_x_tpss, mgga_x_tpss_cupy, mgga_c_tpss, mgga_c_tpss_cupy
from pyfock.XC import mgga_x_m06_l, mgga_x_m06_l_cupy, mgga_c_m06_l, mgga_c_m06_l_cupy
from pyfock.XC import mgga_x_r2scan, mgga_x_r2scan_cupy, mgga_c_r2scan, mgga_c_r2scan_cupy
from pyfock.XC import mgga_x_task, mgga_x_task_cupy

# LibXC IDs of implemented functionals
# 1   - LDA_X
# 7   - LDA_C_VWN
# 9   - LDA_C_PZ
# 10  - LDA_C_PZ_MOD
# 12  - LDA_C_PW
# 13  - LDA_C_PW_MOD
# 101 - GGA_X_PBE
# 106 - GGA_X_B88
# 109 - GGA_X_PW91
# 116 - GGA_X_PBE_SOL
# 117 - GGA_X_RPBE
# 130 - GGA_C_PBE
# 131 - GGA_C_LYP
# 132 - GGA_C_P86
# 133 - GGA_C_PBE_SOL
# 134 - GGA_C_PW91
# 202 - MGGA_X_TPSS
# 203 - MGGA_X_M06_L
# 231 - MGGA_C_TPSS
# 233 - MGGA_C_M06_L
# 497 - MGGA_X_R2SCAN
# 498 - MGGA_C_R2SCAN
# 707 - MGGA_X_TASK
_IMPLEMENTED_IDS = {
    1, 7, 9, 10, 12, 13,
    101, 106, 109, 116, 117, 130, 131, 132, 133, 134,
    202, 203, 231, 233, 497, 498, 707,
}


# ──────────────────────────────────────────────────────────────────────────────
# Metadata tables
# ──────────────────────────────────────────────────────────────────────────────

# Add to _FUNCTIONAL_DATA or as a separate mapping
_FUNCTIONAL_FAMILY = {
    1:   1,  # LDA_X        → LDA
    7:   1,  # LDA_C_VWN    → LDA
    9:   1,  # LDA_C_PZ     → LDA
    10:  1,  # LDA_C_PZ_MOD → LDA
    12:  1,  # LDA_C_PW     → LDA
    13:  1,  # LDA_C_PW_MOD → LDA
    101: 2,  # GGA_X_PBE    → GGA
    106: 2,  # GGA_X_B88    → GGA
    109: 2,  # GGA_X_PW91   → GGA
    116: 2,  # GGA_X_PBE_SOL→ GGA
    117: 2,  # GGA_X_RPBE   → GGA
    130: 2,  # GGA_C_PBE    → GGA
    131: 2,  # GGA_C_LYP    → GGA
    132: 2,  # GGA_C_P86    → GGA
    133: 2,  # GGA_C_PBE_SOL→ GGA
    134: 2,  # GGA_C_PW91   → GGA
    74:  4,  # MGGA_C_MN12_L   → MGGA
    75:  4,  # MGGA_C_M11_L    → MGGA
    202: 4,  # MGGA_X_TPSS     → MGGA
    203: 4,  # MGGA_X_M06_L    → MGGA
    212: 4,  # MGGA_X_REVTPSS  → MGGA
    226: 4,  # MGGA_X_M11_L    → MGGA
    227: 4,  # MGGA_X_MN12_L   → MGGA
    231: 4,  # MGGA_C_TPSS     → MGGA
    233: 4,  # MGGA_C_M06_L    → MGGA
    241: 4,  # MGGA_C_REVTPSS  → MGGA
    497: 4,  # MGGA_X_R2SCAN   → MGGA
    498: 4,  # MGGA_C_R2SCAN   → MGGA
    642: 4,  # MGGA_C_R2SCAN01 → MGGA
    645: 4,  # MGGA_X_R2SCAN01 → MGGA
    700: 4,  # MGGA_X_SCANL    → MGGA
    702: 4,  # MGGA_C_SCANL    → MGGA
    707: 4,  # MGGA_X_TASK     → MGGA
    718: 4,  # MGGA_X_R2SCANL  → MGGA
    719: 4,  # MGGA_C_R2SCANL  → MGGA
}

FAMILY_LDA    = 1
FAMILY_GGA    = 2
FAMILY_HYBRID = 3
FAMILY_MGGA   = 4

# Maps LibXC ID → (canonical name, citation)
_FUNCTIONAL_DATA = {
    1: (
        "LDA_X",
        "P. A. M. Dirac, Math. Proc. Cambridge Philos. Soc. 26, 376 (1930); "
        "F. Bloch, Z. Phys. 57, 545 (1929).",
    ),
    7: (
        "LDA_C_VWN",
        "S. H. Vosko, L. Wilk, and M. Nusair, Can. J. Phys. 58, 1200 (1980).",
    ),
    9: (
        "LDA_C_PZ",
        "J. P. Perdew and A. Zunger, Phys. Rev. B 23, 5048 (1981).",
    ),
    10: (
        "LDA_C_PZ_MOD",
        "J. P. Perdew and A. Zunger, Phys. Rev. B 23, 5048 (1981) [modified parameterisation].",
    ),
    12: (
        "LDA_C_PW",
        "J. P. Perdew and Y. Wang, Phys. Rev. B 45, 13244 (1992).",
    ),
    13: (
        "LDA_C_PW_MOD",
        "J. P. Perdew and Y. Wang, Phys. Rev. B 45, 13244 (1992) [modified parameterisation].",
    ),
    101: (
        "GGA_X_PBE",
        "J. P. Perdew, K. Burke, and M. Ernzerhof, Phys. Rev. Lett. 77, 3865 (1996).",
    ),
    106: (
        "GGA_X_B88",
        "A. D. Becke, Phys. Rev. A 38, 3098 (1988).",
    ),
    109: (
        "GGA_X_PW91",
        "J. P. Perdew, in Electronic Structure of Solids '91, edited by P. Ziesche and H. Eschrig "
        "(Akademie Verlag, Berlin, 1991); "
        "J. P. Perdew, J. A. Chevary, S. H. Vosko, K. A. Jackson, M. R. Pederson, D. J. Singh, "
        "and C. Fiolhais, Phys. Rev. B 46, 6671 (1992).",
    ),
    116: (
        "GGA_X_PBE_SOL",
        "J. P. Perdew, A. Ruzsinszky, G. I. Csonka, O. A. Vydrov, G. E. Scuseria, "
        "L. A. Constantin, X. Zhou, and K. Burke, Phys. Rev. Lett. 100, 136406 (2008).",
    ),
    117: (
        "GGA_X_RPBE",
        "B. Hammer, L. B. Hansen, and J. K. Nørskov, Phys. Rev. B 59, 7413 (1999).",
    ),
    130: (
        "GGA_C_PBE",
        "J. P. Perdew, K. Burke, and M. Ernzerhof, Phys. Rev. Lett. 77, 3865 (1996).",
    ),
    131: (
        "GGA_C_LYP",
        "C. Lee, W. Yang, and R. G. Parr, Phys. Rev. B 37, 785 (1988); "
        # "B. Miehlich, A. Savin, H. Stoll, and H. Preuss, Chem. Phys. Lett. 157, 200 (1989).",
    ),
    132: (
        "GGA_C_P86",
        "J. P. Perdew, Phys. Rev. B 33, 8822 (1986).",
    ),
    133: (
        "GGA_C_PBE_SOL",
        "J. P. Perdew, A. Ruzsinszky, G. I. Csonka, O. A. Vydrov, G. E. Scuseria, "
        "L. A. Constantin, X. Zhou, and K. Burke, Phys. Rev. Lett. 100, 136406 (2008).",
    ),
    134: (
        "GGA_C_PW91",
        "J. P. Perdew, J. A. Chevary, S. H. Vosko, K. A. Jackson, M. R. Pederson, D. J. Singh, "
        "and C. Fiolhais, Phys. Rev. B 46, 6671 (1992).",
    ),
    74: (
        "MGGA_C_MN12_L",
        "R. Peverati and D. G. Truhlar, Phys. Chem. Chem. Phys. 14, 16187 (2012).",
    ),
    75: (
        "MGGA_C_M11_L",
        "R. Peverati and D. G. Truhlar, J. Phys. Chem. Lett. 3, 117 (2012).",
    ),
    202: (
        "MGGA_X_TPSS",
        "J. Tao, J. P. Perdew, V. N. Staroverov, and G. E. Scuseria, "
        "Phys. Rev. Lett. 91, 146401 (2003); "
        "J. P. Perdew, J. Tao, V. N. Staroverov, and G. E. Scuseria, "
        "J. Chem. Phys. 120, 6898 (2004).",
    ),
    203: (
        "MGGA_X_M06_L",
        "Y. Zhao and D. G. Truhlar, J. Chem. Phys. 125, 194101 (2006).",
    ),
    212: (
        "MGGA_X_REVTPSS",
        "J. P. Perdew, A. Ruzsinszky, G. I. Csonka, L. A. Constantin, and J. Sun, Phys. Rev. Lett. 103, 026403 (2009).",
    ),
    226: (
        "MGGA_X_M11_L",
        "R. Peverati and D. G. Truhlar, J. Phys. Chem. Lett. 3, 117 (2012).",
    ),
    227: (
        "MGGA_X_MN12_L",
        "R. Peverati and D. G. Truhlar, Phys. Chem. Chem. Phys. 14, 16187 (2012).",
    ),
    231: (
        "MGGA_C_TPSS",
        "J. Tao, J. P. Perdew, V. N. Staroverov, and G. E. Scuseria, "
        "Phys. Rev. Lett. 91, 146401 (2003); "
        "J. P. Perdew, J. Tao, V. N. Staroverov, and G. E. Scuseria, "
        "J. Chem. Phys. 120, 6898 (2004).",
    ),
    233: (
        "MGGA_C_M06_L",
        "Y. Zhao and D. G. Truhlar, J. Chem. Phys. 125, 194101 (2006).",
    ),
    241: (
        "MGGA_C_REVTPSS",
        "J. P. Perdew, A. Ruzsinszky, G. I. Csonka, L. A. Constantin, and J. Sun, Phys. Rev. Lett. 103, 026403 (2009).",
    ),
    497: (
        "MGGA_X_R2SCAN",
        "J. W. Furness, A. D. Kaplan, J. Ning, J. P. Perdew, and J. Sun, J. Phys. Chem. Lett. 11, 8208 (2020).",
    ),
    498: (
        "MGGA_C_R2SCAN",
        "J. W. Furness, A. D. Kaplan, J. Ning, J. P. Perdew, and J. Sun, J. Phys. Chem. Lett. 11, 8208 (2020).",
    ),
    642: (
        "MGGA_C_R2SCAN01",
        "P. Bartok and J. R. Yates, J. Chem. Phys. 150, 161101 (2019).",
    ),
    645: (
        "MGGA_X_R2SCAN01",
        "P. Bartok and J. R. Yates, J. Chem. Phys. 150, 161101 (2019).",
    ),
    700: (
        "MGGA_X_SCANL",
        "A. Mejia-Rodriguez and S. B. Trickey, Phys. Rev. A 96, 052512 (2017).",
    ),
    702: (
        "MGGA_C_SCANL",
        "A. Mejia-Rodriguez and S. B. Trickey, Phys. Rev. A 96, 052512 (2017).",
    ),
    707: (
        "MGGA_X_TASK",
        "T. Aschebrock and S. Kümmel, Phys. Rev. Res. 1, 033082 (2019).",
    ),
    718: (
        "MGGA_X_R2SCANL",
        "A. P. Bartok and J. R. Yates, J. Chem. Phys. 150, 161101 (2019).",
    ),
    719: (
        "MGGA_C_R2SCANL",
        "A. P. Bartok and J. R. Yates, J. Chem. Phys. 150, 161101 (2019).",
    ),
}

# Reverse map: upper-cased canonical name → LibXC ID
_NAME_TO_ID = {name.upper(): fid for fid, (name, _) in _FUNCTIONAL_DATA.items()}

# Common shorthand → list of LibXC IDs (exchange first, then correlation)
_ALIAS_TO_IDS = {
    "PBE":      [101, 130],   # GGA_X_PBE     + GGA_C_PBE
    "PBESOL":   [116, 133],   # GGA_X_PBE_SOL + GGA_C_PBE_SOL
    "RPBE":     [117, 130],   # GGA_X_RPBE    + GGA_C_PBE  (RPBE uses PBE correlation)
    "PW91":     [109, 134],   # GGA_X_PW91    + GGA_C_PW91
    "BP86":     [106, 132],   # GGA_X_B88     + GGA_C_P86
    "BLYP":     [106, 131],   # GGA_X_B88     + GGA_C_LYP
    "LDA":      [1,   7],     # LDA_X         + LDA_C_VWN  (most common LDA combo)
    "SVWN":     [1,   7],     # synonym for LDA/VWN
    "SVWN5":    [1,   7],
    "SPZ":      [1,   9],     # LDA_X         + LDA_C_PZ
    "SPW":      [1,  12],     # LDA_X         + LDA_C_PW
    "PW91C":    [12],         # bare PW correlation (no common exchange alias here)
    "R2SCAN":   [497, 498],   # MGGA_X_R2SCAN   + MGGA_C_R2SCAN
    "TPSS":     [202, 231],   # MGGA_X_TPSS     + MGGA_C_TPSS
    "TASK":     [707],        # MGGA_X_TASK     only
    "MN12L":    [227, 74],    # MGGA_X_MN12_L   + MGGA_C_MN12_L
    "MN12-L":   [227, 74],
    "M06L":     [203, 233],   # MGGA_X_M06_L    + MGGA_C_M06_L
    "M06-L":    [203, 233],
    "M11L":     [226, 75],    # MGGA_X_M11_L    + MGGA_C_M11_L
    "M11-L":    [226, 75],
    "REVTPSS":  [212, 241],   # MGGA_X_REVTPSS  + MGGA_C_REVTPSS
    "SCANL":    [700, 702],   # MGGA_X_SCANL    + MGGA_C_SCANL
    "R2SCANL":  [718, 719],   # MGGA_X_R2SCANL  + MGGA_C_R2SCANL
    "R2SCAN01": [645, 642],   # MGGA_X_R2SCAN01 + MGGA_C_R2SCAN01
}


def get_functional_id(name: str) -> int:
    """
    Return the LibXC ID for a functional given its canonical name.

    Parameters
    ----------
    name : str
        Canonical LibXC-style name, e.g. ``"GGA_X_PBE"`` or ``"LDA_X"``.
        Case-insensitive.

    Returns
    -------
    int
        LibXC functional ID.

    Raises
    ------
    KeyError
        If the name is not recognised.

    Examples
    --------
    >>> get_functional_id("GGA_X_PBE")
    101
    >>> get_functional_id("lda_x")
    1
    """
    key = name.strip().upper()
    if key not in _NAME_TO_ID:
        raise KeyError(
            f"Unknown functional name '{name}'. "
            f"Available names: {sorted(_NAME_TO_ID.keys())}"
        )
    return _NAME_TO_ID[key]


def get_functional_citation(funcid: int) -> str:
    """
    Return the primary literature citation for a functional given its LibXC ID.

    Parameters
    ----------
    funcid : int
        LibXC functional ID (e.g. 101 for GGA_X_PBE).

    Returns
    -------
    str
        Citation string.

    Raises
    ------
    KeyError
        If the ID is not recognised.

    Examples
    --------
    >>> print(get_functional_citation(101))
    J. P. Perdew, K. Burke, and M. Ernzerhof, Phys. Rev. Lett. 77, 3865 (1996).
    """
    if funcid not in _FUNCTIONAL_DATA:
        raise KeyError(
            f"Unknown functional ID {funcid}. "
            f"Available IDs: {sorted(_FUNCTIONAL_DATA.keys())}"
        )
    _, citation = _FUNCTIONAL_DATA[funcid]
    return citation


def resolve_functional(name: str):
    """
    Resolve a common functional shorthand or canonical name to a list of LibXC IDs.

    For shorthands like ``"PBE"`` the function returns the conventional
    exchange + correlation pair.  For canonical LibXC names like
    ``"GGA_X_PBE"`` it returns a single-element list.

    Parameters
    ----------
    name : str
        Common shorthand (e.g. ``"PBE"``, ``"BLYP"``, ``"LDA"``) or a
        canonical LibXC name (e.g. ``"GGA_X_PBE"``).  Case-insensitive.

    Returns
    -------
    list[int]
        One or more LibXC functional IDs.

    Raises
    ------
    KeyError
        If the name cannot be resolved.

    Examples
    --------
    >>> resolve_functional("PBE")
    [101, 130]
    >>> resolve_functional("GGA_X_PBE")
    [101]
    >>> resolve_functional("BLYP")
    [106, 131]
    """
    key = name.strip().upper()

    # 1. Try common aliases first
    if key in _ALIAS_TO_IDS:
        return list(_ALIAS_TO_IDS[key])

    # 2. Try canonical name
    if key in _NAME_TO_ID:
        return [_NAME_TO_ID[key]]

    raise KeyError(
        f"Cannot resolve functional '{name}'. "
        f"Known aliases: {sorted(_ALIAS_TO_IDS.keys())}. "
        f"Known canonical names: {sorted(_NAME_TO_ID.keys())}."
    )



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
        - 130 : GGA_C_PBE
        - 131 : GGA_C_LYP
    For unsupported functionals, you can use LibXC directly.
    """
    if funcid not in _IMPLEMENTED_IDS:
        print(
            "ERROR: The specified functional is not implemented in PyFock. "
            "You need to use LibXC to calculate the functional values."
        )
        exit()


def get_implemented_ids():
    return sorted(_IMPLEMENTED_IDS)


def func_compute(funcid, rho, sigma=None, tau=None, use_gpu=True):
    """
    Compute exchange-correlation energy and potential using the specified functional.

    Parameters
    ----------
    funcid : int
        Identifier for the XC functional. Matches LibXC IDs:
            - 1   : LDA_X
            - 7   : LDA_C_VWN
            - 9   : LDA_C_PZ
            - 10  : LDA_C_PZ_MOD
            - 12  : LDA_C_PW
            - 13  : LDA_C_PW_MOD
            - 101 : GGA_X_PBE
            - 106 : GGA_X_B88
            - 109 : GGA_X_PW91
            - 116 : GGA_X_PBE_SOL
            - 117 : GGA_X_RPBE
            - 130 : GGA_C_PBE
            - 131 : GGA_C_LYP
            - 132 : GGA_C_P86
            - 133 : GGA_C_PBE_SOL
            - 134 : GGA_C_PW91

    rho : ndarray
        Electron density array.

    sigma : ndarray, optional
        Gradient of the density (∇ρ·∇ρ), required for GGA functionals.

    use_gpu : bool, default=True
        Whether to use CuPy (GPU) versions of the functionals.

    Returns
    -------
    energy_density : ndarray
        The exchange-correlation energy density at each grid point.

    potential : ndarray or tuple
        The XC potential (and sigma-derivative for GGA).

    Raises
    ------
    SystemExit
        If the functional is not implemented in PyFock.
    """

    # ── Dispatch tables (built once, cached by Python) ────────────────────
    _LDA_CPU = {
        1:  lda_x,
        7:  lda_c_vwn,
        9:  lda_c_pz,
        10: lda_c_pz_mod,
        12: lda_c_pw,
        13: lda_c_pw_mod,
    }

    _LDA_GPU = {
        1:  lda_x_cupy,
        7:  lda_c_vwn_cupy,
        9:  lda_c_pz_cupy,
        10: lda_c_pz_mod_cupy,
        12: lda_c_pw_cupy,
        13: lda_c_pw_mod_cupy,
    }

    _GGA_CPU = {
        101: gga_x_pbe,
        106: gga_x_b88,
        109: gga_x_pw91,
        116: gga_x_pbe_sol,
        117: gga_x_rpbe,
        130: gga_c_pbe,
        131: gga_c_lyp,
        132: gga_c_p86,
        133: gga_c_pbe_sol,
        134: gga_c_pw91,
    }

    _GGA_GPU = {
        101: gga_x_pbe_cupy,
        106: gga_x_b88_cupy,
        109: gga_x_pw91_cupy,
        116: gga_x_pbe_sol_cupy,
        117: gga_x_rpbe_cupy,
        130: gga_c_pbe_cupy,
        131: gga_c_lyp_cupy,
        132: gga_c_p86_cupy,
        133: gga_c_pbe_sol_cupy,
        134: gga_c_pw91_cupy,
    }

    _MGGA_CPU = {
        202: mgga_x_tpss,
        203: mgga_x_m06_l,
        231: mgga_c_tpss,
        233: mgga_c_m06_l,
        497: mgga_x_r2scan,
        498: mgga_c_r2scan,
        707: mgga_x_task,
    }

    _MGGA_GPU = {
        202: mgga_x_tpss_cupy,
        203: mgga_x_m06_l_cupy,
        231: mgga_c_tpss_cupy,
        233: mgga_c_m06_l_cupy,
        497: mgga_x_r2scan_cupy,
        498: mgga_c_r2scan_cupy,
        707: mgga_x_task_cupy,
    }

    # ── Look up and call ──────────────────────────────────────────────────
    if use_gpu:
        lda_table, gga_table, mgga_table = _LDA_GPU, _GGA_GPU, _MGGA_GPU
    else:
        lda_table, gga_table, mgga_table = _LDA_CPU, _GGA_CPU, _MGGA_CPU

    if funcid in lda_table:
        return lda_table[funcid](rho)

    if funcid in gga_table:
        if sigma is None:
            raise ValueError(
                f"GGA functional (ID {funcid}) requires 'sigma' (∇ρ·∇ρ), "
                f"but sigma=None was provided."
            )
        return gga_table[funcid](rho, sigma)

    if funcid in mgga_table:
        if sigma is None or tau is None:
            raise ValueError(
                f"MGGA functional (ID {funcid}) requires 'sigma' and 'tau', "
                f"but sigma={sigma is not None} and tau={tau is not None} were provided."
            )
        return mgga_table[funcid](rho, sigma, tau)

    # ── Functional not found ──────────────────────────────────────────────
    name = _FUNCTIONAL_DATA.get(funcid, (f"ID={funcid}",))[0]
    raise NotImplementedError(
        f"Functional '{name}' (LibXC ID {funcid}) is not implemented in PyFock. "
        f"Use LibXC to compute this functional.\n"
        f"Implemented IDs: {sorted(set(lda_table) | set(gga_table) | set(mgga_table))}"
    )

def get_family(funcid: int) -> int:
    """
    Return the family of a functional given its LibXC ID.

    Parameters
    ----------
    funcid : int
        LibXC functional ID.

    Returns
    -------
    int
        1 = LDA, 2 = GGA, 3 = Hybrid, 4 = mGGA

    Raises
    ------
    KeyError
        If the functional ID is not recognised.

    Examples
    --------
    >>> get_family(1)
    1
    >>> get_family(101)
    2
    """
    if funcid not in _FUNCTIONAL_FAMILY:
        raise KeyError(
            f"Unknown functional ID {funcid}. "
            f"Available IDs: {sorted(_FUNCTIONAL_FAMILY.keys())}"
        )
    return _FUNCTIONAL_FAMILY[funcid]
