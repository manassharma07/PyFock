from pyfock.XC import lda_x, lda_x_cupy, lda_c_vwn, lda_c_vwn_cupy, lda_c_pw, lda_c_pw_cupy, lda_c_pw_mod, lda_c_pw_mod_cupy
from pyfock.XC import lda_c_pz, lda_c_pz_cupy, lda_c_pz_mod, lda_c_pz_mod_cupy
from pyfock.XC import gga_x_pbe, gga_x_pbe_cupy, gga_c_pbe, gga_c_pbe_cupy, gga_x_b88, gga_x_b88_cupy, gga_c_lyp, gga_c_lyp_cupy
from pyfock.XC import gga_x_pbe_sol, gga_x_pbe_sol_cupy, gga_c_pbe_sol, gga_c_pbe_sol_cupy
from pyfock.XC import gga_x_rpbe, gga_x_rpbe_cupy
from pyfock.XC import gga_x_pw91, gga_x_pw91_cupy, gga_c_pw91, gga_c_pw91_cupy
from pyfock.XC import gga_c_p86, gga_c_p86_cupy

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
    if funcid not in _FUNCTIONAL_DATA:
        print(
            "ERROR: The specified functional is not implemented in PyFock. "
            "You need to use LibXC to calculate the functional values."
        )
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

    # ── Look up and call ──────────────────────────────────────────────────
    if use_gpu:
        lda_table, gga_table = _LDA_GPU, _GGA_GPU
    else:
        lda_table, gga_table = _LDA_CPU, _GGA_CPU

    if funcid in lda_table:
        return lda_table[funcid](rho)

    if funcid in gga_table:
        if sigma is None:
            raise ValueError(
                f"GGA functional (ID {funcid}) requires 'sigma' (∇ρ·∇ρ), "
                f"but sigma=None was provided."
            )
        return gga_table[funcid](rho, sigma)

    # ── Functional not found ──────────────────────────────────────────────
    name = _FUNCTIONAL_DATA.get(funcid, (f"ID={funcid}",))[0]
    raise NotImplementedError(
        f"Functional '{name}' (LibXC ID {funcid}) is not implemented in PyFock. "
        f"Use LibXC to compute this functional.\n"
        f"Implemented IDs: {sorted(set(lda_table) | set(gga_table))}"
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