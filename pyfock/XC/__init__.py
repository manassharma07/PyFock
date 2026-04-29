"""
This module provides a collection of exchange-correlation (XC) functionals 
commonly used in density functional theory (DFT) calculations. It includes 
both exchange-only (X) and correlation (C) functionals within the LDA and GGA 
frameworks, along with their respective CuPy-accelerated implementations 
for GPU computation.

Available functionals:
- LDA exchange and correlation (X: lda_x, C: lda_c_vwn, lda_c_pw, lda_c_pw_mod)
- GGA exchange and correlation (X: gga_x_pbe, gga_x_b88; C: gga_c_pbe, gga_c_lyp)
- CuPy variants for GPU-accelerated computation (e.g., lda_x_cupy, gga_c_pbe_cupy)

Also includes utility functions:
- `check_implemented`: verifies if a given XC functional is implemented.
- `func_compute`: generic interface to compute XC energy and potential.

All components can be imported directly from this module.
"""
from .lda_x import lda_x, lda_x_cupy
from .lda_c_vwn import lda_c_vwn, lda_c_vwn_cupy
from .lda_c_pw import lda_c_pw, lda_c_pw_cupy
from .lda_c_pw_mod import lda_c_pw_mod, lda_c_pw_mod_cupy
from .lda_c_pz import lda_c_pz, lda_c_pz_cupy
from .lda_c_pz_mod import lda_c_pz_mod, lda_c_pz_mod_cupy
from .gga_x_pbe import gga_x_pbe, gga_x_pbe_cupy
from .gga_c_pbe import gga_c_pbe, gga_c_pbe_cupy
from .gga_x_pbe_sol import gga_x_pbe_sol, gga_x_pbe_sol_cupy
from .gga_c_pbe_sol import gga_c_pbe_sol, gga_c_pbe_sol_cupy
from .gga_x_rpbe import gga_x_rpbe, gga_x_rpbe_cupy
from .gga_x_pw91 import gga_x_pw91, gga_x_pw91_cupy
from .gga_c_pw91 import gga_c_pw91, gga_c_pw91_cupy
from .gga_c_p86 import gga_c_p86, gga_c_p86_cupy
from .gga_x_b88 import gga_x_b88, gga_x_b88_cupy
from .gga_c_lyp import gga_c_lyp, gga_c_lyp_cupy
from .mgga_x_tpss import mgga_x_tpss, mgga_x_tpss_cupy
from .mgga_c_tpss import mgga_c_tpss, mgga_c_tpss_cupy
from .mgga_m06_l import mgga_x_m06_l, mgga_x_m06_l_cupy, mgga_c_m06_l, mgga_c_m06_l_cupy
from .xcfunc_handler import check_implemented, func_compute, get_functional_id, resolve_functional, get_implemented_ids
from .xcfunc_handler import get_family, get_functional_citation
# get_functional_citation, _ALIAS_TO_IDS, _NAME_TO_ID, _FUNCTIONAL_DATA

__all__ = [
    'lda_x', 'lda_x_cupy',
    'lda_c_vwn', 'lda_c_vwn_cupy',
    'lda_c_pw', 'lda_c_pw_cupy',
    'lda_c_pw_mod', 'lda_c_pw_mod_cupy',
    'lda_c_pz', 'lda_c_pz_cupy',
    'lda_c_pz_mod', 'lda_c_pz_mod_cupy',
    'gga_x_pbe', 'gga_x_pbe_cupy',
    'gga_c_pbe', 'gga_c_pbe_cupy',
    'gga_x_pbe_sol', 'gga_x_pbe_sol_cupy',
    'gga_c_pbe_sol', 'gga_c_pbe_sol_cupy',
    'gga_x_rpbe', 'gga_x_rpbe_cupy',
    'gga_x_pw91', 'gga_x_pw91_cupy',
    'gga_c_pw91', 'gga_c_pw91_cupy',
    'gga_c_p86', 'gga_c_p86_cupy',
    'gga_x_b88', 'gga_x_b88_cupy',
    'gga_c_lyp', 'gga_c_lyp_cupy',
    'mgga_x_tpss', 'mgga_x_tpss_cupy',
    'mgga_c_tpss', 'mgga_c_tpss_cupy',
    'mgga_x_m06_l', 'mgga_x_m06_l_cupy',
    'mgga_c_m06_l', 'mgga_c_m06_l_cupy',
    'check_implemented', 'func_compute', 'get_functional_id', 'get_implemented_ids',
    'resolve_functional', 'get_functional_citation', 'get_family'
]



