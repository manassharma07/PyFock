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
from .gga_x_pbe import gga_x_pbe, gga_x_pbe_cupy
from .gga_c_pbe import gga_c_pbe, gga_c_pbe_cupy
from .gga_x_b88 import gga_x_b88, gga_x_b88_cupy
from .gga_c_lyp import gga_c_lyp, gga_c_lyp_cupy
from .xcfunc_handler import check_implemented, func_compute

__all__ = [
    'lda_x', 'lda_x_cupy', 'lda_c_vwn', 'lda_c_vwn_cupy',
    'lda_c_pw', 'lda_c_pw_cupy', 'lda_c_pw_mod', 'lda_c_pw_mod_cupy',
    'gga_x_pbe', 'gga_x_pbe_cupy', 'gga_c_pbe', 'gga_c_pbe_cupy',
    'gga_x_b88', 'gga_x_b88_cupy', 'gga_c_lyp', 'gga_c_lyp_cupy',
    'check_implemented', 'func_compute'
]



