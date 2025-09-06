"""
This submodule provides highly optimized and symmetry-aware routines for 
evaluating all essential one-electron and two-electron integrals needed in 
electronic structure calculations. The integrals are computed over Gaussian 
type orbitals (GTOs), and both CPU (NumPy) and GPU (CuPy) implementations are 
available for many routines to accelerate quantum chemistry workflows.

Included modules and functionality:
-----------------------------------
- One-electron integrals:
    * Overlap integrals
    * Kinetic energy integrals
    * Nuclear attraction integrals
    * Dipole moment integrals
    * Gradients of one-electron integrals
- Two-electron integrals:
    * Full 4-center two electron repulsion integrals using Rys quadrature
    * 3-center and 2-center two electron integrals for density fitting
    * Schwarz screening utilities for integral pruning
- Exchange-correlation evaluation routines compatible with numerical grids
- GPU-accelerated (CuPy) versions for key performance-critical routines
- Modular helper functions for computing factorials, boys functions, contraction coefficients, etc.

Usage:
------

Example:
    from pyfock.Integrals import overlap_mat_symm
    S = overlap_mat_symm(basis)

"""
# from .integral_helpers import fac
# from .integral_helpers import fastFactorial
# from .integral_helpers import comb
# from .integral_helpers import doublefactorial
# from .integral_helpers import c2k
# from .integral_helpers import calcS
# from .integral_helpers import vlriPartial
# from .integral_helpers import calcCgamminc
# from .integral_helpers import Fboys
from .mmd_nuc_mat_symm import mmd_nuc_mat_symm
from .nuc_mat_symm import nuc_mat_symm
from .nuc_mat_symm_cupy import nuc_mat_symm_cupy
from .kin_mat_symm import kin_mat_symm
from .kin_mat_symm_cupy import kin_mat_symm_cupy
from .kin_mat_symm_shell_cupy import kin_mat_symm_shell_cupy
from .overlap_mat_symm import overlap_mat_symm
from .overlap_mat_symm_cupy import overlap_mat_symm_cupy
from .cross_overlap_mat_symm import cross_overlap_mat_symm
from .dipole_moment_mat_symm import dipole_moment_mat_symm
from .dipole_moment_mat_symm_cupy import dipole_moment_mat_symm_cupy
from .conv_4c2e_symm import conv_4c2e_symm
from .mmd_4c2e_symm import mmd_4c2e_symm
from .rys_4c2e_symm import rys_4c2e_symm, rys_4c2e_symm_old
from .conv_3c2e_symm import conv_3c2e_symm
from .rys_3c2e_symm import rys_3c2e_symm
from .rys_3c2e_symm_cupy import rys_3c2e_symm_cupy
from .rys_3c2e_symm_cupy_fp32 import rys_3c2e_symm_cupy_fp32
from .conv_2c2e_symm import conv_2c2e_symm
from .rys_2c2e_symm import rys_2c2e_symm
from .rys_2c2e_symm_cupy import rys_2c2e_symm_cupy
from .rys_3c2e_tri import rys_3c2e_tri
from .rys_nuc_mat_symm import rys_nuc_mat_symm
from .schwarz_helpers import rys_3c2e_tri_schwarz
from .eval_xc_1 import eval_xc_1
from .eval_xc_2 import eval_xc_2
from .eval_xc_3 import eval_xc_3
from .eval_xc_1_cupy import eval_xc_1_cupy
from .eval_xc_2_cupy import eval_xc_2_cupy
from .eval_xc_3_cupy import eval_xc_3_cupy
from . import bf_val_helpers
from . import rys_helpers
from . import schwarz_helpers
from . import schwarz_helpers_cupy
from . import integral_helpers
from .overlap_mat_grad_symm import overlap_mat_grad_symm
from .kin_mat_grad_symm import kin_mat_grad_symm
from .nuc_mat_grad_symm import nuc_mat_grad_symm

# __all__ = ['fac', 'fastFactorial', 'comb', 'doublefactorial', 'c2k', 'calcS', 'vlriPartial', 'calcCgamminc', 'Fboys', 'comb'\
#     , 'comb', 'comb', 'comb', 'comb', 'comb']

__all__ = ['integral_helpers', 'mmd_nuc_mat_symm', 'nuc_mat_symm', 'kin_mat_symm', 'overlap_mat_symm', 'conv_4c2e_symm', 'mmd_4c2e_symm', 'rys_helpers', 'rys_4c2e_symm',\
     'rys_4c2e_symm_old', 'conv_3c2e_symm', 'rys_3c2e_symm', 'conv_2c2e_symm', 'rys_2c2e_symm', 'rys_3c2e_tri', 'rys_nuc_mat_symm', 'schwarz_helpers', 'rys_3c2e_tri_schwarz'\
        'bf_val_helpers', 'eval_xc_1', 'eval_xc_2', 'eval_xc_3', 'eval_xc_1_cupy', 'eval_xc_2_cupy', 'eval_xc_3_cupy', 'dipole_moment_mat_symm', 'kin_mat_symm_cupy'\
        , 'overlap_mat_symm_cupy', 'dipole_moment_mat_symm_cupy', 'nuc_mat_symm_cupy', 'kin_mat_symm_shell_cupy', 'rys_2c2e_symm_cupy', 'rys_3c2e_symm_cupy',\
        'rys_3c2e_symm_cupy_fp32', 'schwarz_helpers_cupy', 'overlap_mat_grad_symm', 'kin_mat_grad_symm', 'nuc_mat_grad_symm', 'cross_overlap_mat_symm']

# # This code will import all of the modules in the library directory and expose all of 
# # the functions in those modules to the user. The user can then use the functions just 
# # like they would any other function in the library package:

# # Keep in mind that this code will expose all functions in the library directory, 
# # including any private functions (i.e. functions that start with an underscore). 
# # If you only want to expose certain functions, 
# # you can modify the __all__ list to include only the functions that you want to expose.

# import glob

# # Get a list of all module filenames in the library directory
# # module_filenames = glob.glob('pyfock/Integrals/*.py')
# module_filenames = glob.glob('*.py')

# print(module_filenames)

# # Remove the `__init__.py` file from the list
# # module_filenames.remove('pyfock/Integrals/__init__.py')
# module_filenames.remove('__init__.py')

# # Use a for loop to import all of the modules in the Integrals directory
# for module_filename in module_filenames:
#     # module_filename = module_filename.replace("pyfock/Integrals/", "")
#     module_name = module_filename[:-3]
#     exec(f'from .{module_name} import *')

# # Expose all of the functions in the __all__ variable
# __all__ = [
#     func for func in dir()
#     if func[0] != '_' and callable(eval(func))
# ]

