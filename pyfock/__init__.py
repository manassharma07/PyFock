"""
.. include:: ../README.md

Modules:
- `Mol`: Molecular properties
- `Basis`: Basis set management
- `Integrals`: 1e/2e integral routines
- `XC`, `Grids`: Exchange-correlation and grid definitions
- `Utils`, `Graphics`: Helper utilities and visualization

This project benefits from the Python scientific stack: NumPy, SciPy, Opt_Einsum, NumExpr, Joblib, and more.  
  
Repository: https://github.com/manassharma07/PyFock  
License: MIT  
Author: Manas Sharma  
  
For usage examples, demos, and API documentation, refer to the online documentation or example notebooks.  
"""

__version__ = "0.0.9"
__author__ = 'Manas Sharma'
__credits__ = 'Phys Whiz (bragitoff.com)'


#NOTE: The order in which these statements appear is very important.
# The Data import comes before Basis import because Basis requires stuff from
# Data, therefore it should be imported first.
from .Data import Data
from .Basis import Basis
from .Mol import Mol
from .Grids import Grids
from .DFT import DFT
# from .PBC_ring import ring

