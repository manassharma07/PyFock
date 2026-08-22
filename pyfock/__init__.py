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

__version__ = "0.1.6"
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
from .DFT_NumGrad import DFT_NumGrad
from .DFT_Grad import DFT_Grad
from .HF_atoms import HF_atoms
# from .PBC_ring import ring


def __getattr__(name):
    # Lazy import of the ASE calculator (PEP 562) so that ASE remains an
    # optional dependency: `import pyfock` and all integral/DFT functionality
    # work without ASE installed; only accessing PyFockCalculator requires it.
    if name == "PyFockCalculator":
        try:
            from .ase_calculator import PyFockCalculator
        except ImportError as exc:
            raise ImportError(
                "PyFockCalculator requires the optional 'ase' package. "
                "Install it with: pip install ase"
            ) from exc
        return PyFockCalculator
    raise AttributeError(f"module 'pyfock' has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + ["PyFockCalculator"])
