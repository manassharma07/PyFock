# HF.py
# Author: Manas Sharma (manassharma07@live.com)
# This is a part of CrysX (https://bragitoff.com/crysx)
#
#
#  .d8888b.                            Y88b   d88P       8888888b.           8888888888                888      
# d88P  Y88b                            Y88b d88P        888   Y88b          888                       888      
# 888    888                             Y88o88P         888    888          888                       888      
# 888        888d888 888  888 .d8888b     Y888P          888   d88P 888  888 8888888  .d88b.   .d8888b 888  888 
# 888        888P"   888  888 88K         d888b          8888888P"  888  888 888     d88""88b d88P"    888 .88P 
# 888    888 888     888  888 "Y8888b.   d88888b  888888 888        888  888 888     888  888 888      888888K  
# Y88b  d88P 888     Y88b 888      X88  d88P Y88b        888        Y88b 888 888     Y88..88P Y88b.    888 "88b 
#  "Y8888P"  888      "Y88888  88888P' d88P   Y88b       888         "Y88888 888      "Y88P"   "Y8888P 888  888 
#                         888                                            888                                    
#                    Y8b d88P                                       Y8b d88P                                    
#                     "Y88P"                                         "Y88P"                                       
from re import T
import pyfock.Mol as Mol
import pyfock.Basis as Basis
import pyfock.Integrals as Integrals
import pyfock.Grids as Grids
from threadpoolctl import ThreadpoolController, threadpool_info, threadpool_limits
controller = ThreadpoolController()
import numpy as np
from numpy.linalg import eig, multi_dot as dot
import scipy 
from timeit import default_timer as timer
import numba
from opt_einsum import contract
# import sparse
# import dask.array as da
from scipy.sparse import csr_matrix, csc_matrix
# from memory_profiler import profile
import os
from numba import njit, prange, cuda
import numexpr
try:
    import cupy as cp
    from cupy import fuse
    import cupyx
except Exception as e:
    print('Cupy is not installed. GPU acceleration is not availble.')
    pass
