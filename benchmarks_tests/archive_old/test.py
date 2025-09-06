import ctypes
import os

# Force load OpenMP shared library
libomp_path = "/opt/homebrew/opt/libomp/lib/libomp.dylib"
if os.path.exists(libomp_path):
    ctypes.CDLL(libomp_path, mode=ctypes.RTLD_GLOBAL)
else:
    raise RuntimeError("libomp.dylib not found.")

# Now import Numba
import numba
print("Threading backend:", numba.threading_layer())

