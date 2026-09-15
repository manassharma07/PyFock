"""Small device helpers that replace Python builtins Numba's CUDA target cannot lower.

Numba implements ``max``/``min`` as a variadic ``@overload`` whose implementation is a
``def impl(*x)`` device function.  The CUDA target compiles device functions by signature and
rejects the ``*args`` form ("Signature mismatch: N argument types given, but function takes 1
arguments"), so *any* use of ``max`` or ``min`` inside a ``@cuda.jit`` function fails to
compile, whatever the number of arguments.  These helpers are the explicit equivalents; they
inline to a single comparison and are exact for the integers the kernels use them on.
"""
from numba import cuda

__all__ = ['imax', 'imin', 'imax3']


@cuda.jit(device=True, cache=True)
def imax(a, b):
    """``max(a, b)``."""
    return a if a > b else b


@cuda.jit(device=True, cache=True)
def imin(a, b):
    """``min(a, b)``."""
    return a if a < b else b


@cuda.jit(device=True, cache=True)
def imax3(a, b, c):
    """``max(a, b, c)``."""
    m = a if a > b else b
    return m if m > c else c
