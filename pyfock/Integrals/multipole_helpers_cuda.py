"""CUDA device versions of the solid-harmonic recurrences of :mod:`~pyfock.Integrals.multipole_helpers`.

Only the recurrence the per-iteration far field needs is here (the irregular harmonics of the
branch-atom vectors); the translation tables, the Gaussian product moments and their
branch-centred translations are built once on the host.  ``out`` is written at
``base + harmonic_index(l, m)``, ``harmonic_index(l, m) = l*l + l + m``, so one shared-memory
block can hold one vector per warp without array slicing inside the kernel.

The recurrence is sequential in the degree ``l`` but the ``2l + 1`` components of one degree are
independent, so the cooperative variants below spread them over the lanes of a warp (or the
threads of a block) and synchronize once per degree.  That turns ``(lmax+1)^2`` serial steps into
``lmax`` of them - worth having, because the branch-atom coupling evaluates one of these vectors
per (branch, atom) pair and a single-lane recurrence would leave the rest of the warp waiting.
Each component is computed by exactly the arithmetic of the serial CPU function, in the same
order, so the results are bit-identical to :func:`pyfock.Integrals.multipole_helpers.irregular_harmonics`.
"""
import math

from numba import cuda

__all__ = ['regular_harmonics', 'irregular_harmonics',
           'irregular_harmonics_warp', 'irregular_harmonics_block']

SQRT2 = 1.4142135623730951


@cuda.jit(device=True, cache=True)
def regular_harmonics(x, y, z, lmax, out, base):
    """Real scaled regular solid harmonics ``R_lm(x, y, z)``, ``l <= lmax``, into ``out[base:]``."""
    rr = x * x + y * y + z * z
    out[base] = 1.0
    if lmax == 0:
        return
    out[base + 1] = -y * SQRT2 * 0.5
    out[base + 2] = z
    out[base + 3] = -x * SQRT2 * 0.5
    for l in range(2, lmax + 1):
        b = base + l * l + l
        pb = base + (l - 1) * (l - 1) + (l - 1)
        ppb = base + (l - 2) * (l - 2) + (l - 2)
        re_p = out[pb + (l - 1)]
        im_p = out[pb - (l - 1)]
        out[b + l] = -(x * re_p - y * im_p) / (2.0 * l)
        out[b - l] = -(x * im_p + y * re_p) / (2.0 * l)
        for m in range(0, l):
            denom = 1.0 / ((l - m) * (l + m))
            t = (2.0 * l - 1.0) * z * out[pb + m]
            if l - 2 >= m:
                t -= rr * out[ppb + m]
            out[b + m] = t * denom
            if m > 0:
                t = (2.0 * l - 1.0) * z * out[pb - m]
                if l - 2 >= m:
                    t -= rr * out[ppb - m]
                out[b - m] = t * denom


@cuda.jit(device=True, cache=True)
def irregular_harmonics(x, y, z, lmax, out, base):
    """Real scaled irregular solid harmonics ``I_lm(x, y, z)``, ``l <= lmax``, into ``out[base:]``.

    The distance must be non-zero; the far-field classification guarantees that for every
    (branch, atom) pair the callers pass.
    """
    rr = x * x + y * y + z * z
    inv_rr = 1.0 / rr
    out[base] = math.sqrt(inv_rr)
    if lmax == 0:
        return
    out[base + 1] = -y * inv_rr * out[base] * SQRT2
    out[base + 2] = z * inv_rr * out[base]
    out[base + 3] = -x * inv_rr * out[base] * SQRT2
    for l in range(2, lmax + 1):
        b = base + l * l + l
        pb = base + (l - 1) * (l - 1) + (l - 1)
        ppb = base + (l - 2) * (l - 2) + (l - 2)
        re_p = out[pb + (l - 1)]
        im_p = out[pb - (l - 1)]
        f = (2.0 * l - 1.0) * inv_rr
        out[b + l] = -f * (x * re_p - y * im_p)
        out[b - l] = -f * (x * im_p + y * re_p)
        for m in range(0, l):
            t = (2.0 * l - 1.0) * z * out[pb + m]
            if l - 2 >= m:
                t -= ((l - 1) * (l - 1) - m * m) * out[ppb + m]
            out[b + m] = t * inv_rr
            if m > 0:
                t = (2.0 * l - 1.0) * z * out[pb - m]
                if l - 2 >= m:
                    t -= ((l - 1) * (l - 1) - m * m) * out[ppb - m]
                out[b - m] = t * inv_rr


@cuda.jit(device=True, cache=True)
def irregular_harmonics_warp(x, y, z, lmax, out, base, lane):
    """:func:`irregular_harmonics` with the components of each degree spread over a warp.

    Every lane of the warp must call this with the same arguments; it synchronizes the warp
    once per degree.  Within a degree every component reads only degrees ``l-1`` and ``l-2``,
    which the previous synchronization has finalized, so no intra-degree ordering is needed.
    """
    rr = x * x + y * y + z * z
    inv_rr = 1.0 / rr
    if lane == 0:
        out[base] = math.sqrt(inv_rr)
    cuda.syncwarp()
    if lmax == 0:
        return
    if lane == 0:
        out[base + 1] = -y * inv_rr * out[base] * SQRT2
        out[base + 2] = z * inv_rr * out[base]
        out[base + 3] = -x * inv_rr * out[base] * SQRT2
    cuda.syncwarp()
    for l in range(2, lmax + 1):
        b = base + l * l + l
        pb = base + (l - 1) * (l - 1) + (l - 1)
        ppb = base + (l - 2) * (l - 2) + (l - 2)
        re_p = out[pb + (l - 1)]
        im_p = out[pb - (l - 1)]
        f = (2.0 * l - 1.0) * inv_rr
        if lane == 0:
            out[b + l] = -f * (x * re_p - y * im_p)
            out[b - l] = -f * (x * im_p + y * re_p)
        for m in range(lane, l, 32):
            t = (2.0 * l - 1.0) * z * out[pb + m]
            if l - 2 >= m:
                t -= ((l - 1) * (l - 1) - m * m) * out[ppb + m]
            out[b + m] = t * inv_rr
            if m > 0:
                t = (2.0 * l - 1.0) * z * out[pb - m]
                if l - 2 >= m:
                    t -= ((l - 1) * (l - 1) - m * m) * out[ppb - m]
                out[b - m] = t * inv_rr
        cuda.syncwarp()


@cuda.jit(device=True, cache=True)
def irregular_harmonics_block(x, y, z, lmax, out, base, tid, nthreads):
    """:func:`irregular_harmonics_warp` for a whole block (``cuda.syncthreads`` per degree)."""
    rr = x * x + y * y + z * z
    inv_rr = 1.0 / rr
    if tid == 0:
        out[base] = math.sqrt(inv_rr)
    cuda.syncthreads()
    if lmax == 0:
        return
    if tid == 0:
        out[base + 1] = -y * inv_rr * out[base] * SQRT2
        out[base + 2] = z * inv_rr * out[base]
        out[base + 3] = -x * inv_rr * out[base] * SQRT2
    cuda.syncthreads()
    for l in range(2, lmax + 1):
        b = base + l * l + l
        pb = base + (l - 1) * (l - 1) + (l - 1)
        ppb = base + (l - 2) * (l - 2) + (l - 2)
        re_p = out[pb + (l - 1)]
        im_p = out[pb - (l - 1)]
        f = (2.0 * l - 1.0) * inv_rr
        if tid == 0:
            out[b + l] = -f * (x * re_p - y * im_p)
            out[b - l] = -f * (x * im_p + y * re_p)
        for m in range(tid, l, nthreads):
            t = (2.0 * l - 1.0) * z * out[pb + m]
            if l - 2 >= m:
                t -= ((l - 1) * (l - 1) - m * m) * out[ppb + m]
            out[b + m] = t * inv_rr
            if m > 0:
                t = (2.0 * l - 1.0) * z * out[pb - m]
                if l - 2 >= m:
                    t -= ((l - 1) * (l - 1) - m * m) * out[ppb - m]
                out[b - m] = t * inv_rr
        cuda.syncthreads()
