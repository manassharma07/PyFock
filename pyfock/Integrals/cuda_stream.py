"""One long-lived CUDA stream per device, shared by the gradient routines.

Two reasons not to make a stream per call.

*Ordering.* Several of the CuPy integral routines allocate and fill their packed basis arrays with
whatever stream happens to be current, and only then create the non-blocking stream they launch the
kernel on. Those are different streams, so nothing orders the host-to-device writes before the
kernel that reads them, and the kernel can see uninitialised exponents -- which surfaces as an
occasional NaN rather than an error. Making the shared stream current *before* calling them, and
passing it in, puts the uploads and the kernel on the same stream.

*Memory pool.* CuPy keeps its free lists keyed by stream pointer. A block freed on a stream that is
then destroyed stays on a list belonging to a handle the driver is free to hand out again, so a
later stream can be given memory that an earlier one has not finished with. A stream that is created
once and never destroyed cannot get into that state.

Use it as::

    cp_stream, nb_stream = gradient_stream()
    with cp_stream:
        ...                      # CuPy work
        kernel[blocks, threads, nb_stream](...)
    cp_stream.synchronize()
"""
try:
    import cupy as cp
except Exception:                                  # pragma: no cover - CPU-only install
    cp = None
from numba import cuda

__all__ = ['gradient_stream']

_STREAMS = {}


def gradient_stream(device=None):
    """The shared ``(cupy.cuda.Stream, numba.cuda.Stream)`` pair for ``device`` (default: current)."""
    if cp is None:
        raise RuntimeError('CuPy is required for gradient_stream.')
    if device is None:
        device = cp.cuda.Device().id
    pair = _STREAMS.get(device)
    if pair is None:
        with cp.cuda.Device(device):
            cp_stream = cp.cuda.Stream(non_blocking=True)
        pair = (cp_stream, cuda.external_stream(cp_stream.ptr))
        _STREAMS[device] = pair
    return pair
