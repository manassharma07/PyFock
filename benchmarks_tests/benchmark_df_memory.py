"""CuPy allocator high-water mark and synchronized runtime memory samples."""
from contextlib import contextmanager


@contextmanager
def device_memory_tracker(enabled=True):
    result = {}
    if not enabled:
        yield result
        return
    import cupy as cp
    from numba import cuda

    # Establish the shared primary context before the first runtime-only
    # memory query (important for CuPy/Numba interoperability on Windows).
    cuda.current_context()

    pool = cp.get_default_memory_pool()

    class Peak(cp.cuda.MemoryHook):
        name = 'df_benchmark_peak'

        def malloc_preprocess(self, device_id, size, mem_size):
            result['peak_pool_used_bytes'] = max(result['peak_pool_used_bytes'], pool.used_bytes() + mem_size)

    cp.cuda.get_current_stream().synchronize()
    free, total = cp.cuda.runtime.memGetInfo()
    result.update(peak_pool_used_bytes=pool.used_bytes(), initial_pool_used_bytes=pool.used_bytes(),
                  initial_device_free_bytes=free, device_total_bytes=total)
    with Peak():
        yield result
    cp.cuda.get_current_stream().synchronize()
    result.update(final_pool_used_bytes=pool.used_bytes(), pool_reserved_bytes=pool.total_bytes(),
                  final_device_free_bytes=cp.cuda.runtime.memGetInfo()[0])
