# Shell-blocked DF-J on CUDA

Algorithm 11 is available on CPU and GPU. The GPU path uses fp64 values, the
same CPU plan metadata and cache ranking, and the same row-block layout. A pair
owns lower-triangular rows within a diagonal shell, or all Cartesian rows for
different shells. Its columns are complete significant auxiliary shells.

## Implementation decisions

`df_algo11_helpers._plan_metadata` performs packing, screening, cost ranking and
cache selection without allocating or evaluating integral values. Both builders
call it. The GPU builder constructs integer `(pair, aux shell, column)` work
items in compiled linear passes and stably counting-sorts their small integer
angular-momentum/primitive-cost keys. This avoids costly general-purpose sorting
and thousands of small NumPy allocations.

The generated module contains cacheable kernels for every `(lA,lB,lC)` through
`(6,6,6)`. The generator is `tools/generate_df_algo11_cuda.py`. Only used kernels
are compiled. Root, recurrence and shift work is shared across all components
of a primitive shell triple; the per-function Coulomb routine is never called.

Measured dispatch:

* Common items with orbital `l<=2` and auxiliary `l<=4` share one bounded d-d-g
  kernel launch. Exact-class launches were slower in measured small/medium
  systems because argument adaptation and launch overhead outweighed smaller
  scratch. The common kernel does **not** allocate buffers sized for h/i.
* Remaining blocks with at most 540 Cartesian elements use one thread per item
  and exact-class local scratch.
* Larger blocks use 128 threads per item, shared roots/recurrences/shifts, and
  strided local component accumulators. This supports i-i-i without allocating
  an entire 21,952-element block in every thread.
* Auxiliary SAO projection is a separate row-owned pass. It projects complete
  shells in place without concurrent reads and writes of the same row.

Gamma uses eight lanes per auxiliary-shell item and one atomic addition per
column. J uses one warp per AO-pair row, contiguous integral reads and a warp
reduction. A shared-per-pair int32 column map avoids replaying a shell loop in
every J warp. Both contractions retain the strict per-function-pair cutoff.

CuPy and Numba share explicit zero-copy device views. Internal operations stay
on the plan stream, avoiding redundant synchronization for every kernel
argument. Public calls synchronize before returning and restore the caller's
stream/device context. Newly added kernels do not enable fastmath; the existing
Rys device helpers retain their original compilation settings. Numerical tests,
not assumptions about floating-point flags, establish the tolerances below.

## Memory

`max_memory_gb` (driver: `max_memory_ints3c2e`) bounds cached device values.
`None` caches all selected pairs, zero caches none. Uncached pairs are batched
in both gamma and J using temporary offsets and a reusable buffer. The default
temporary capacity is at most 1 GB and one quarter of free device memory. A
single whole row block must fit; `batch_memory_gb` can override this capacity.
The plan reports the actual largest batch, separately from cached storage.

`keep_ints3c2e_in_gpu=False` does not move cached algorithm-11 values to the host;
the driver prints this explicitly. Use the cache budget to limit device storage.
Host and GPU cache selection, offsets and values remain directly comparable.

Work items, the column map and row-work arrays consume additional device memory.
Therefore the stored-value ratio alone is not the total memory ratio. Benchmark
JSON includes CuPy allocation high-water marks, pool reservations, free-device
samples, and separate direct-mode peaks. The allocator hook excludes allocations
made directly by CUDA libraries/other processes; free-device samples include
them but are not a continuous process-memory peak.

## Validation and baseline limits

The GPU suite compares every stored value with CPU algorithm 11 at **1e-12
absolute tolerance**, and gamma/J at **1e-10 absolute tolerance**. It covers
two separated waters, strict on/off, thresholds 1e-9/1e-12, CAO/SAO, full/partial/
zero cache, forced multiple direct batches, non-default streams, empty plans,
f/g molecular bases and normalized h/i shells on either side, including i-i-i
at noncoincident centers. Debug mode independently checks device column counts,
column offsets, screening decisions and finiteness.

The original GPU algorithm-10 kernels have fixed 5x5 recurrence arrays, five
root slots and seven auxiliary-primitive slots. They return invalid values for
orbital f/g shells in **both CAO and SAO**. The public wrapper now rejects those
unsupported cases. Benchmarks mark this baseline unavailable; apparent speedups
against its invalid larger-basis results must not be used. CPU algorithm 11
remains the reference for those systems.

The driver's legacy GPU Schwarz diagonal also lacks the CPU analytic fallback
for more than ten roots. Algorithm 11 therefore obtains h/i orbital screening
bounds from the CPU diagonal; three-center builds and contractions stay on CUDA.

The SCF benchmark's requested GPU XC algorithm 2 previously referenced an
undefined `use_libxc`. The driver now forwards that setting; native LDA/GGA
functional results are adapted to the hybrid evaluator, and its CPU AO evaluator
receives host indices. This permits `XC_algo=2, use_libxc=False` in GPU comparisons.

## Reproduce

```sh
python -m pytest tests/test_df_algo11.py tests/test_df_algo11_cupy.py
python benchmarks_tests/benchmark_DF_algo11_kernels.py benchmarks_tests/Caffeine.xyz def2-SVP 4 --gpu --cpu-reference --repeats 3
python benchmarks_tests/benchmark_DF_algo11_kernels.py benchmarks_tests/Caffeine.xyz def2-SVP 4 --gpu --cpu-reference --sao --nostrict --repeats 3
python benchmarks_tests/benchmark_DF_algos_scf.py benchmarks_tests/H2O.xyz def2-SVP cao strict 1e-9 11 none 4 --gpu
```

Kernel benchmarks perform an untimed warm-up, then report the best of the
requested repeats. CPU comparisons use the same number of host threads. Build
times include planning and device setup, not just CUDA event time. Use
`--classes` for diagnostic exact-class timings, register counts and local/shared
memory. These diagnostic class launches differ from the grouped common dispatch.

Full SCF defaults to PySCF level-3 grids, as in the supplied benchmark. When
PySCF is unavailable, `--native-grids` explicitly selects native level-3 grids;
all energy comparisons must use the same grid option. Native-grid H2O GPU
regressions in `tests/gpu/h2o_*` use measured CPU references and require no PySCF.
The CH4/def2-QZVP GPU-11 regression cases use the existing CPU references and
require PySCF for matching grids and initial density.

Windows notes (development machine, RTX 5070, CuPy 12.3, CUDA 12.8 toolkit):

* CuPy enables its CUB reduction accelerator by default and compiles those kernels
  with `nvcc`, which needs MSVC's `cl.exe` on `PATH`. If it is not (and CuPy's
  auto-detection fails with setuptools >= 80, which dropped `setuptools.msvc`), every
  uncached reduction such as `cp.isfinite(values).all()` raises
  `CompileException: nvcc fatal : Cannot find compiler 'cl.exe'`. Either run from a
  Visual Studio developer prompt or set `CUPY_ACCELERATORS=` (empty) before starting
  Python; the algorithm-11 kernels themselves are Numba kernels and do not need nvcc.
* PyFock prints a logo with Unicode block characters; redirecting stdout to a file
  needs `PYTHONUTF8=1` (the benchmark scripts now replace unencodable characters).
* Do not start several CUDA processes on the same GPU at once (for example a
  benchmark loop in the background while running the tests): see the note on
  `CUDA_ERROR_INVALID_CONTEXT` in [df_algo11_benchmarks.md](df_algo11_benchmarks.md).

No machine-wide CUDA configuration was changed; the default Numba (`__pycache__`) and
CuPy (`~/.cupy/kernel_cache`) compilation caches are sufficient.

Measured results (accuracy against the CPU plan, timings against GPU algorithm 10,
memory, kernel classes) are collected in [df_algo11_benchmarks.md](df_algo11_benchmarks.md).
