# DF_algo=11 on the GPU: measured results

Companion to [df_algo11_gpu.md](df_algo11_gpu.md). All numbers below were measured on
one machine:

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 5070 (12 GB, WDDM), driver 591.86 (CUDA 13.1), CUDA runtime 12.8 |
| Software | CuPy 12.3.0, Numba 0.61.2, Python 3.10, Windows 11 |
| CPU | Intel i9-12900K, 4 Numba threads for every CPU number |
| Settings | Schwarz threshold 1e-9, auxiliary basis def2-universal-jfit, fp64 throughout |

The kernel and SCF numbers were recorded by the first implementation session and
re-measured independently afterwards (caffeine CAO/SAO and icosane kernel benchmarks,
caffeine and decane SCF runs): the re-measurements agree with the tables within run-to-run
noise (build times within 8 %, energies identical to the digits shown).

Kernel numbers come from `benchmarks_tests/benchmark_DF_algo11_kernels.py` (warm
compilation cache, best of 3 repeats, each GPU operation synchronised); SCF numbers
come from `benchmarks_tests/benchmark_DF_algos_scf.py` with native level-3 grids,
LDA (Slater + VWN5), `XC_algo=2`, `conv_crit=1e-7`. The commands are listed at the end.

## 1. Accuracy: GPU algorithm 11 against CPU algorithm 11

The CPU plan is the oracle: both builders share the host-side screening, ranking and
cache selection (`_plan_metadata`), so the stored block sets are identical and every
stored value can be compared element by element.

| System (AO / aux) | AO type, strict | max abs error: stored values | gamma | J |
|---|---|---|---|---|
| H2O def2-SVP (24 / 85) | CAO, strict | 9.8e-15 | 1.4e-14 | 1.5e-14 |
| H2O def2-SVP | SAO, no strict | 9.8e-15 | 7.1e-15 | 1.3e-14 |
| Decane def2-SVP (260 / 874) | CAO, strict | 1.5e-14 | 1.6e-12 | 2.0e-14 |
| Decane def2-SVP | SAO, no strict | 1.5e-14 | 1.4e-12 | 2.1e-14 |
| Caffeine def2-SVP (260 / 974) | CAO, strict | 2.0e-14 | 1.5e-12 | 4.4e-14 |
| Caffeine def2-SVP | SAO, no strict | 2.0e-14 | 1.8e-12 | 3.1e-14 |
| Serotonin def2-SVP | CAO, strict | 1.6e-14 | 3.0e-12 | 4.1e-14 |
| Serotonin def2-SVP | SAO, no strict | 1.6e-14 | 2.5e-12 | 3.4e-14 |
| Icosane def2-SVP (522 / 1746) | CAO, strict | 4.0e-14 | 5.3e-12 | 4.9e-14 |
| Icosane def2-SVP | SAO, no strict | 4.0e-14 | 5.3e-12 | 5.5e-14 |
| Water cluster (20 H2O) def2-SVP | CAO, strict | 6.5e-13 | 1.1e-11 | 2.2e-12 |
| Water cluster (20 H2O) def2-SVP | SAO, no strict | 6.5e-13 | 9.0e-12 | 2.2e-12 |
| Caffeine def2-TZVP (564 / 974, f) | CAO, strict | 7.0e-14 | 1.2e-11 | 1.7e-13 |
| Caffeine def2-TZVP | SAO, no strict | 6.7e-14 | 1.0e-11 | 1.2e-13 |
| H2O def2-QZVP (142 / 85, g) | CAO, strict | 9.2e-14 | 1.1e-12 | 7.0e-14 |
| H2O def2-QZVP | SAO, no strict | 9.2e-14 | 1.4e-12 | 6.1e-14 |

gamma is a sum over up to 10^5 terms per auxiliary function accumulated with atomics
in a different order than on the CPU; 1e-11 absolute on values of order 10 to 100 is
fp64 rounding. The unit tests (`tests/test_df_algo11_cupy.py`) require 1e-12 on the
stored values and 1e-10 on gamma and J, including h and i shells on either side, the
i-i-i cooperative kernel, partial and zero cache budgets and non-default streams.

## 2. Speed: GPU algorithm 11 against GPU algorithm 10

Kernel-level timings (seconds; "build" includes plan metadata and device setup for
algorithm 11 and the offsets-based launch for algorithm 10).

| System | AO type, strict | build 10 | build 11 | speed-up | gamma 10 -> 11 (ms) | J 10 -> 11 (ms) |
|---|---|---|---|---|---|---|
| H2O def2-SVP | CAO, strict | 0.080 | 0.012 | 7x | 1.12 -> 0.64 | 1.17 -> 0.66 |
| Decane def2-SVP | CAO, strict | 1.33 | 0.075 | 18x | 3.3 -> 1.6 | 1.8 -> 0.9 |
| Decane def2-SVP | SAO, no strict | 1.42 | 0.096 | 15x | 3.7 -> 1.4 | 2.0 -> 0.9 |
| Caffeine def2-SVP | CAO, strict | 1.98 | 0.083 | 24x | 4.3 -> 1.8 | 2.1 -> 1.0 |
| Caffeine def2-SVP | SAO, no strict | 2.07 | 0.104 | 20x | 4.5 -> 1.4 | 2.2 -> 1.0 |
| Serotonin def2-SVP | CAO, strict | 1.70 | 0.080 | 21x | 3.6 -> 1.7 | 2.0 -> 0.9 |
| Serotonin def2-SVP | SAO, no strict | 1.85 | 0.099 | 19x | 4.1 -> 1.4 | 2.1 -> 1.0 |
| Icosane def2-SVP | CAO, strict | 4.55 | 0.297 | 15x | 7.0 -> 4.4 | 4.3 -> 1.7 |
| Icosane def2-SVP | SAO, no strict | 4.87 | 0.390 | 12x | 6.8 -> 3.8 | 5.6 -> 1.8 |
| Water cluster (20 H2O) | CAO, strict | 4.57 | 0.308 | 15x | 5.5 -> 4.4 | 4.4 -> 1.6 |
| Water cluster (20 H2O) | SAO, no strict | 5.12 | 0.437 | 12x | 7.2 -> 4.1 | 5.4 -> 1.8 |
| Caffeine def2-TZVP (f) | CAO, strict | n/a | 0.364 | n/a | n/a -> 5.2 | n/a -> 2.1 |
| Caffeine def2-TZVP (f) | SAO, no strict | n/a | 0.420 | n/a | n/a -> 3.8 | n/a -> 2.1 |
| H2O def2-QZVP (g) | CAO, strict | n/a | 0.256 | n/a | n/a -> 0.75 | n/a -> 0.66 |

"n/a": the algorithm-10 CUDA kernels have fixed 5x5 recurrence buffers and five root
slots and return wrong values for orbital f and g shells (they were never valid there;
the wrapper now raises for orbital `l > 2`, auxiliary `l > 4` or more than seven
auxiliary primitives). Algorithm 11 supports orbital and auxiliary shells through `l = 6`.

The GPU build is 2.5 to 5x faster than the 4-thread CPU algorithm 11 build on the same
machine for the def2-SVP systems (caffeine 0.38 s -> 0.08 s), and the direct
(zero-cache) gamma or J pass costs about 0.6 to 0.9 of one build.

### Full SCF (native grids, 4 host threads)

Wall times in seconds from the driver's profiling block; "3c2e" is the
"Coulomb Integrals (2c2e + 3c2e)" line, which also contains the 2c2e integrals, the
Schwarz diagonal and (for algorithm 11) the host-side planning. gamma and J are the
totals over all SCF iterations.

| System, AO type | algorithm | total energy (Hartree) | 3c2e | gamma | J | peak CuPy pool |
|---|---|---|---|---|---|---|
| H2O, CAO | CPU 11 | -75.7995608151 | 0.09 | 0.04 | 0.05 | |
| | GPU 10 | -75.7995608151 | 0.33 | 0.03 | 0.03 | 10 MB |
| | GPU 11 | -75.7995608151 | 0.40 | 0.01 | 0.01 | 4 MB |
| Decane, CAO | CPU 11 | -390.3133364307 | 0.54 | 0.13 | 0.13 | |
| | GPU 10 | -390.3133364306 | 1.62 | 0.07 | 0.04 | 202 MB |
| | GPU 11 | -390.3133364306 | 0.50 | 0.03 | 0.02 | 239 MB |
| Decane, SAO | CPU 11 | -390.2864219416 | 0.69 | 0.14 | 0.13 | |
| | GPU 10 | -390.2864219414 | 1.73 | 0.07 | 0.04 | 224 MB |
| | GPU 11 | -390.2864219414 | 0.60 | 0.03 | 0.01 | 252 MB |
| Caffeine, CAO | CPU 11 | -674.3223383492 | 0.85 | 0.19 | 0.18 | |
| | GPU 10 | -674.3223383495 | 2.21 | 0.09 | 0.05 | 256 MB |
| | GPU 11 | -674.3223383495 | 0.52 | 0.04 | 0.02 | 298 MB |
| Caffeine, SAO | CPU 11 | -674.2804942599 | 0.95 | 0.29 | 0.27 | |
| | GPU 10 | -674.2804942602 | 2.48 | 0.12 | 0.06 | 288 MB |
| | GPU 11 | -674.2804942600 | 0.56 | 0.04 | 0.02 | 310 MB |
| Serotonin, CAO | CPU 11 | -567.6483898036 | 0.88 | 0.23 | 0.27 | |
| | GPU 10 | -567.6483898036 | 1.96 | 0.09 | 0.06 | 244 MB |
| | GPU 11 | -567.6483898036 | 0.57 | 0.05 | 0.03 | 283 MB |
| Serotonin, SAO | CPU 11 | -567.6114007174 | 0.71 | 0.20 | 0.18 | |
| | GPU 10 | -567.6114007171 | 2.17 | 0.09 | 0.05 | 269 MB |
| | GPU 11 | -567.6114007171 | 0.56 | 0.04 | 0.02 | 292 MB |
| Icosane, CAO | CPU 11 | -779.4543287471 | 2.90 | 0.51 | 0.54 | |
| | GPU 10 | -779.4543287464 | 4.92 | 0.14 | 0.10 | 740 MB |
| | GPU 11 | -779.4543287464 | 1.13 | 0.14 | 0.04 | 909 MB |
| Icosane, SAO | CPU 11 | -779.3995877383 | 1.96 | 0.74 | 0.71 | |
| | GPU 10 | -779.3995877369 | 5.53 | 0.27 | 0.17 | 859 MB |
| | GPU 11 | -779.3995877367 | 1.23 | 0.16 | 0.05 | 974 MB |

GPU algorithm 11 reproduces the CPU algorithm 11 total energies to 1e-10 Hartree for
the def2-SVP systems up to caffeine and to 2e-9 Hartree for icosane (the same
difference as GPU algorithm 10; GPU and CPU differ in the XC quadrature order as well,
not only in the Coulomb term). GPU algorithms 10 and 11 agree to 3e-10 Hartree.

With 4 host threads the SCF wall time is dominated by the hybrid XC evaluation
(`XC_algo=2`, 60 to 80 % of the total), so the Coulomb speed-up changes the total SCF
time by only a few percent in these runs.

## 3. Memory

| System | AO type | stored 10 | stored 11 | ratio | peak pool 10 | peak pool 11 |
|---|---|---|---|---|---|---|
| Caffeine def2-SVP | CAO | 174.2 MB | 182.3 MB | +4.7 % | 176 MB | 218 MB |
| Caffeine def2-SVP | SAO | 203.4 MB | 189.6 MB | -6.8 % | 205 MB | 228 MB |
| Decane def2-SVP | CAO | 128.6 MB | 132.6 MB | +3.1 % | 131 MB | 168 MB |
| Decane def2-SVP | SAO | 149.0 MB | 140.8 MB | -5.5 % | 151 MB | 179 MB |
| Icosane def2-SVP | CAO | 555.3 MB | 582.0 MB | +4.8 % | 563 MB | 733 MB |
| Icosane def2-SVP | SAO | 667.9 MB | 624.8 MB | -6.5 % | 675 MB | 791 MB |
| Water cluster (20 H2O) | CAO | 522.6 MB | 556.2 MB | +6.4 % | 530 MB | 718 MB |
| Water cluster (20 H2O) | SAO | 731.1 MB | 627.1 MB | -14 % | 738 MB | 818 MB |
| Caffeine def2-TZVP | CAO | 829.6 MB | 857.6 MB | +3.4 % | n/a | 964 MB |
| Caffeine def2-TZVP | SAO | 947.8 MB | 872.0 MB | -8.0 % | n/a | 983 MB |

The stored integrals behave like the CPU version (+3 to 6 % in CAO because whole shell
blocks are kept, 5 to 14 % less in SAO because only projected whole shells are stored).
The *total* device footprint of algorithm 11 is however 20 to 35 % above the stored
values because of its device metadata: the int32 work items (`(pair, aux shell,
column)`, 12 bytes each), the per-pair `column -> aux function` map used by the J kernel
(4 bytes per stored column of every shell pair, i.e. up to 50 % of an s-s block) and the
row list for J. Algorithm 10 has almost no metadata beyond `offsets`. If device memory
becomes the binding constraint, the column map is the first thing to remove (replay the
shell test per warp, or store one entry per significant aux *shell* instead of per
function); the `max_memory_ints3c2e` budget trades cached values for direct batches.

## 4. Kernel tiers and angular-momentum classes

Diagnostic per-class launches (`--classes`) on the same GPU:

| basis | classes present | tier A (one thread per item) | tier B (128 threads per item) | registers / thread | local memory / thread |
|---|---|---|---|---|---|
| def2-SVP + jfit | 45 | 45 | 0 | 210 to 255 | up to 10.4 KB (d-d-g) |
| def2-TZVP + jfit | 80 | 73 | 7 (l_A + l_B >= 5 with aux g, f-f-d and above) | A: 210 to 255, B: 161 to 162 | A: up to 10.4 KB; B: <= 192 B plus <= 2.9 KB shared |
| def2-QZVP + jfit | 125 | 102 | 23 | as above | B: <= 432 B plus <= 4.2 KB shared |

Tier B is chosen for blocks with more than 540 Cartesian elements; the cooperative
kernel keeps the roots, recurrences and shift tables in shared memory and each thread
accumulates a strided subset of the block, so even i-i-i (21952 elements) never
allocates a whole block per thread. In production all items with orbital `l <= 2` and
auxiliary `l <= 4` are launched together through the d-d-g kernel: on def2-SVP this is
one launch instead of 45 and was measured faster (0.08 s versus 0.15 s for caffeine)
because the launch and argument-adaptation overhead of dozens of small launches
outweighed the smaller per-class scratch. All 343 class kernels are generated by
`tools/generate_df_algo11_cuda.py`; only the ones a basis pair needs are compiled.

## 5. Process-start failures seen during the first benchmark session

Nine of the 40 GPU SCF processes launched by the first session's benchmark loop died at
their first Numba CUDA call (`cuda.get_current_device()` in `DFT.__init__`, retried in
`DFT.scf`) with

```
numba.cuda.cudadrv.driver.CudaAPIError: [201] Call to cuCtxGetDevice results in CUDA_ERROR_INVALID_CONTEXT
```

before any PyFock kernel ran; both algorithms 10 and 11 were affected, and every
affected run passed when repeated. The failures were clustered (20:58:50 to 20:59:04,
21:06 to 21:12) and never occurred after 21:13 when the same loop kept running for a
further two hours. What is known:

* That loop ran as a background job of the agent's shell while the agent started
  other GPU work (tests, kernel benchmarks) in the foreground, so CUDA processes were
  created and torn down concurrently on one WDDM display GPU.
* The Windows System log holds a WHEA "corrected hardware error, PCI Express Root Port
  0:1.0 (the GPU's port), Advanced Error Reporting" at 20:58:24, 26 seconds before the
  first failure; the same event recurs every few days on this machine, and the active
  power plan keeps PCIe Link State Power Management at "Moderate power savings".
* The failure could not be reproduced afterwards with the same scripts: 16 sequential
  back-to-back starts (GPU after GPU and GPU after CPU runs), 6 starts while another
  GPU SCF ran, 12 starts from two parallel loops and 6 more with the GPU test suite in
  the background all succeeded, as did 32 probe processes that query Numba, then CuPy,
  then Numba again.

Changes made in response: `DFT.__init__` no longer touches CUDA for CPU runs (it used
to create a CUDA context in every process, GPU or not) and prints a warning instead of
silently swallowing a failed device query, so a recurrence shows up at the first call.
Recommended on this machine: do not run several CUDA processes at once, set PCIe Link
State Power Management to "Off", and watch the WHEA events if the error returns.

## 6. Commands

```sh
python -m pytest tests/test_df_algo11.py tests/test_df_algo11_cupy.py
python benchmarks_tests/benchmark_DF_algo11_kernels.py benchmarks_tests/Caffeine.xyz def2-SVP 4 --gpu --cpu-reference --repeats 3
python benchmarks_tests/benchmark_DF_algo11_kernels.py benchmarks_tests/Caffeine.xyz def2-SVP 4 --gpu --cpu-reference --sao --nostrict --repeats 3
python benchmarks_tests/benchmark_DF_algo11_kernels.py benchmarks_tests/Caffeine.xyz def2-TZVP 4 --gpu --cpu-reference --classes
python benchmarks_tests/benchmark_DF_algos_scf.py benchmarks_tests/Caffeine.xyz def2-SVP cao strict 1e-9 11 none 4 --gpu --native-grids
python benchmarks_tests/benchmark_DF_algos_scf.py benchmarks_tests/Caffeine.xyz def2-SVP cao strict 1e-9 10 none 4 --gpu --native-grids
python tests/test gpu --case 'h2o_*' --ncores 4
```

Run each SCF twice and report the second run (warm Numba and CuPy caches).
