# XC_algo=3: the exchange-correlation term on the CPU

`XC_algo=3` is the default CPU algorithm for the exchange-correlation (XC) energy, potential
matrix and nuclear gradient since this version. Code: `pyfock/Integrals/eval_xc_3.py` (driver and
grid plan), `pyfock/Integrals/eval_xc_3_helpers.py` (Numba kernels), `pyfock/Integrals/eval_xc_grad_3.py`
(gradient) and `pyfock/XC/pointwise.py` (point-wise meta-GGA and PW91 functionals). It needs
nothing beyond NumPy, SciPy (whose BLAS the kernels call) and Numba, like the rest of PyFock.

```python
dft_obj.XC_algo = 3            # the default on the CPU; 2 and 1 are the previous algorithms
dft_obj.save_ao_values = True  # optional: keep the basis functions on the grid between iterations
```

## 1. Why algorithm 2 was slow and did not scale

Algorithm 2 (`eval_xc_2`) cuts the grid into batches of 5000 consecutive points and gives each
batch, in a joblib thread, the list of basis functions whose radial cutoff reaches at least one of
its points. Three things limit it:

1. **Batch size.** A batch of 5000 points of the box-sorted grid spans a region that is reached by
   about three times as many functions as a single point (307 on average against 111 for the
   47-water cluster in def2-SVP). The two dense matrix products per batch, `ao @ D` for the density
   and `aow^T @ ao` for the potential, scale with the square of that number, so they are about an
   order of magnitude larger than they need to be.
2. **Basis function values.** Every function was evaluated separately, with a `pow` and an `exp`
   per primitive and point; the six Cartesian components of a d shell recomputed the same
   exponentials six times. On one core this was 30 of the 44 seconds of a PBE evaluation.
3. **Threads.** The joblib workers run NumPy, `numexpr` (with its own thread pool) and `opt_einsum`
   on transposed views; the GIL, the nested thread pools and the memory traffic of the per-batch
   temporaries roughly triple the per-thread work at 24 threads. The speed-up from 1 to 24 cores
   was 6.9x (28% efficiency) without the AO cache and 2.5x (10%) with it.

## 2. What algorithm 3 does

**Leaves.** The grid is ordered by recursive median bisection (a k-d tree, longest extent first)
into *leaves* of at most 256 points (`leaf_size`), spatially compact in every region of the grid.
Each leaf gets the list of shells whose cutoff sphere reaches one of its points: a bounding sphere
of the leaf settles most shells, the few that straddle the cutoff are checked point by point, so
the criterion is exactly the one of algorithm 2 at the finer granularity. For the 47-water cluster
a leaf holds 156 points and 135 basis functions on average (the per-point count is 111); algorithm 2
had 307 per batch.

**Macro blocks.** Sixteen consecutive leaves (a sub-tree) form a macro block; the block gathers its
sub-matrix of the density matrix once and accumulates its part of the potential matrix once, so the
leaves work on L2/L3-resident copies with relative indices.

**One parallel region.** The macro blocks are assigned to the threads in advance (longest processing
time first, with a cost model of 4 k n² flops per leaf for the matrix products plus a linear term), and
a single Numba `prange` over the threads does everything: basis function values and gradients (shell
by shell, the exponentials shared by the Cartesian components, primitives dropped point by point
once `alpha r² > 46`), the density, the functional, and the two matrix products per leaf, which call
BLAS through `np.dot` with one BLAS thread each. No Python runs inside the region; the per-thread
workspaces are allocated once per grid and reused.

**Deterministic assembly.** Every macro block writes its unsymmetrized potential-matrix contribution
`M` (with `V = M + M^T`) into its own slot of a flat store; the full matrix is assembled afterwards
by row bands, each band written by one thread in a fixed order. The result is bit-for-bit
independent of the number of threads, and no thread writes to another thread's rows. The gradient
kernel does the same with per-block slots of the per-function gradient and of the grid-translation
term.

**Functionals.** All natively implemented functionals are evaluated inside the kernel in a single
pass over the grid: LDA and GGA functionals through their Numba implementations (VWN, VWN-RPA, PW,
B88 and LYP were Numba-compiled for this), the meta-GGAs (TPSS, M06-L, r2SCAN, TASK) and PW91
correlation point by point through `pyfock/XC/pointwise.py`, scalar transcriptions of the array
implementations with the same finite-difference and complex-step derivatives (the array versions
remain for CuPy; `tests/test_eval_xc_3.py` checks that the two agree). Global hybrids use their
semilocal components. LibXC functionals (`use_libxc=True`) are evaluated on the whole grid between
two passes of the kernel (density, then potential), the functional itself chunked over the threads.

**AO cache.** With `save_ao_values=True` the values (LDA) or values and gradients (GGA, meta-GGA)
of the basis functions are kept leaf by leaf in exactly the layout the kernel reads, so the cache
holds only the per-leaf lists (135 functions per point here against 307 in the per-batch cache of
algorithm 2) and is used without any copying. Since the recomputation is cheap, the cache pays off
only where memory bandwidth per core is ample (few cores); on a workstation whose 24 threads share
the memory bus it can be slower than recomputing (see the benchmarks).

**Gradient.** `eval_xc_grad_3` uses the same plan with the basis functions' first and second
derivatives (`grid_response=True` and the same Pulay, translation and Becke-weight terms as
`eval_xc_grad_2`). `DFT_Grad` reuses the plan the SCF built.

**Accuracy.** Compared with an evaluation without any screening, the energies agree to ~5e-9 Ha for
the 47-water cluster (-433 Ha of XC energy); the difference comes from the tails of the functions
beyond their 1e-9 cutoff radius, which algorithm 2 includes for the functions of a batch and
algorithm 3 drops point by point. Algorithms 2 and 3 agree to 1e-8 Ha in the XC energy, 1e-8 in the
potential matrix elements and 1e-9 Ha/Bohr in the forces (SCF energies to 1e-10 Ha).

## 3. Benchmarks

All numbers from `frey` (AMD Threadripper PRO 5975WX, 32 cores, 24 threads used unless stated;
single DDR4 channel populated, ~17 GB/s memory bandwidth, which matters for the cache mode),
Python 3.12, NumPy 2.5, SciPy 1.18 (OpenBLAS), Numba 0.68. System: `water_cluster_47.xyz`
(141 atoms) in def2-SVP (1175 Cartesian functions), density-fitted with def2-universal-jfit,
the default grid (1.28 million points after pruning), SCF to 1e-7 Ha. Scripts:
`benchmarks_tests/benchmark_xc_term.py` and the SCF input of the README quick start.

### 3.1 Complete SCF runs (24 threads)

| functional | algorithm | AO cache | XC term, all iterations | complete SCF | peak RSS |
|---|---|---|---|---|---|
| PBE | 2 | off | 74.3 s | 95.5 s | 6.5 GB |
| PBE | **3** | off | **5.0 s** (15.0x) | 26.8 s | 3.9 GB |
| PBE | 2 | on | 51.2 s | 75.6 s | 17.8 GB |
| PBE | **3** | on | **7.5 s** (6.8x) | 31.7 s | 9.2 GB |
| LDA (SVWN) | 2 | off | 42.9 s | 64.3 s | 5.7 GB |
| LDA (SVWN) | **3** | off | **4.2 s** (10.3x) | 25.1 s | 3.9 GB |
| LDA (SVWN) | 2 | on | 30.1 s | 51.9 s | 8.4 GB |
| LDA (SVWN) | **3** | on | **4.0 s** (7.5x) | 25.2 s | 5.2 GB |
| r2SCAN | 2 | off | 149.4 s | 167.4 s | 7.6 GB |
| r2SCAN | **3** | off | **7.9 s** (18.8x) | 25.2 s | 3.9 GB |
| r2SCAN | 2 | on | 132.2 s | 152.6 s | 18.6 GB |
| r2SCAN | **3** | on | **9.5 s** (14.0x) | 28.6 s | 9.2 GB |

(15 SCF iterations for LDA and PBE, 10 for r2SCAN; the energies of the two algorithms agree to
5e-9 Ha (LDA, PBE) and 2e-8 Ha (r2SCAN).) The whole SCF is 2.6x (LDA), 3.6x (PBE) and 6.6x
(r2SCAN) faster; the rest is the density-fitted Coulomb term, which algorithm 3 does not touch.

### 3.2 The XC term alone on the converged density

Best of three evaluations of energy and potential matrix (the `Exchange-Correlation Term` of one
SCF iteration), PBE:

| threads | algo 2 | algo 2, cache | algo 3 | algo 3, cache |
|---|---|---|---|---|
| 1 | 43.80 s | 12.77 s | 4.89 s | 2.70 s |
| 2 | 22.83 s | 6.87 s | 2.45 s | 1.38 s |
| 4 | 12.16 s | 4.20 s | 1.24 s | 0.76 s |
| 8 | 7.43 s | 3.28 s | 0.70 s | 0.64 s |
| 12 | | | 0.49 s | 0.53 s |
| 16 | 5.25 s | 3.18 s | 0.40 s | 0.61 s |
| 24 | 6.37 s | 5.15 s | 0.30 s | 0.58 s |

Speed-up of algorithm 3 over algorithm 2 without the cache: 9.0x on one thread, 21x on 24. The
parallel efficiency of algorithm 3 from 1 to 24 threads is 68% measured naively
(T1 / (24 T24)), 87% at 8 threads. Twenty-four *independent* single-thread copies of the kernel
run 1.19x slower each than one copy alone on this machine (clock speed and shared caches), which
caps the naive efficiency of any code at 84%; relative to that, algorithm 3 reaches 81%. The
remaining loss is the global lock of OpenBLAS around concurrent single-threaded `dgemm` calls
(see section 4) and the assembly of the matrix (5%). Algorithm 2 scales to 6.9x on 24 threads
(29% efficiency; 2.5x with the cache).

The cache mode of algorithm 3 streams 5.5 GB of basis function values per evaluation (algorithm
2: 15 GB): on this machine's 17 GB/s it is bandwidth-bound beyond 8 threads and slower than
recomputing, while on one thread it saves the 2.2 s of basis function evaluation. The cache of
algorithm 3 takes 2.7x less memory than that of algorithm 2 (5.5 GB against 15 GB for PBE; 1.4 GB
against 3.7 GB for an LDA).

| functional | threads | algo 2 | algo 3 | speed-up | algo 2, cache | algo 3, cache |
|---|---|---|---|---|---|---|
| LDA | 1 | 19.02 s | 4.39 s | 4.3x | 9.81 s | 2.25 s |
| LDA | 24 | 3.27 s | 0.31 s | 10.5x | 1.98 s | 0.20 s |
| PBE | 1 | 43.80 s | 4.89 s | 9.0x | 12.77 s | 2.70 s |
| PBE | 24 | 6.37 s | 0.30 s | 21x | 5.15 s | 0.58 s |
| r2SCAN | 1 | 78.02 s | 11.79 s | 6.6x | 53.96 s | 9.64 s |
| r2SCAN | 24 | 15.08 s | 0.72 s | 21x | 11.87 s | 1.06 s |

### 3.3 On a laptop (Apple M4, macOS, Python 3.9, NumPy 1.26, Numba 0.60)

`water_cluster_20.xyz` (60 atoms, 500 functions, 555k points), XC term alone, best of three:

| functional | threads | algo 2 | algo 3 | speed-up |
|---|---|---|---|---|
| PBE | 1 | 7.18 s | 1.52 s | 4.7x |
| PBE | 4 | 2.58 s | 0.38 s | 6.8x |
| PBE | 8 | 1.97 s | 0.27 s | 7.4x |
| PBE, cache | 8 | 1.16 s | 0.21 s | 5.6x |
| LDA | 8 | 1.10 s | 0.27 s | 4.1x |
| r2SCAN | 8 | 4.65 s | 0.98 s | 4.7x |

The 4-thread run of algorithm 3 scales perfectly (1.52 s / 4 = 0.38 s); the M4 has four performance
and six efficiency cores, and the static schedule gives every thread the same work, so beyond four
threads the slow cores set the pace (the cache mode helps here, where memory bandwidth is ample).
The smaller system also has shorter per-batch lists in algorithm 2 (its disadvantage grows with
the system size).

### 3.4 Nuclear gradient (XC contribution, fixed grid, 24 threads)

| functional | algo 2 | algo 3 | speed-up |
|---|---|---|---|
| LDA | 4.9 s | 0.32 s | 15x |
| PBE | 19.3 s | 0.70 s | 28x |
| r2SCAN | 27.9 s | 1.46 s | 19x |
| PBE, 1 thread | 94.2 s | 7.34 s | 12.8x |

With `grid_response=True` the XC gradient of PBE takes 8.4 s instead of 191 s; of these, 7.1 s are
the Becke weight derivatives (`Grids.becke_weight_gradient`), which both algorithms share. The
per-function gradients of the two algorithms agree to 1e-9, the per-atom XC forces to 2e-9 Ha/Bohr,
and the translation-invariance defect of the grid-response gradient is identical (2e-13 Ha/Bohr).


## 4. Notes

* **BLAS lock.** `np.dot` inside the Numba kernel calls SciPy's OpenBLAS with one thread. OpenBLAS
  serializes concurrent calls from different threads behind a global lock: 24 threads calling the
  leaf-sized `dgemm` back to back each see a 20x longer call. Inside the kernel the matrix products
  are a quarter of the work, so the effect is a few per cent at 24 threads, but it is what keeps the
  measured efficiency below the hardware cap. A lock-free Numba matrix kernel would remove it (at a
  lower per-core rate).

* `blocksize`, `list_nonzero_indices`, `count_nonzero_indices` are accepted by `eval_xc_3` for
  compatibility with `eval_xc_2` and ignored; the plan (`Integrals.eval_xc_3.build_xc_plan`) is built
  once per grid, in the SCF preprocessing, and memoized for direct calls on the same grid arrays.
* `xc_bf_screen=False` evaluates every shell on every leaf (for checking the screening only).
* Spin-polarized densities are not supported, as in algorithm 2.
* The benchmark harness is `benchmarks_tests/benchmark_xc_term.py`.
