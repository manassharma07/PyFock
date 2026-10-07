# RI-K (exact exchange): implementation, benchmark, acceleration, the GPU path, and the DF_algo=12 plan

RI exact exchange (`xc='HF'` and global hybrids such as B3LYP/PBE0) is built from the
shell-blocked three-center integrals of `DF_algo=11`, on the CPU (sections 1-3) and, since
2026-10-07, on the GPU (section 7: the rows stay on the device, the contraction is cuBLAS
throughout, and the early SCF iterations run it in single precision). This note walks through the
implementation, records the def2-SVP benchmarks of 2026-10-05 (CPU) and 2026-10-07 (GPU),
documents the CPU acceleration (2.2x on the exchange matrix, identical energies), analyses what a
`DF_algo=12` (multipole) RI-K can and cannot buy, and ranks the remaining options.

Code: `pyfock/Integrals/df_algo11_exchange.py` (CPU rows, J, K),
`pyfock/Integrals/df_algo11_exchange_cupy.py` (the same on the GPU), `pyfock/DFT_Helper_Coulomb.py`
(`density_fitting_prelims_for_DFT_development`, `Jmat_from_density_fitting`,
`Kmat_from_density_fitting`), `pyfock/DFT.py` (`exx_coef`, density factor, fallback 12 -> 11,
dynamic precision), `pyfock/Integrals/eval_xc_3_cupy.py` (GPU XC driver over a list of
functionals, so that the semilocal part of a hybrid is evaluated on the device).
Benchmark scripts: `benchmarks_tests/benchmark_RI_K.py` (SCF driver, one subprocess per
molecule), `benchmarks_tests/profile_RI_K.py` (library-level timing of the exchange build),
`benchmarks_tests/farfield_share_jkfit.py` (DF_algo=12 far-field shares with the JK basis),
`benchmarks_tests/sanity_b3lyp_rik.py`. Logs and tables: `benchmarks_tests/RI_K_def2-SVP/`.

## 1. The current RI-K path

**Setup (`DFT.scf`).** `exx_coef` is 1 for HF and the hybrid coefficient otherwise. With
`DF_algo=12` selected (the default) and `exx_coef > 0` the run switches to `DF_algo=11`,
because exchange needs the complete `(ij|P)` blocks that the far field of 12 replaces by
expansions. Hybrids need `XC_algo=2` on the CPU and `XC_algo=3` on the GPU (the defaults).

**Build (`density_fitting_prelims_for_DFT_development`).** The DF_algo=11 plan evaluates every
Schwarz-significant shell pair block `(ij|P)` with the Rys kernel (`max_memory_ints3c2e` is
ignored: all blocks must be cached). `df_algo11_exchange.build_exchange` then

1. enumerates the active function pairs `i >= j` (strict Schwarz cut-off applied per pair),
2. copies every block into one dense row per pair, contracting the auxiliary index with the
   Cartesian-to-spherical matrices in SAO mode (the fit space is the true spherical auxiliary
   basis, so the metric needs no regularisation),
3. factors the fit metric `(P|Q) = L L^T` and orthonormalises the rows in place,
   `B = R L^-T` (one `dtrsm` of `nrows x naux^2` flops), and
4. builds the bookkeeping for the per-iteration contractions.

The plan's raw blocks are released afterwards; `B` is the only three-center storage.

**Per iteration.** Coulomb: `gamma = d . B` and `J_r = B gamma` (two DGEMVs, no metric solve;
`gamma . gamma` is the DF Coulomb energy term). Exchange with the density factor
`D = F F^T` (`F` = occupied MO coefficients times `sqrt(occupation)`, CAO basis, provided by
the SCF after each diagonalisation; the first iteration factors the guess density by `eigh`):

    X[i, Q, o] = sum_k B[(ik), Q] F[k, o]        (half transform; the pair sparsity is built in)
    K_ij       = sum_{Q, o} X[i, Q, o] X[j, Q, o] (rank update, dsyrk)

over blocks of the auxiliary index whose work buffers fit `DFAlgo11Exchange.block_memory_bytes`
(512 MB). Flop count per iteration: `4 nrows naux nocc` (half transform, every stored row
feeds `X[i]` and `X[k]`) plus `nao^2 naux nocc` (rank update); memory traffic at least two
reads of `B`. For icosane/def2-SVP (nao 510, naux 2256, nocc 81, 45 308 rows) that is 33 + 47
GFlop per iteration.

## 2. Baseline benchmark

HF / def2-SVP / def2-universal-jkfit, SAO, Schwarz 1e-9 (no strict cut-off), `conv_crit=1e-7`,
SANO guess, 8 Numba threads and 8 BLAS threads, Intel i9-12900K (8 P + 8 E cores), Windows 11,
Python 3.10, numpy 1.26.4 (OpenBLAS 0.3.23, DYNAMIC_ARCH), Numba 0.61.2. Each molecule runs in
a fresh process after a warm-up run that fills the Numba cache (`benchmark_RI_K.py --warmup`).
Timings from the SCF profile block (`baseline.json`, `baseline_*.log`).

| molecule | atoms | nao (CAO) | naux (sph.) | pair rows | iters | 3c2e build [s] | orthonormalize rows [s] | K per iter [s] | K total [s] | J per iter [s] | SCF total [s] |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Benzene | 12 | 120 | 558 | 6945 | 8 | 0.09 | 0.12 | 0.019 | 0.15 | 0.01 | 1.06 |
| Serotonin | 25 | 255 | 1197 | 23172 | 10 | 0.36 | 0.24 | 0.138 | 1.38 | 0.02 | 3.16 |
| Caffeine | 24 | 260 | 1242 | 24332 | 12 | 0.36 | 0.26 | 0.152 | 1.83 | 0.02 | 3.79 |
| Decane | 32 | 260 | 1146 | 20144 | 8 | 0.28 | 0.19 | 0.119 | 0.95 | 0.01 | 2.46 |
| Icosane | 62 | 510 | 2256 | 45308 | 8 | 1.13 | 1.05 | 0.839 | 6.71 | 0.06 | 11.17 |

The exchange matrix is 60 % of the icosane SCF and 40-50 % of the others; the one-time build
(3c2e blocks + orthonormalisation + 2c2e) is the next item (icosane 2.6 s). J from the
orthonormal rows is negligible.

**Where the exchange time went (icosane, instrumented copy of the old `K_from_exchange`):**

| step | time [s] | rate |
|---|---|---|
| slab gathers (`_gather_shell`) | 0.09 | 1.64 GB moved, 19 GB/s |
| half-transform DGEMMs (one per shell and aux block, BLAS-threaded) | 0.42 | 33 GFlop, **78 GFlop/s** |
| rank update `X @ X.T` (numpy -> dsyrk, BLAS-threaded) | 0.29 | 47 GFlop (syrk), 162 GFlop/s |
| total per iteration | 0.81-0.85 | |

Reference rates on this box: a 3000x3000 DGEMM reaches 73 GFlop/s on 1 thread, 250 on 4, 274 on
8, 309 on 16; a 0.8 GB memory copy runs at 14 GB/s. The per-shell DGEMMs (M = nA x 1584,
K = 85-293 partner functions, N = 81) run at 55-58 GFlop/s on a single thread, i.e. the shapes
are fine; it is OpenBLAS's threading of ~500 medium calls per iteration that loses the factor 3.
Consistently, the old routine scaled only 2.2x from 1 to 8 BLAS threads (1.68 s -> 0.77 s).
The one-time triangular solve already runs at 332 GFlop/s (icosane 0.70 s).

## 3. What was changed (implemented)

`df_algo11_exchange.py` was restructured; the public API (`build_exchange`, `K_from_exchange`,
`gamma_from_exchange`, `J_from_exchange`, `B`, `row_mu`, `row_nu`, `aux_block_size`) is unchanged.

1. **Row order.** Rows are sorted by `(i, j)`, so the *own* rows of every function `i`
   (partners `j <= i`) are one contiguous range of `B` that BLAS reads in place through a
   transposed strided view - no gather. The rows in which `i` is the smaller index (its
   *partner* rows `(k, i)`, `k > i`) are either copied once into a partner-ordered array `P`
   (`DFAlgo11Exchange.store_partner_slabs`: `None` = automatic when the copy is below 35 % of the
   available memory, `True`/`False` to force) or gathered per iteration into a thread-local
   buffer. The copy doubles the three-center memory and buys about 13 % of the exchange time
   (icosane: 0.32 s stored vs 0.36 s gathered per iteration).
2. **Threading.** The half transform is one DGEMM per function and aux block (plus one for the
   partner rows), distributed heaviest-first over `numba.get_num_threads()` Python threads that
   call *single-threaded* BLAS (the OpenBLAS limit is set through `threadpoolctl` for the
   duration of the call). The memory-bound gathers and the compute-bound DGEMMs of different
   functions overlap.
3. **Rank update.** `K += X X^T` is split along the contracted `(Q, o)` axis: every thread forms a
   partial `K` by `dsyrk` (numpy's `A @ A.T` dispatch) and the partials are summed. This runs at
   ~320 GFlop/s, at the DGEMM ceiling of the machine.

Validation: `tests/test_df_algo11_exchange.py` (K and J against dense references on the same
screened integrals, CAO/SAO, both Schwarz modes, f functions, blocking independence, the MO
factor path), `tests/test_rihf_algo11_scf.py` (RI-HF SCF, DF_algo 11 against the dense
DF_algo 3 path) and the new `tests/test_rik_hybrid_scf.py` (B3LYP and PBE0 SCF with RI-K
against the dense path in CAO and SAO mode; exchange contraction independent of the thread
count, of stored vs gathered partner rows and of the aux block size) pass; the benchmark
energies agree with the baseline to 1e-11 Ha with identical iteration counts. The CH4
B3LYP/PBE0 cases of the short regression suite (`tests/short/ch4_*_df_algo_11`) cover the
hybrid RI-K path with PySCF grids.

B3LYP/def2-SVP decane (native B3LYP, default grids, same settings otherwise; logs
`b3lyp_decane_before_*` and `b3lyp_decane_after_*`): exchange matrix 1.09 -> 0.53 s over 8
iterations (2.1x), SCF 10.1 -> 9.0 s, because the semilocal XC term (5.9 s) dominates a hybrid
run on the CPU; energies agree to 1e-12 Ha.

**Before/after (same settings as section 2; `v1_threads_slabs.json`, `comparison.md`):**

| molecule | K per iter: before | after | speedup | K total: before | after | SCF total: before | after | speedup |
|---|---|---|---|---|---|---|---|---|
| Benzene | 0.019 | 0.011 | 1.7x | 0.15 | 0.09 | 1.06 | 0.89 | 1.19x |
| Serotonin | 0.138 | 0.062 | 2.2x | 1.38 | 0.62 | 3.16 | 2.43 | 1.30x |
| Caffeine | 0.152 | 0.068 | 2.2x | 1.83 | 0.82 | 3.79 | 2.84 | 1.33x |
| Decane | 0.119 | 0.055 | 2.2x | 0.95 | 0.44 | 2.46 | 1.95 | 1.26x |
| Icosane | 0.839 | 0.379 | 2.2x | 6.71 | 3.03 | 11.17 | 8.07 | 1.38x |

The orthonormalisation time is unchanged within noise (icosane 1.05 -> 1.11 s; it now includes
the partner copy). Benzene is overhead-dominated (10 ms per exchange build).

**Remaining floor for the exact algorithm.** Icosane per iteration: the half transform reads
the rows twice, 1.63 GB at 14 GB/s = 0.12 s, and the rank update needs 47 GFlop at 320 GFlop/s
= 0.15 s; the implementation sits at 0.32-0.38 s against this ~0.27 s floor. Beyond that the
exchange time can only come down algorithmically (section 5) or on the GPU.

## 4. Plan: RI-K with DF_algo=12

### 4.1 What DF_algo=12 offers

`DF_algo=12` evaluates only the near-field primitive-pair / auxiliary-shell combinations with the
Rys kernel and represents the far field by exact finite moments of the product distributions,
translated to box-level branch centres (`Mtil`, `(lmax+1)^2 = 169` doubles per row and branch),
coupled to exact point multipoles of the auxiliary shells on each atom
(`docs/df_algo12_multipoles.md`). Per SCF iteration the far field costs `O(169)` per row for
`gamma` and `J` instead of one term per far-field auxiliary function, because the density
contraction collapses the row index before the branch-to-atom coupling.

Far-field shares with the JK-fitting basis (`farfield_share_jkfit.py`, def2-SVP /
def2-universal-jkfit, SAO, default multipole options):

| molecule | atoms | far field: share of significant (ij\|P) | share of the Rys work | branches | (pair, branch) entries |
|---|---|---|---|---|---|
| Benzene | 12 | 0.0 % | 1.1 % | 45 | 87 |
| Serotonin | 25 | 4.6 % | 17.2 % | 149 | 3823 |
| Caffeine | 24 | 4.2 % | 16.0 % | 147 | 3244 |
| Decane | 32 | 17.2 % | 27.1 % | 168 | 6654 |
| Icosane | 62 | 51.3 % | 58.9 % | 298 | 18977 |
| Tetracontane (C40H82) | 122 | 74.0 % | 78.2 % | 597 | 39807 |

### 4.2 Why the far field cannot replace the per-iteration exchange contraction

Exchange keeps two indices open that the Coulomb term contracts away. Written with the raw
blocks, `K = sum_PQ Xt_P M^-1_PQ Xt_Q^T` with `Xt_P[i, o] = sum_k (ik|P) F[k, o]`:

* The far-field half transform factorises per branch `b`: `Z_b[i, LM, o] = sum_k Mtil[(ik), b, LM]
  F[k, o]` costs only `rows-in-entries x 169 x nocc` (icosane ~25 GFlop, DGEMM-friendly). But
  expanding `Z_b` to the auxiliary functions, `Xt_P^ff[i, o] += sum_LM Z_b[i, LM, o] t_bP[LM]`,
  costs `nao_b x 169 x nocc` per (branch, far-field auxiliary function) or, atom-wise through the
  interaction tensors, `nao_b x (l_A+1)^2 x 169 x nocc` per (branch, atom): for icosane
  ~300 branches x 40 atoms x 100 functions x 25 x 169 x 81 x 2 = 0.8 TFlop per iteration, an order
  of magnitude above the dense contraction. The row index `i` (~100 functions per branch) and
  the orbital index `o` survive the coupling that costs `(l_A+1)^2 x 169` in the Coulomb case.
* The metric transform `L^-T` mixes all auxiliary functions, so after it the near/far structure
  is gone; applying it after the half transform instead (`Xt L^-T`) costs `naux^2 nao nocc / 2`
  per iteration (icosane 210 GFlop), 2.5x the whole dense contraction. Local-metric variants
  (PARI/ARI-K, pair-atomic fit domains) change the approximation and are a separate method.

Conclusion: with the dense orthonormal rows in memory, the per-iteration exchange is already
the cheapest exact RI-K; DF_algo=12 can only change the **build** of those rows.

### 4.3 What DF_algo=12 can do: build the rows from the near-field blocks plus multipoles

New module `pyfock/Integrals/df_algo12_exchange.py` with
`build_exchange(plan12, basis, auxbasis, metric, sao, release_plan_values)` returning the same
`DFAlgo11Exchange` object (so `K_from_exchange`, `gamma/J_from_exchange` and the SCF code are
reused unchanged):

1. **Rows and order.** Reuse `_count_active_rows`/`_active_pairs` from `df_algo11_exchange`
   (the DF_algo=12 plan has the same `pair_I/pair_J/work_iter/sqrt_ints4c2e_diag/strict_schwarz`
   fields); rows sorted by `(i, j)` as now.
2. **Near-field columns.** A variant of `_fill_rows` that walks the stored block of pair `p`
   through its near-field column map (`plan.cols[plan.col_off[p]:plan.col_off[p+1]]`, shell
   contiguous) instead of the Schwarz-only shell test, with the same SAO contraction into the
   spherical fit space. Near-field blocks of a (pair, shell) for which only *some* groups are
   near field hold only those groups' Rys contribution; the rest is added in step 3.
3. **Far-field columns.** For every (pair `p`, branch `b`) entry with branch-centred row
   moments `Mtil_e` (`nrows_p x 169`; requires `low_memory=False`) and every atom `A` with
   `plan.any_ff[b, A]`: `Y = Mtil_e @ T_bA^T` (`T_bA` = `multipole_helpers.interaction_tensor`
   of the irregular harmonics of `B_b - A`, shape `(l_A+1)^2 x 169`), then for every auxiliary
   shell `K` on `A` with `plan.ff[b, K]`: `R[rows_p, cols_K] += Y[:, :nK] @ W_K^T` with
   `W_K = C2S_K @ plan.aux_mom[k0:k0+nC, :nK]` (`C2S_K` = identity in CAO mode; `aux_mom` is
   already projected in SAO mode, and `C pinv(C) C = C`, so the contraction to the spherical
   columns is exact). This is exactly `approx_block` of `df_algo12_helpers`, batched: the
   irregular harmonics of all (branch, atom) pairs are cached once (icosane 43 MB,
   tetracontane 170 MB), the loop is parallel over *pairs* (entries of one pair in different
   branches add to the same rows - a branch-parallel loop would race) and the two small
   products per (entry, atom) are plain DGEMMs.
   Cost: icosane has ~19 000 entries of ~4 rows each, ~40 far-field atoms per branch and
   `(l_A+1)^2 <= 25`: ~73 000 x 40 x 25 x 169 x 2 = 25 GFlop plus ~5 GFlop of shell
   contractions, i.e. 0.1-0.3 s against the 0.64 s of far-field Rys work (59 % of 1.09 s) it
   replaces. The gain grows with the far-field share (tetracontane 78 %) and with the cost of
   the primitives (core-heavy elements, larger auxiliary bases); it is nil for the three
   compact molecules.
4. **Orthonormalisation and bookkeeping** exactly as in `df_algo11_exchange.build_exchange`
   (Cholesky of the spherical/Cartesian metric, `dtrsm`, own/partner structures).
5. **SCF integration.** In `density_fitting_prelims_for_DFT_development`, build the DF_algo=12
   plan with `max_memory_gb=None` when `rihf` and call the new builder; store it as
   `ints3c2e.exchange`. In `Jmat_from_density_fitting` the `DF_algo==12` branch takes the
   `exchange is not None` path of the 11 branch (`gamma`/`J` from the rows: the multipole
   `gamma`/`J` passes are not needed once `B` exists, and both must come from the same
   approximate tensor so that the energy stays a functional of the density). Remove the 12 -> 11
   fallback in `DFT.__init__` for the CPU; keep the GPU error. `DFT_Grad` is unaffected (RI-K
   gradients are not implemented; `DFT_Grad` already uses the DF_algo=12 plan for the Coulomb
   part).
6. **Validation.** Unit test: rows from `df_algo12_exchange.build_exchange` against rows from
   `df_algo11_exchange.build_exchange` on the same geometry (differences bounded by the
   multipole precision, 1e-10 relative to a unit-charge interaction; compare before the metric
   transform with `plan12.approx_block`/`exact_block` per block), K and J against the dense
   references of `test_df_algo11_exchange.py`, and an SCF energy comparison 11 vs 12 on the
   water chain of `test_df_algo12.py` (the Coulomb energies of 11 and 12 agree to ~1e-9 Ha there).
7. **Expected outcome.** Icosane: build 3c2e 1.1 -> ~0.6 s, SCF 8.1 -> ~7.6 s; C40: build ~4 s
   less; the compact molecules unchanged; per-iteration exchange unchanged. Worth doing for
   consistency (hybrids no longer silently change the DF algorithm) and for large systems, but
   it ranks below the items of section 5 in gain per effort.

## 5. Further acceleration, ranked by expected gain per effort

1. **GPU RI-K (cuBLAS).** Implemented on 2026-10-07, see section 7. (Before: `DFT.scf` exited
   for exact exchange with `use_gpu=True` and for any hybrid on the GPU, the GPU XC driver
   evaluated exactly one exchange and one correlation functional, and the prelims raised for
   RI-HF with the CUDA plan.) Measured fp64 rates on the RTX 5070 used here: 472 GFlop/s
   (`cupy`, 4096^3 GEMM) against 23.7 TFlop/s in fp32, the usual 1/50 consumer ratio, while the
   CPU reaches 274-320 GFlop/s on 8 threads - which is why the GPU path contracts the exchange
   in single precision until the energy change is small (section 7.3). On data-centre GPUs with
   full-rate fp64 (A100 19.5 TFlop/s, H100 ~60 TFlop/s) the same work is 0.01-0.02 s per
   iteration in double precision and the exchange becomes negligible.
2. **LK-type sparsity (Aquilante, Pedersen, Lindh, JCP 126, 194106 (2007)).** Replace the
   canonical density factor by a pivoted Cholesky factor of `D` (localised "Cholesky orbitals",
   `dpstrf`, one `nao^3` call) and truncate its tails at a threshold; then `X[i, Q, o]` is
   block-sparse in `(i, o)` and both the half transform (`F[partners, o]` restricted to the
   orbitals that touch the partners of `i`) and the rank update (`K[S_o, S_o] += X_o X_o^T` on
   the support `S_o` of each orbital) shrink with the support fraction squared. Estimate for
   icosane (supports of ~6-8 of 20 atoms): 2-4x on top of section 3; decane ~1.5x;
   benzene/serotonin/caffeine ~1x (compact, conjugated). Controlled approximation (threshold
   ~1e-10 on the factor); the SCF energy at convergence changes by the truncation error only.
3. **occ-RI-K (Manzer, Horn, Mardirossian, Head-Gordon, JCP 143, 024113 (2015)).** Build only
   `K C_occ` (`nao x nocc`) and use the occupied projection of `K` in the Fock matrix: the rank
   update becomes `X (X^T C)` with `nao nocc^2 naux` flops (icosane 15 GFlop instead of 47; the
   half transform stays), exact at convergence, energy from `Tr[D K]` which needs only `K C`.
   Changes the SCF iterations (the virtual-virtual block of `K` is dropped), so DIIS and
   convergence behaviour must be re-validated; moderate effort inside `DFT.scf`.
4. **Pipelining.** Overlap the memory-bound half transform of aux block `k+1` with the
   compute-bound rank update of block `k` (split the thread pool): 10-20 % on the per-iteration
   exchange, no change in results.
5. **Build side.** Write the rows straight from the Rys kernel instead of through
   `plan.values` (saves one 1 GB write + read, ~0.2 s for icosane) and fuse the SAO contraction;
   the `dtrsm` is at peak. The far-field route of section 4 belongs here.
6. **Thread count.** With all 16 physical cores the dynamic scheduling of the new code gains a
   further 20-30 % on this box (the E-cores add bandwidth and FMA throughput); `ncores=8` was
   kept for the comparison.
7. **Not worth pursuing:** per-iteration multipole exchange (section 4.2); incremental `K` from
   density differences (`D_new - D_old` has rank `2 nocc`, no saving with a factorised build);
   fp32 for any part of the exchange.

## 6. Reproduction

```bash
cd benchmarks_tests
PYTHONUTF8=1 CUPY_ACCELERATORS= python benchmark_RI_K.py --ncores 8 --tag baseline --warmup
PYTHONUTF8=1 CUPY_ACCELERATORS= python benchmark_RI_K.py --gpu --tag gpu_dynamic --warmup
PYTHONUTF8=1 CUPY_ACCELERATORS= python benchmark_RI_K.py --gpu --no-dynamic-precision --tag gpu_fp64
PYTHONUTF8=1 CUPY_ACCELERATORS= python profile_RI_K.py --xyz Icosane_C20H42.xyz --ncores 8
PYTHONUTF8=1 CUPY_ACCELERATORS= python farfield_share_jkfit.py
PYTHONUTF8=1 CUPY_ACCELERATORS= python -m pytest tests/test_df_algo11_exchange.py tests/test_rihf_algo11_scf.py tests/test_rik_gpu.py -q
```

`benchmark_RI_K.py` runs every molecule in its own subprocess, parses the SCF profile block
and writes `<tag>.json`/`<tag>.md` next to the logs; `comparison.md` in the same directory
holds the before/after table of section 3 and `gpu_comparison.md` the CPU/GPU table of
section 7.

## 7. RI-K on the GPU (implemented 2026-10-07)

`use_gpu=True` with `xc='HF'` or a global hybrid now runs the whole SCF on the device:
`pyfock/Integrals/df_algo11_exchange_cupy.py` is the device counterpart of the CPU module and
is validated against it by `tests/test_rik_gpu.py` (rows, K, J and gamma in CAO and SAO mode,
in double and single precision and for several auxiliary block sizes; HF, B3LYP and PBE0 SCF
energies on the GPU against the CPU RI-K path, 2e-7 Ha; the dynamic-precision run against the
double-precision run, 1e-8 Ha).

### 7.1 Build

The DF_algo=11 CUDA plan (`df_algo11_helpers_cupy.build_plan_cupy`, `max_memory_ints3c2e`
ignored as on the CPU) leaves the screened `(ij|P)` blocks on the device. One warp per active
function pair (strict Schwarz cut-off applied per pair, rows sorted by `(i, j)`) expands its
block row into the dense row matrix `R`, in SAO mode contracting the pseudo-Cartesian columns
of every auxiliary shell with its Cartesian-to-spherical matrix, so the fit space is the true
spherical auxiliary basis and the spherical metric needs no regularisation (the prelims keep
the spherical 2c2e matrix for this and use only the pseudo-Cartesian diagonal for the Schwarz
bounds). The plan's blocks are released as soon as `R` is filled. The metric is factored with
cuSOLVER (`cupy.linalg.cholesky`) and `B = R L^-T` is one in-place cuBLAS `dtrsm` on `R`'s
memory (the C-ordered `R` is `R^T` in column-major terms, the C-ordered lower `L` is the upper
`L^T`; solve `(L^T)^T Y = R^T`), so no second copy of the rows is ever made. `B` is the only
three-center storage; the device needs `nrows x naux x 8` bytes for it plus the plan's blocks
during the fill (icosane/def2-SVP/jkfit: 0.82 GB for `B`). A clear `MemoryError` is raised
when `B` does not fit the free device memory.

### 7.2 Per iteration

Everything is cuBLAS; nothing moves between host and device.

* Coulomb: `gamma = d . B` and `J_r = B gamma` are two GEMVs over the rows (`d_r = D_ij` on the
  diagonal, `D_ij + D_ji` otherwise), `J` is scattered symmetrically with fancy indexing, and
  `gamma . gamma` is the DF Coulomb energy term - exactly the CPU algorithm.
* Exchange: the density factor `F` (occupied MO coefficients times `sqrt(occupation)`, formed
  on the device after each diagonalisation) gives `X[i, o, Q] = sum_k B[(ik), Q] F[k, o]` and
  `K = sum_{o,Q} X X^T`. The rows that feed function `i` (own rows `(i, k <= i)` and partner
  rows `(k > i, i)`) are listed once at build time; the functions are sorted by that count
  (heaviest first) and cut into *bins* in which every count is at least 80 % of the bin's
  maximum. Per auxiliary block of `nb` functions and per bin, one coalesced kernel gathers
  the rows into a zero-padded slab `U[i, k, Q]` (the same kernel converts to fp32 when asked;
  `B` is read exactly twice per iteration), one strided-batched GEMM
  `X[bin] = F_pad[bin] @ U[bin]` does the half transform with the pair sparsity built in and
  at most 25 % padding, and after all bins one `syrk` adds `X X^T` of the block to `K`. `nb`
  is chosen so that `U` and `X` fit `DFAlgo11ExchangeGPU.block_memory_bytes` (default: a
  quarter of the free device memory, at most 2 GB; icosane needs one block). Flop count as on
  the CPU (`4 nrows naux nocc` + `nao^2 naux nocc`, 80 GFlop per icosane iteration) plus the
  padding.

### 7.3 Dynamic precision

Under `DFT.dynamic_precision` (the default for GPU runs, already used for the XC term) the
exchange contraction runs in single precision - fp32 slabs, `sgemmStridedBatched`, an `sgemm`
rank update (`ssyrk` from 1024 functions on), `K` cast back to float64 - until the relative energy change between two iterations falls
below 5e-7, and in double precision from there on; `B` is always stored in double precision.
The converged energy is therefore a double-precision one (`test_rik_gpu.py`: within 1e-8 Ha of
the fp64-only run on H2O; see the table below for the benchmark molecules), while the early
iterations cost a small fraction of an fp64 iteration on consumer GPUs with their 1/32-1/64
fp64 rate. `dynamic_precision=False` runs everything in double precision from the first
iteration.

### 7.4 Benchmark

Same settings as section 2 (HF / def2-SVP / def2-universal-jkfit, SAO orbital and fit space,
Schwarz 1e-9, `conv_crit=1e-7`, SANO guess, 8 CPU threads for the host-side work), GeForce RTX
5070 (12 GB; 472 GFlop/s in fp64, 23.7 TFlop/s in fp32 on a 4096^3 GEMM), one fresh process per
molecule with warm Numba/CuPy caches (`gpu_dynamic.json`, `gpu_fp64.json`, logs
`gpu_*_<molecule>.log`; `gpu_comparison.md` holds this table). The CPU column is section 3
(`v1_threads_slabs`, 8 threads). "dyn" = dynamic precision (fp32 exchange until the switch, the
default), "fp64" = `dynamic_precision=False`. The converged energies of all three runs agree to
3e-9 Ha on every molecule (icosane: -781.279228271 CPU, -781.279228269 fp64 and dyn), and the
dynamic run needs exactly as many iterations as the fp64 and CPU runs.

| molecule | iters CPU / fp64 / dyn (fp32 iters) | K per iter [s] CPU / fp64 / dyn | K total [s] CPU / fp64 / dyn | 3c2e + rows [s] CPU / GPU | SCF total [s] CPU / fp64 / dyn |
|---|---|---|---|---|---|
| Benzene | 8 / 8 / 8 (6) | 0.011 / 0.010 / 0.013 | 0.09 / 0.08 / 0.10 | 0.16 / 0.24 | 0.89 / 1.11 / 1.10 |
| Serotonin | 10 / 10 / 10 (6) | 0.062 / 0.053 / 0.030 | 0.62 / 0.53 / 0.30 | 0.61 / 0.47 | 2.43 / 2.12 / 1.82 |
| Caffeine | 12 / 12 / 12 (7) | 0.068 / 0.085 / 0.043 | 0.82 / 1.02 / 0.52 | 0.67 / 0.49 | 2.84 / 2.68 / 2.15 |
| Decane | 8 / 8 / 8 (6) | 0.055 / 0.069 / 0.031 | 0.44 / 0.55 / 0.25 | 0.50 / 0.44 | 1.95 / 2.00 / 1.63 |
| Icosane | 8 / 8 / 8 (6) | 0.379 / 0.298 / 0.094 | 3.03 / 2.38 / 0.75 | 2.20 / 1.47 | 8.07 / 5.58 / 3.83 |

Reading the table:

* **Double precision only.** On this card the fp64 contraction of icosane takes 0.30 s per
  iteration against 0.38 s on 8 CPU threads (1.3x); the small molecules are launch- and
  overhead-bound (benzene: 10 ms either way). The whole icosane SCF is 5.6 s against 8.1 s,
  because the three-center build (1.09 -> 0.75 s), the orthonormalisation (1.11 -> 0.79 s) and
  the diagonalisation also run on the device. A component profile of the icosane fp64
  contraction (`cp.cuda.Device().synchronize()` between the steps; the integral routines
  leave their own stream current, so a stream-level synchronisation measures nothing): slab
  gathers 0.017 s, batched GEMMs 0.100 s (36 GFlop padded, 365 GFlop/s = 78 % of the card's
  471 GFlop/s), `dsyrk` 0.152 s (47 GFlop, 312 GFlop/s), 0.26 s in all - the implementation is
  at this card's fp64 floor; the remaining lever is the algorithm (section 5) or a GPU with a
  real fp64 rate. In fp32 the same build takes 0.016 s (gather 0.005 s, GEMMs 0.005 s at 8
  TFlop/s, GEMM rank update; cuBLAS's `ssyrk` would take 0.024-0.043 s alone, so below 1024
  functions the fp32 rank update is an `sgemm`).
* **Dynamic precision.** The fp32 iterations of icosane take 0.06-0.08 s in total (exchange
  0.016 s, 18x cheaper than in fp64; the rest is diagonalisation, DIIS and the Python-level
  overhead of an iteration) against 0.32 s for an fp64 GPU iteration and 0.50 s on the CPU.
  The switch to double precision (relative energy change below 5e-7, reached after 6 of the 8
  iterations here) moves the energy by a few mHa, after which the fp64 iterations converge as
  the fp64-only run does; the exchange time drops from 2.38 s to 0.75 s (4x over the CPU)
  and the SCF from 8.1 s to 3.8 s. (An earlier variant of the fp32 rank update with `ssyrk`
  rounded differently and took two extra iterations on icosane: the iteration at which the
  switch fires is sensitive to fp32 rounding, the converged energy is not.) The threshold is
  shared with the XC term.
* **Hybrid functionals.** B3LYP/def2-SVP decane with the default grids (CPU reference:
  `b3lyp_decane_after`, section 3; GPU: `gpu_b3lyp_decane`, dynamic precision, 5 fp32
  iterations): SCF 8.97 s -> 3.29 s with 8 iterations on both sides and the same energy to
  1e-9 Ha (-394.052967870). The semilocal XC term, which dominates a CPU hybrid run, drops
  from 5.86 s to 1.33 s on the device, the exchange from 0.53 s to 0.31 s; a GPU iteration
  takes 0.12 s in the fp32 phase and 0.26-0.31 s in fp64, against 0.86 s on the CPU. This is
  the practical point of the GPU path on consumer cards: hybrids run where the XC term already
  is, instead of being refused.
* **What to expect elsewhere.** The host-side parts of an iteration are now comparable to the
  exchange itself on this card (icosane: 0.3 s of diagonalisation and 0.4 s of miscellaneous
  per 8 iterations). On a data-centre GPU with full-rate fp64 (A100 19.5 TFlop/s, H100 ~60
  TFlop/s) the fp64 contraction of an icosane iteration is 0.01-0.02 s, dynamic precision
  becomes irrelevant for the exchange, and the build (one `dtrsm` of `naux^2 nrows` flops,
  230 GFlop for icosane) is a few tens of milliseconds.

