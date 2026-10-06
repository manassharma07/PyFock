# RI-K (exact exchange): implementation, benchmark, acceleration, and the DF_algo=12 plan

RI exact exchange (`xc='HF'` and global hybrids such as B3LYP/PBE0) is built on the CPU from
the shell-blocked three-center integrals of `DF_algo=11`. This note walks through the
implementation, records the def2-SVP benchmark of 2026-10-05, documents the acceleration that
was implemented on that day (2.2x on the exchange matrix, identical energies), analyses what a
`DF_algo=12` (multipole) RI-K can and cannot buy, and ranks the remaining options.

Code: `pyfock/Integrals/df_algo11_exchange.py` (rows, J, K), `pyfock/DFT_Helper_Coulomb.py`
(`density_fitting_prelims_for_DFT_development`, `Jmat_from_density_fitting`,
`Kmat_from_density_fitting`), `pyfock/DFT.py` (`exx_coef`, density factor, fallback 12 -> 11).
Benchmark scripts: `benchmarks_tests/benchmark_RI_K.py` (SCF driver, one subprocess per
molecule), `benchmarks_tests/profile_RI_K.py` (library-level timing of the exchange build),
`benchmarks_tests/farfield_share_jkfit.py` (DF_algo=12 far-field shares with the JK basis),
`benchmarks_tests/sanity_b3lyp_rik.py`. Logs and tables: `benchmarks_tests/RI_K_def2-SVP/`.

## 1. The current RI-K path

**Setup (`DFT.scf`).** `exx_coef` is 1 for HF and the hybrid coefficient otherwise. With
`DF_algo=12` selected (the default) and `exx_coef > 0` the run switches to `DF_algo=11`,
because exchange needs the complete `(ij|P)` blocks that the far field of 12 replaces by
expansions. RI-K is CPU only and needs `XC_algo=2` for hybrids.

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

1. **GPU RI-K (cuBLAS).** The dense rows fit the 12 GB card for every molecule here
   (0.8-1.6 GB for icosane). The half transform and the rank update are 80 GFlop of fp64
   DGEMM/SYRK per iteration, i.e. ~0.03-0.06 s on the RTX 5070 against 0.38 s on 8 CPU threads;
   the `dtrsm` of the build runs on the GPU too. The DF_algo=11 CUDA plan
   (`df_algo11_helpers_cupy`) already provides the blocks; needed are a CuPy `build_exchange`
   (row fill with the SAO contraction, `cupyx.scipy.linalg.solve_triangular`), the per-function
   DGEMMs as batched `cublasDgemmStridedBatched`-style calls or a single gather + GEMM per aux
   block, and a `K_from_exchange_cupy`. `DFT` currently exits for RI-K on the GPU.
   Expected: exchange 6-10x faster than the new CPU code; largest practical win.
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
PYTHONUTF8=1 CUPY_ACCELERATORS= python profile_RI_K.py --xyz Icosane_C20H42.xyz --ncores 8
PYTHONUTF8=1 CUPY_ACCELERATORS= python farfield_share_jkfit.py
PYTHONUTF8=1 CUPY_ACCELERATORS= python -m pytest tests/test_df_algo11_exchange.py tests/test_rihf_algo11_scf.py -q
```

`benchmark_RI_K.py` runs every molecule in its own subprocess, parses the SCF profile block
and writes `<tag>.json`/`<tag>.md` next to the logs; `comparison.md` in the same directory
holds the before/after table of section 3.
