# XC integration grids

PyFock offers two schemes for the grids of the numerical exchange-correlation integration
(`Grids(mol, level, scheme=...)`, `DFT.grids_scheme`):

- **`'treutler'` (default)**: Treutler-Ahlrichs radial grids, Lebedev angular grids, angular pruning by
  radial regions and Becke partitioning with Treutler's atomic-size adjustment, built by PyFock for grid
  levels 0 (coarsest) to 9 (finest).
- **`'numgrid'`**: the grids of the [numgrid](https://github.com/dftlibs/numgrid) library, the
  previous PyFock scheme, with two parameter presets (`'compact'` and `'dense'`, the old PyFock
  defaults).

## The default scheme

The grid of every atom is the product of a radial and an angular quadrature, and the atomic grids
are joined with Becke's fuzzy-cell partitioning:

1. **Radial grid**: the Treutler-Ahlrichs M4 quadrature (O. Treutler, R. Ahlrichs, J. Chem. Phys.
   102, 346 (1995)): Gauss-Chebyshev nodes of the second kind mapped with
   r = xi/ln2 (1+x)^0.6 ln(2/(1-x)), with the element-specific scaling xi of Table I of the paper
   (H-Kr) and the customary values for the heavier elements. The number of radial points per level
   and period of the element is tabulated in `RADIAL_POINTS_TABLE` (level 3: 50 for H-He, 75 for
   Li-Ne, 80 for Na-Ar, ...).
2. **Angular grids**: Lebedev-Laikov grids; the tables are taken from numgrid at run time
   (`numgrid.angular_grid`). The largest grid per level and period is tabulated in `Mapping`
   (level 3: 302 points for H-Ne, 434 beyond).
3. **Pruning** (`region_angular_sizes`): 50 and 86 points in the two innermost
   regions, the full grid in the valence region and one Lebedev step smaller beyond, with region
   boundaries at fixed multiples of the Bragg-Slater radius (0.25/0.5/1.0/4.5 for H-He,
   0.1667/0.5/0.9/3.5 for Li-Ne, 0.1/0.4/0.8/2.5 beyond). `pruning=None` disables it.
4. **Partitioning**: Becke's fuzzy cells (A. D. Becke, J. Chem. Phys. 88, 2547 (1988)) with three
   smoothing iterations and the atomic-size adjustment a_ij = (chi - 1/chi)/4, |a| <= 1/2, applied
   to the square roots of the Bragg-Slater radii as proposed by Treutler and Ahlrichs
   (`size_adjustment='treutler'`; `'becke'` uses the radii themselves, `None` no adjustment). The
   partition weights of all points are computed by a parallel Numba kernel over all atom pairs.

The Bragg-Slater radii are Slater's table (J. Chem. Phys. 41, 3199 (1964)) with the customary
0.35 A for H and 1.40 A for He. The points are finally grouped into 1.2 Bohr boxes so that the
batches of the XC evaluation are spatially compact, and `atom_idx` records the atom of every
point. `points_per_element={'C': (75, 302)}` overrides the numbers of radial and angular points.

### Validation and cost

With these ingredients the grid coincides with the grid PySCF builds at the same level: for water,
SnCl4, decane and C60 at levels 1-5 the number of points, the set of points (to 1e-14 Bohr) and the
weights (to 1e-13 relative) are the same, and SCF energies agree to the convergence threshold (see
the benchmark below). `tests/test_grids.py` checks this for water and SnCl4, including the
unpruned, Becke-adjusted and unadjusted variants.

Build times (4 cores of an Apple M4, Numba kernel already compiled):

| molecule | level | points | PySCF | PyFock `Grids` |
|---|---|---|---|---|
| decane | 3 | 356,956 | 0.14 s | 0.11 s |
| decane | 5 | 947,660 | 0.37 s | 0.25 s |
| C60 | 3 | 847,080 | 1.09 s | 0.83 s |
| C60 | 5 | 2,578,680 | 8.6 s | 5.5 s |

The cost grows as N_points x N_atoms^2 because all atom pairs enter the partition weight of every
point.

## The numgrid scheme and how the two compare

### What changed in the numgrid scheme

- `Grids(mol, level=3, scheme='numgrid')` takes `level` 0-9 like the default scheme. Its
  preset `'compact'` (default of the scheme) maps the level to numgrid's parameters (`RADIAL_PRECISION` and `MIN_ANGULAR`
  tables in `pyfock/Grids.py`) and uses the per-element table of the largest angular grid of the
  default scheme (`Mapping`, e.g. 302 points for H-Ne at level 3, 434 for heavier elements). The previous grids
  are `preset='dense'` (levels 3-8, identical points and weights to before). `radial_precision`
  and `angular_points=(min, max)` override the preset; the def2-QZVP basis is built internally
  when no `basis` is given.
- The numgrid output is assembled with NumPy instead of Python lists, the grid points are grouped
  into 1.2 Bohr boxes (pure NumPy) instead of the Python `list.sort`, and the
  density pruning of `DFT.scf` evaluates only the basis functions that reach each block of points
  (the same screening as the XC evaluation). `Grids` also records `atom_idx`, the atom every
  point belongs to, and keeps it consistent through pruning.
- `DFT` exposes the choices as `grids_scheme`, `grids_options` (native scheme), `grids_preset`,
  `grids_radial_precision` and `grids_angular_points` (numgrid scheme); `use_pyscf_grids=True` and
  passing a PySCF `grids` object still work.

Decane, def2-SVP, PBE, 4 cores, grid pipeline of `DFT.scf` before the SCF (generation, density
pruning, sorting):

| pipeline | points (raw / after pruning) | time |
|---|---|---|
| before (radial precision 1e-13, 110-590 angular, Python lists, dense pruning, Python sort) | 1,151,592 / 836,860 | 5.9 s (0.3 basis + 1.1 numgrid and lists + 3.2 pruning + 1.3 sort) |
| `preset='dense'` now (same grid, NumPy assembly, box grouping, sparse pruning) | 1,151,592 / 836,858 | 3.3 s (1.7 generation incl. 0.3 s def2-QZVP basis + 1.7 pruning) |
| `preset='compact'` now | 359,176 / 295,358 | 1.3 s (0.8 generation incl. the basis + 0.6 pruning) |
| `scheme='treutler'` (default) | 356,956 / 287,846 | 1.1 s (0.4 generation + 0.7 pruning) |
| PySCF `grids.build()` level 3 (no pruning) | 356,956 | 0.32 s |

The remaining generation time is dominated by numgrid's own Becke partitioning of every atomic grid
against all atoms (Rust, parallel) and the def2-QZVP basis construction; the pruning time by the
evaluation of the significant basis functions at every point.

Full SCF (DF-PBE/def2-SVP, conv 1e-7, MINAO start, 9 iterations, 4 cores of an Apple M4):

| calculation | grid points (after pruning) | E (Ha) | E - E(PySCF) | grid generation + pruning | total time |
|---|---|---|---|---|---|
| PySCF level 3 | 298,032 | -393.4655072 | | | 19.5 s |
| PyFock, PySCF's grid passed in | 298,032 | -393.4655067 | 5.3e-7 | | 7.5 s |
| PyFock, `treutler` scheme (default) | 287,846 | -393.4655067 | 5.3e-7 | 0.36 s + 0.74 s | 7.9 s |
| PyFock, numgrid `compact` preset | 295,358 | -393.4653886 | 1.2e-4 | 1.1 s + 0.7 s | 8.9 s |
| PyFock, numgrid `dense` preset | 836,858 | -393.4654376 | 7.0e-5 | 2.4 s + 2.2 s | 22.2 s |

With the default scheme the energy is the same as with PySCF's own grid object (the two agree to 5e-9
Ha; the remaining 5e-7 Ha against PySCF is the 1e-7 Ha SCF convergence threshold of both codes). The
1.2e-4 Ha of the numgrid `compact` preset is the sum of the two grid errors (PySCF -6.7e-5, numgrid
+5.1e-5 Ha in Exc at the fixed density); the dense grid's 7.0e-5 Ha is PySCF's grid error alone.

### How the numgrid grids compare with PySCF's (fixed density)

The grid error is measured at a fixed density: a PySCF DF-RKS/PBE/def2-SVP calculation is converged
once and the electron count and the XC energy of that density are evaluated on every grid; the
reference is PySCF's level-8 grid (2.5M points for decane; the level-9 grid agrees with it to 2e-9
Ha). `benchmarks_tests/benchmark_grids.py` reproduces the tables. PyFock's density pruning
(|rho w| >= 1e-11) changes the numbers below by less than 1e-8 Ha.

#### Decane (C10H22)

| grid | points | dN(electrons) | dExc (Ha) |
|---|---|---|---|
| PySCF level 2 | 230,264 | 3.4e-4 | -6.4e-5 |
| PySCF level 3 | 356,956 | 3.7e-4 | -6.7e-5 |
| PySCF level 4 | 631,656 | -1.6e-5 | 2.3e-6 |
| PySCF level 5 | 947,660 | 6.4e-7 | -2.0e-7 |
| PySCF level 6 | 1,399,312 | -6.8e-7 | 1.1e-7 |
| numgrid `pyscf` level 2 (rp 1e-7, 194/302) | 246,096 | 1.0e-4 | -1.3e-5 |
| numgrid `pyscf` level 3 (rp 1e-8, 302) | 359,176 | 1.2e-5 | 5.1e-5 |
| numgrid `pyscf` level 4 (rp 1e-9, 434/590) | 649,056 | 1.7e-5 | -2.6e-5 |
| numgrid `pyscf` level 5 (rp 1e-10, 590/770) | 939,024 | -4.3e-5 | -1.8e-5 |
| numgrid `pyscf` level 6 (rp 1e-11, 770/974) | 1,289,424 | -3.3e-6 | -5.3e-6 |
| numgrid `dense` level 3 (rp 1e-13, 86-590 for H, 110-590 for C) | 1,151,592 | -1.2e-5 | 1.9e-6 |
| numgrid rp 1e-13, 590/770 | 1,217,592 | -2.6e-6 | 1.8e-6 |

(rp: radial precision; angular grids given as H/C when they differ.) At level 3 the default preset
reproduces PySCF's grid size within 1% with an error of the same magnitude (5e-5 vs 7e-5 Ha). From
level 4 on, a numgrid grid of PySCF's size is 10-100 times less accurate than PySCF's, and no
numgrid setting up to a radial precision of 1e-13 with 770/974 angular points reaches the 2e-7 Ha
of PySCF's level 5: the numgrid error levels off around 2e-6 Ha for this molecule.

#### Four molecules, all levels

Same protocol (PBE/def2-SVP density, reference PySCF level 8); the `compact` preset at level L
against PySCF's level L. Angular grids are the largest Lebedev grids per period (H-He / Li-Ne /
heavier).

| molecule | level | PySCF points | PySCF dExc (Ha) | numgrid `pyscf` points | ratio | numgrid dExc (Ha) | angular grids |
|---|---|---|---|---|---|---|---|
| decane C10H22 | 1 | 106,184 | 4.0e-04 | 133,248 | 1.25 | -6.5e-04 | 110/194 |
| decane C10H22 | 2 | 230,264 | -6.4e-05 | 246,096 | 1.07 | -1.3e-05 | 194/302 |
| decane C10H22 | 3 | 356,956 | -6.7e-05 | 359,176 | 1.01 | 5.1e-05 | 302 |
| decane C10H22 | 4 | 631,656 | 2.3e-06 | 649,056 | 1.03 | -2.6e-05 | 434/590 |
| decane C10H22 | 5 | 947,660 | -2.0e-07 | 939,024 | 0.99 | -1.8e-05 | 590/770 |
| decane C10H22 | 6 | 1,399,312 | 1.1e-07 | 1,289,424 | 0.92 | -5.3e-06 | 770/974 |
| caffeine C8H10N4O2 | 1 | 97,000 | 4.5e-05 | 116,848 | 1.20 | -7.6e-05 | 110/194 |
| caffeine C8H10N4O2 | 2 | 209,768 | -1.1e-05 | 209,240 | 1.00 | -1.0e-04 | 194/302 |
| caffeine C8H10N4O2 | 3 | 294,796 | -1.1e-05 | 272,752 | 0.93 | -2.1e-05 | 302 |
| caffeine C8H10N4O2 | 4 | 553,968 | 5.3e-07 | 530,520 | 0.96 | -1.9e-05 | 434/590 |
| caffeine C8H10N4O2 | 5 | 833,636 | 2.7e-07 | 751,584 | 0.90 | 1.3e-05 | 590/770 |
| caffeine C8H10N4O2 | 6 | 1,221,400 | -8.8e-08 | 1,023,904 | 0.84 | 3.4e-06 | 770/974 |
| water | 1 | 10,124 | 4.7e-06 | 12,582 | 1.24 | -4.9e-06 | 110/194 |
| water | 2 | 21,952 | 4.9e-08 | 23,206 | 1.06 | -1.0e-06 | 194/302 |
| water | 3 | 33,698 | -3.0e-08 | 33,488 | 0.99 | 2.3e-06 | 302 |
| water | 4 | 59,676 | 9.4e-09 | 60,912 | 1.02 | -8.3e-07 | 434/590 |
| water | 5 | 90,058 | -7.1e-09 | 87,862 | 0.98 | -4.5e-07 | 590/770 |
| water | 6 | 132,380 | 3.6e-09 | 120,596 | 0.91 | -1.7e-07 | 770/974 |
| SnCl4 (ECP on Sn) | 1 | 33,290 | -2.9e-04 | 29,444 | 0.88 | -2.2e-03 | 194 |
| SnCl4 (ECP on Sn) | 2 | 61,028 | 7.0e-06 | 50,000 | 0.82 | -1.1e-03 | 302 |
| SnCl4 (ECP on Sn) | 3 | 97,478 | -1.7e-05 | 76,134 | 0.78 | 5.4e-04 | 434 |
| SnCl4 (ECP on Sn) | 4 | 146,888 | 1.0e-05 | 121,104 | 0.82 | 4.3e-04 | 590 |
| SnCl4 (ECP on Sn) | 5 | 219,194 | 1.0e-05 | 165,286 | 0.75 | 2.8e-04 | 770 |
| SnCl4 (ECP on Sn) | 6 | 313,820 | 2.2e-06 | 220,324 | 0.70 | -1.6e-05 | 974 |

For the organic molecules the preset tracks PySCF's grid size within about 10% (levels 2-6) and
matches its accuracy at levels 1-3, while at levels 4-6 PySCF's grids are 10-100 times more
accurate for the same size. Water shows how favourable PySCF's construction is for small
molecules: its level-3 grid is accurate to 3e-8 Ha, the numgrid grid of the same size to 2e-6 Ha.
For SnCl4 (third- and fifth-row elements) numgrid is 30 times less accurate than PySCF at every
level (3e-4 to 5e-4 Ha at levels 3-5), and no radial precision cures it: with PySCF's partitioning
and the same unpruned 434-point angular grid, Treutler radial grids (95 points for Sn, 80 for Cl)
give -1.8e-5 Ha, LMG radial grids of precision 1e-8 (75/87 points) 1.6e-4 Ha and of precision
1e-13 (140/158 points) 2.2e-5 Ha. The largest def2-QZVP exponent of chlorine is 1.5e6, so 90 of
the 158 LMG points of the 1e-13 grid lie below 0.01 Bohr and only 5-7 per octave in the valence
region (Treutler-80: 8-11). For molecules with elements beyond the second row use PySCF's grids
(`use_pyscf_grids=True`) when a grid error below 1e-4 Ha matters.


#### Where numgrid's accuracy floor lies

Extreme settings at the same fixed density (reference PySCF level 9). Disabling numgrid's core
pruning (smallest angular grid = largest) changes nothing but the point count, so the floor is
not the Bragg/5 pruning.

| grid | decane points | decane dExc (Ha) | decane dN | SnCl4 points | SnCl4 dExc (Ha) | SnCl4 dN |
|---|---|---|---|---|---|---|
| PySCF level 5 | 947,660 | -2.0e-7 | 7.5e-7 | 219,194 | 9.7e-6 | -6.8e-6 |
| PySCF level 6 | 1,399,312 | 1.0e-7 | -5.7e-7 | 313,820 | 1.7e-6 | -7.4e-7 |
| PySCF level 7 | 1,968,684 | -2.6e-8 | 3.0e-7 | 432,434 | -5.6e-7 | 7.7e-7 |
| PySCF level 8 | 2,482,520 | -2.3e-9 | 1.1e-7 | 476,756 | -5.9e-7 | 7.7e-7 |
| numgrid rp 1e-13, 86-770 | 1,418,760 | 1.9e-6 | -2.8e-6 | 215,072 | 3.1e-5 | -5.9e-5 |
| numgrid rp 1e-13, 86-1202 | 2,127,696 | 1.6e-6 | 5.2e-6 | 312,872 | 1.0e-6 | -2.7e-5 |
| numgrid rp 1e-15, 86-770 | 1,626,248 | 1.7e-6 | -1.1e-5 | 247,726 | -1.6e-5 | 2.3e-5 |
| numgrid rp 1e-15, 86-1202 | 2,410,568 | 1.4e-6 | -4.3e-6 | 356,398 | -3.1e-5 | 3.2e-5 |
| numgrid rp 1e-17, 86-770 | 1,859,816 | 1.1e-6 | -1.0e-5 | 281,466 | 6.4e-5 | -3.0e-5 |
| numgrid rp 1e-17, 86-1202 | 2,747,720 | 5.7e-7 | -2.9e-6 | 403,314 | 2.7e-5 | -1.5e-5 |
| numgrid rp 1e-20, 86-770 | 2,231,040 | 8.4e-8 | -6.8e-6 | 337,236 | 1.7e-6 | -9.9e-6 |
| numgrid rp 1e-20, 86-1202 | 3,260,832 | -3.8e-7 | 5.9e-7 | 478,296 | -1.1e-5 | 7.4e-6 |

For decane the numgrid error creeps down from 2e-6 to a few 1e-7 Ha between 1.4M and 3.3M points
(PySCF: 2.6e-8 Ha with 2.0M), with electron-count errors that stay around 1e-6 to 1e-5 where
PySCF's are 1e-7. Agreement with PySCF's fine grids to about 1e-6 Ha is therefore reachable with
the `dense` preset, agreement to 1e-7 Ha only with radial precisions of 1e-20 and more than 2M
points. For SnCl4 the error fluctuates between 1e-6 and 6e-5 Ha with no trend up to 1.7M points
(PySCF: 6e-7 Ha from 430k points on); numgrid grids should not be relied on below 1e-4 Ha for
elements beyond the second row.

### Why the numgrid grids are less efficient

The three ingredients of the default scheme are all fixed inside `numgrid.atom_grid` and cannot be
selected through its API:

1. **Radial grid.** The default scheme uses Treutler-Ahlrichs M4 grids (J. Chem. Phys. 102, 346 (1995)),
   which concentrate their points in the valence region. numgrid's LMG grid is an exponential grid
   whose spacing is set by the radial precision and whose innermost point is set by the largest
   basis exponent, so it spends the same number of points on every decade of the radius. Points
   per radial range for carbon:

   | radial grid | < 0.01 | 0.01-0.1 | 0.1-0.5 | 0.5-1 | 1-2 | 2-4 | 4-8 | > 8 Bohr |
   |---|---|---|---|---|---|---|---|---|
   | Treutler, 75 points (default scheme, level 3) | 8 | 9 | 12 | 7 | 9 | 11 | 11 | 8 |
   | LMG, precision 1e-8, 78 points | 31 | 15 | 10 | 4 | 4 | 5 | 4 | 5 |
   | LMG, precision 1e-13, 145 points | 76 | 20 | 15 | 6 | 6 | 7 | 6 | 9 |
   | numgrid Krack-Koster, 75 points | 13 | 8 | 10 | 6 | 8 | 9 | 10 | 11 |

   For an isolated atom this does not matter (Ne/def2-SVP: LMG 78 points 2.9e-7 Ha, Treutler 75
   points 3e-9 Ha, both fine), but in a molecule every atomic grid has to integrate its Becke cell,
   whose boundary lies in the valence region: N2 with the same 302-point angular grid gives 1.3e-5
   Ha for LMG at precision 1e-8 (about 78 radial points) and 1.1e-6 Ha for Treutler with 75 points, and
   swapping numgrid's partitioning for PySCF's does not change the LMG number. This is the
   accuracy floor seen above. numgrid also provides Krack-Koster radial grids
   (`numgrid.radial_grid_kk`), which distribute their points like Treutler's, but only the LMG
   grid is available inside `points_per_element`, i.e. with the partitioning.
2. **Angular pruning.** The region-wise scheme of the default grids integrates the inner shells with 50 and 86
   points out to 0.9-1.0 Bragg radii and the far region with the next smaller Lebedev grid;
   it removes 36% of PySCF's level-3 points for decane at no cost in accuracy (unpruned: 558,704
   points, -6.7e-5 Ha). numgrid only reduces the angular grid inside one fifth of the Bragg radius
   (0.26 Bohr for carbon), so its grids carry the full angular grid on almost every shell.
3. **Partitioning.** numgrid applies Becke's original atomic-size adjustment (ratio of the Bragg
   radii), the default scheme Treutler's (ratio of their square roots). On the same numgrid points
   for decane at precision 1e-13, the Treutler adjustment lowers the error from 1.8e-6 to 3.9e-7 Ha;
   at precision 1e-8 both are 5e-5.

## Consequences for comparisons with PySCF

- With the default scheme the SCF energies of PyFock and PySCF differ only by the SCF convergence
  and by PyFock's density pruning of the generated grid (|rho w| < 1e-11, about 1e-9 Ha), i.e. by
  1e-8 Ha or less. Passing PySCF's `grids` object or `use_pyscf_grids=True` is no longer needed for
  benchmarks, though both still work.
- The numgrid scheme (`grids_scheme='numgrid'`) remains available. Its `'compact'` preset gives
  grids of the default scheme's size with comparable accuracy at level 3 for light elements, its
  `'dense'` preset the previous PyFock grids; with either, energies differ from the default
  scheme's by the sum of the two grid errors (about 1e-4 Ha at level 3 for decane) and the error
  floor of the LMG radial grids described above applies. For elements beyond the second row the
  numgrid grids should not be relied on below 1e-4 Ha.
