# Starting densities along a geometry path: projection, transfer and extrapolation

During a geometry optimization every SCF starts close to the previous one. PyFock's ASE calculator
turns the converged densities of earlier steps into the starting density of the next SCF. The
implementation lives in [`pyfock/guess_projection.py`](../pyfock/guess_projection.py).

**Credits.** The projection and extrapolation code was written by Prof. Vincenzo Barone while he was
interfacing PyFock with a geometry optimizer, and contributed to PyFock. It was added to PyFock and
adapted by Manas Sharma with Prof. Barone's permission. His code is kept exactly as contributed (with
only the two changes needed for Python 3.9) in
[`pyfock/guess_projection_original.py`](../pyfock/guess_projection_original.py) and can be selected
with `implementation='original'` for comparisons.

## Usage

```python
from ase.optimize import BFGS
from pyfock import PyFockCalculator

atoms.calc = PyFockCalculator(functional='PBE', basis='def2-SVP',
                              density_guess='extrapolate')        # default: 'previous'
BFGS(atoms).run(fmax=0.04)
print(atoms.calc.pyfock_results['density_guess_info'])            # method, points, weights, time
```

| `density_guess` | starting density of the next SCF |
|---|---|
| `'previous'` (default) | the previous converged density matrix, unchanged |
| `'transfer'` | the previous density carried with the atoms: its occupied natural orbitals keep their AO coefficients and are re-orthonormalized in the new overlap metric |
| `'project'` | the previous density projected into the new basis (the natural-orbital projection of the contributed code) |
| `'extrapolate'` | the densities of the last five geometries carried to the new one, combined with weights fitted to the geometries and made N-representable |

`density_guess_options` is passed to `pyfock.guess_projection.DensityHistory`: `max_points`,
`weights` (`'geometry'` or `'uniform'`), `representation` (`'transfer'` or `'project'`), `damping`,
`rcond` and `implementation` (`'pyfock'` or `'original'`). `reuse_density=False` starts every step
from the configured guess (SANO) instead. The checkpoints are the `structure.xyz` and
`converged_dmat.npy` files of the step directories, so the same works with
`run_in_process=False` (the default), where every step runs in its own process.

Prof. Barone's algorithm as contributed (the previous densities projected into the new basis and
extrapolated as equally spaced points, with his N-representability and electron-count safeguards):

```python
PyFockCalculator(..., density_guess='extrapolate', density_guess_options={'implementation': 'original'})
```

Outside ASE, e.g. in a hand-written optimizer loop, the same logic is available directly:

```python
from pyfock import DFT, guess_projection

history = guess_projection.DensityHistory(method='extrapolate')
for geometry in path:
    mol, basis, auxbasis = ...                                    # new geometry, same basis set
    dft = DFT(mol, basis, auxbasis, xc='PBE')
    dft.dmat = history.guess(basis, mol.nelectrons)               # None at the first geometry
    energy, dmat = dft.scf()
    if dft.converged:
        history.append(basis, dmat)
```

## How it works

Everything happens in the AO overlap metric of the new geometry, `S = L L^T` (a Cholesky factor, or
canonical orthogonalization with an eigenvalue cutoff for nearly linearly dependent bases). The
natural orbitals of a density `D` are the eigenvectors of `L^T D L`; their eigenvalues are the
natural occupations.

**Projection** (contributed). The occupied natural orbitals `C` of the old density are mapped into
the new basis with `S_new^-1 S_new,old` and re-orthonormalized, `C (C^T S C)^-1/2`. The result is
N-representable, keeps the occupations and electron count, and represents the old density *at its
old position in space* as well as the new basis can. This is the right tool when the basis set
changes at a fixed geometry.

**Transfer.** The same re-orthonormalization applied to the unchanged coefficients `C`: the density
moves with the atom-centred functions, as the electrons largely do. A rigid translation of the
molecule is reproduced exactly (a projection would leave the density behind), and compared with the
unchanged matrix the density has the right electron count and is idempotent in the new metric; to
first order the difference is exactly the part `-C S^x_oo C^T` of the density derivative that is
fixed by the overlap derivative alone.

**Extrapolation** (contributed idea, extended). The last `k` densities are all brought into the new
basis (transfer by default, projection with `representation='project'`) and combined,
`D = sum_j c_j D_j` with `sum_j c_j = 1`, then made N-representable (natural occupations clipped to
[0, 2]) and normalized to the electron count, as in the contributed code. The contributed weights
assume equally spaced points: linear extrapolation `(-1, 2)` from two points, a least-squares
quadratic fit from three or more (`weights='uniform'`, 3 points by default). Optimizer steps are not
equally spaced, though: they change length and direction and sometimes go back. The default weights
are therefore fitted to the geometries (`weights='geometry'`, 5 points by default): `c` expresses the
new geometry as the closest point of the affine span of the earlier ones, i.e. a linear model of the
density along the directions the path has already explored. It reduces to linear extrapolation for
equal collinear steps, keeps the latest density when the new step points in a new direction and
returns to an earlier density when the optimizer steps back. Directions with a singular value below
`rcond` (1e-2) times the largest are left out, which bounds the weights for nearly collinear histories.

The cost is negligible: for caffeine/def2-SVP (260 basis functions) a guess from earlier densities
takes 0.03-0.06 s (0.19 s with the original implementation), against about 1 s per SCF iteration.
The earlier geometries' bases are the current basis moved to the old atomic positions, so no
basis-set file is read again.

## Benchmark

ASE optimizations from rattled structures (`atoms.rattle(stdev=0.05, seed=7)`), PBE/def2-SVP with
def2-universal-jfit, `conv_crit=1e-7`, `fmax=0.04` eV/Å, `run_in_process=True`, 4 cores (Apple M4).
All strategies follow the same optimization path — the energies of corresponding steps agree to about
1e-4 eV — so the table compares the SCF iterations over the force calls the runs have in common.
(Close to convergence on a flat surface, the step at which fmax first drops below the threshold can
move by a few steps either way; that is noise from the SCF threshold, not a change of path.)

With PyFock's SCF as it is (energy-change convergence test):

| SCF iterations, same path | fresh SANO guess every step | original contributed code | `'previous'` (default) | `'extrapolate'`, 3 points | `'extrapolate'` (5 points) |
|---|---|---|---|---|---|
| ethanol, BFGS (16 force calls) | 144 | 115 | 93 | 81 (-13 %) | 78 (-16 %) |
| ethanol, LBFGS (16) | – | 115 | 93 | 81 (-13 %) | – |
| ethanol, r2SCAN, BFGS (15) | 135 | 109 | 84 | 78 (-7 %) | – |
| caffeine, BFGS (26) | – | 264 | 215 | 184 (-14 %) | 176 (-18 %) |
| serotonin, BFGS (32) | – | 290 | 233 | 206 (-12 %) | 196 (-16 %) |

With the additional orbital-gradient convergence test described in the caveat below, applied to every
strategy (this removes the few SCFs that stop early, which otherwise flatter the extrapolation):

| SCF iterations, same path | `'previous'` (default) | `'extrapolate'`, 3 points | `'extrapolate'` (5 points) |
|---|---|---|---|
| ethanol, BFGS (16 force calls) | 94 | 82 (-13 %) | 79 (-16 %) |
| serotonin, BFGS (32) | 233 | 211 (-9 %) | 199 (-15 %) |

- Projecting the previous density into the displaced basis keeps it fixed in space while the atoms
  move; along the recorded caffeine path it is a 5x worse starting density than the unchanged matrix
  (error `||L^T (D_guess - D) L||_F`, geometric mean over 36 steps: 1.7e-1 against 3.3e-2). Moving the
  density with the atoms (transfer) gives 2.6e-2, and the extrapolation from 3 or 5 transferred
  densities 8.5e-3 and 7.2e-3.
- The contributed algorithm needs about 20 % fewer SCF iterations than starting every step from a
  fresh guess, which was the comparison in the original campaign, but 23-30 % more than the
  calculator's default `'previous'`, which reuses the density matrix without moving it (in the
  atom-centred basis the density follows the atoms).
- The optimization paths, the number of geometry steps and the final energies are the same for all
  strategies within the noise of the SCF threshold: the starting density changes how fast each SCF
  converges, not where it converges to.
- Wall time falls less than the SCF iteration count (3-10 % here), because the gradient, the
  integrals and the grid of every step cost the same.

**Caveat: early SCF termination.** The SCF declares convergence when the energy changes by less than
`conv_crit` between two iterations. From a very good starting density one DIIS step can stagnate, so
two successive energies agree while the density is still slightly off: this happened at 3 of 35
serotonin steps with `'extrapolate'` (forces off by 0.9-2.4e-4 Ha/bohr instead of the usual 1e-5; it
also occurs occasionally with `'previous'`). Tightening `conv_crit` to 1e-8 removed two of the three
cases, not the third. An additional orbital-gradient test (`max|F D S - S D F| < 1e-4`) removed all of
them without adding iterations to normally converged SCFs (second table). PyFock's SCF does not apply
that test, which is why `'extrapolate'` is not the default.
