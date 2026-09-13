# SANO initial guess: superposition of atomic natural-orbital densities

The SCF starts by default from a superposition of spherically averaged free-atom densities
(`DFT(..., dmat_guess_method='sano')`) instead of the core-Hamiltonian guess
(`dmat_guess_method='core'`). The implementation lives in
[`pyfock/Guess.py`](../pyfock/Guess.py); the SCF calls `DFT.guess_dmat`, which dispatches on
`dmat_guess_method`. `'sano'` became the default after the benchmark below.

## How it works

No atomic SCF is run. The contracted functions of the ANO-RCC-MB minimal basis of Roos and
co-workers (shipped in `pyfock/BasisSets`, H to Cm) are the natural orbitals of the free
atoms, so the spherically averaged, spin-averaged density of every atom is a *diagonal*
matrix in that basis:

- 2 electrons in every closed shell,
- the open-shell electrons spread evenly over the 2l+1 orbitals of the shell
  (for example 2/3 per 2p orbital of carbon, 8/5 per 3d orbital of iron),
- 0 in the core shells replaced by an effective core potential.

The electrons per angular momentum come from `Data.ATOMIC_CONFIGURATION_NRSRHF`, the
ground configurations of the spherically averaged spin-restricted HF atom (the same table
PySCF uses; it differs from the experimental aufbau configuration for a few transition metals,
lanthanides and actinides, e.g. Fe is 3d8 4s0). The core shells dropped for an ECP with `n`
core electrons are tabulated in `Data.ECP_CORE_SHELLS`.

Every atomic natural orbital `a` is projected onto the calculation basis with the cross
overlap between the two basis sets and the guess density is the superposition of the
occupied projected orbitals,

```text
c_a = S^-1 S_cross a          D = sum_a  occ_a f_a^2  c_a c_a^T
```

Without the factors `f_a` this is exactly the projected density `P D_min P^T` of PySCF's
default `minao` guess (`Guess.sano_dmat(..., renormalize=False)` reproduces it to 1e-12 in the
natural-occupation spectrum). By default the 2l+1 components of every atomic shell share one
factor that restores the norm of the shell in the calculation basis, so the guess carries
exactly the electron count of the molecule and stays rotationally invariant. The correction
is 0.1 % for all-electron atoms, but 3-7 % for ECP atoms, whose nodeless valence functions
cannot represent the inner part of the all-electron valence natural orbitals; there it saves
one to two SCF iterations.

The whole guess costs one cross-overlap matrix, one Cholesky solve with the overlap matrix
and one matrix product, i.e. a fraction of a second even for a few thousand basis functions.
The density is returned in the Cartesian representation used inside the SCF, so it works in
CAO and SAO mode, on the CPU and on the GPU, with density fitting or exact integrals, and for
pure, hybrid and Hartree-Fock calculations. Ghost atoms carry no density. Elements beyond Cm
or ECPs with an untabulated core size fall back to the core guess with a warning.

The SCF output lists the references to cite when the guess is used (the SAD idea of Almlöf,
Faegri and Korsell and of Van Lenthe et al., the ANO concept of Widmark, Malmqvist and Roos,
the ANO-RCC papers for the elements present, the Basis Set Exchange and PySCF, whose
projection scheme is followed). `Guess.sano_references()` and `Guess.sano_citation_text()`
return the same list.

## Usage

```python
from pyfock import Basis, DFT, Guess, Mol

mol = Mol(coordfile='molecule.xyz')
basis = Basis(mol, {'all': Basis.load(mol=mol, basis_name='def2-SVP')})

dft = DFT(mol, basis, xc='PBE', dmat_guess_method='sano')   # the default; 'core' for the core-Hamiltonian guess
energy, dmat = dft.scf()

# the guess density itself, e.g. for inspection or as a starting point elsewhere
dmat_guess, info = Guess.sano_dmat(mol, basis)
print(info['nelectrons_guess'], info['nelectrons_projected'], info['species'])
```

`Guess.project_dmat(dmat, basis_from, basis_to)` projects any density matrix between two
basis sets with the same scheme (for example a converged density from a smaller basis).

## Grid pruning

PyFock prunes the XC integration grid with the starting density (points with
`|rho * w| < 1e-11` are removed). The core-Hamiltonian density is far too compact, so this
pruning removed points that carry real density. Pruning with the SANO density is accurate:

| System (PBE/def2-SVP, level-3 grid) | E(unpruned grid) | E(core-pruned) - E(unpruned) | E(SANO-pruned) - E(unpruned) |
|---|---|---|---|
| H2O | -76.273790726 Ha | +5.3e-05 Ha | +1.9e-08 Ha |
| Decane | -393.483760304 Ha | +3.1e-04 Ha | +1.5e-07 Ha |

This is why the SANO and core energies in the benchmark below differ by up to 1e-4 Ha for DFT:
the SANO-pruned grid is the more accurate one. With `xc='HF'` (no grid) the energies agree to
1e-10 Ha. A reused density (ASE geometry optimizations, `grid_pruning_use_core_guess=True`)
is pruned with the density of the configured guess method, so all steps use the same grid as
a fresh calculation.

## Convergence benchmark

`benchmarks_tests/benchmark_sano_guess.py` converges every case twice with identical settings
and only the initial guess changed. Settings: `DF_algo=11`, native level-3 grids, `conv_crit`
1e-7 Ha, `max_itr` 100, auxiliary basis def2-universal-jfit (jkfit for B3LYP and HF), 8 Numba
threads on an Apple M4 (macOS, NumPy 1.26, Numba 0.60). Each run prunes the XC grid with
its own starting density, which is what a user gets; the wall time is that of the whole SCF
including integrals and grid setup. The energy difference is the grid-pruning effect discussed
above (the SANO energy is the accurate one; it is largest for the heavy-atom molecules, 1.4 mHa
for AgCl); for Hartree-Fock, which has no grid, the energies agree to 1e-9 Ha.

| System | Basis | XC | N_bf | Iterations core | Iterations SANO | SCF wall core (s) | SCF wall SANO (s) | E(SANO) - E(core) (Ha) | ANOs represented by the basis |
|---|---|---|---|---|---|---|---|---|---|
| H2O | def2-SVP | PBE | 25 | 10 | 7 | 1.2 | 1.0 | -5.3e-05 | 99.86 % |
| H2O | def2-TZVP | PBE | 48 | 8 | 6 | 1.2 | 1.2 | -8.3e-05 | 99.98 % |
| H2O | def2-QZVP | PBE | 142 | 8 | 7 | 4.6 | 4.4 | -1.2e-04 | 99.99 % |
| H2O | def2-SVP | B3LYP | 25 | 10 | 6 | 1.9 | 1.5 | -4.2e-05 | 99.86 % |
| H2O | def2-SVP | HF | 25 | 10 | 7 | 0.7 | 0.6 | +1.3e-10 | 99.86 % |
| Ethane | def2-SVP | PBE | 60 | 11 | 6 | 3.2 | 2.3 | -5.3e-05 | 99.96 % |
| Decane | def2-SVP | PBE | 260 | 18 | 8 | 28.6 | 17.0 | -3.1e-04 | 99.96 % |
| Decane | def2-TZVP | PBE | 492 | 19 | 8 | 89.0 | 51.5 | -5.6e-04 | 100.00 % |
| Decane | def2-SVP | HF | 260 | 14 | 7 | 9.4 | 5.6 | -2.3e-09 | 99.96 % |
| Caffeine | def2-SVP | PBE | 260 | 20 | 12 | 39.8 | 31.1 | -4.4e-04 | 99.95 % |
| Caffeine | def2-SVP | B3LYP | 260 | 17 | 11 | 41.8 | 31.6 | -3.5e-04 | 99.95 % |
| Serotonin | def2-SVP | PBE | 255 | 20 | 11 | 39.5 | 26.1 | -3.6e-04 | 99.96 % |
| Benzene-fulvene dimer | def2-SVP | PBE | 240 | 20 | 9 | 32.6 | 21.1 | -4.1e-04 | 99.96 % |
| Adenine-thymine | def2-SVP | PBE | 340 | 23 | 12 | 90.2 | 31.4 | -5.2e-04 | 99.95 % |
| Cholesterol | def2-SVP | PBE | 650 | 27 | 10 | 394.1 | 187.5 | -7.5e-04 | 99.96 % |
| Zn dimer | def2-SVP | PBE | 72 | 9 | 6 | 3.8 | 2.5 | -5.2e-04 | 99.95 % |
| Cd dimer (ECP) | def2-SVP | PBE | 72 | 9 | 7 | 3.8 | 3.4 | -7.3e-09 | 92.94 % |
| AgCl (ECP) | def2-SVP | PBE | 55 | 11 | 9 | 1.6 | 1.5 | -1.4e-03 | 96.08 % |
| AuCl (ECP) | def2-SVP | PBE | 56 | 12 | 8 | 1.7 | 1.5 | -3.9e-04 | 96.23 % |
| HgCl2 (ECP) | def2-SVP | PBE | 75 | 10 | 8 | 4.7 | 4.3 | -7.3e-04 | 97.30 % |
| I2 (ECP) | def2-SVP | PBE | 56 | 8 | 7 | 1.5 | 1.2 | -4.4e-08 | 93.28 % |
| BiH3 (ECP) | def2-SVP | PBE | 43 | 9 | 7 | 1.8 | 1.6 | -2.4e-08 | 94.23 % |

Total SCF iterations over the converged cases: core 303, SANO 179 (41 % fewer).

The guess never increased the iteration count, and it saved between 12 % (I2) and 63 %
(cholesterol) of the iterations, 41 % over the whole set. The saving grows with molecular size:
the core-Hamiltonian guess deteriorates for larger systems (18-27 iterations for the
C10-C27 molecules) while SANO stays at 8-12. The gain is smallest for the ECP molecules, where
the nodeless ECP basis represents only 93-97 % of the all-electron valence natural orbitals
before the renormalization. Based on this table `'sano'` was made the default guess.


## Tests

`tests/test_sano_guess.py` checks the shell occupations, the exact electron count, the
positive semidefiniteness, the exactness for a closed-shell atom in its own minimal basis,
the rotational invariance, the agreement with PySCF's `minao` density, the ECP handling,
ghost atoms, the `project_dmat` utility, the dispatch in `DFT`, the SCF energy and iteration
count relative to the core guess, and the citation block in the SCF output.
