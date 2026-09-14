# PyFock

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]
[![Documentation][docs-shield]][documentation-url]
[![PyPI version](https://img.shields.io/pypi/v/pyfock.svg?style=for-the-badge)](https://pypi.org/project/pyfock/)

<br />
<div align="center">
  <h3 align="center">PyFock</h3>

  <p align="center">
    A pure Python Gaussian basis DFT code with GPU acceleration for efficient quantum chemistry calculations
    <br />
    <a href="https://pyfock-docs.bragitoff.com"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://pyfock.bragitoff.com">Homepage</a>
    ·
    <a href="https://pyfock-gui.bragitoff.com">Try the GUI</a>
    ·
    <a href="https://www.kaggle.com/code/ducktape07/pyfock-tutorial">View Demo</a>
    ·
    <a href="https://github.com/manassharma07/pyfock/issues">Report Bug</a>
    ·
    <a href="https://github.com/manassharma07/pyfock/issues">Request Feature</a>
  </p>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#about-the-project">About The Project</a></li>
    <li><a href="#key-features">Key Features</a></li>
    <li><a href="#installation">Installation</a></li>
    <li><a href="#quick-start">Quick Start</a></li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#graphical-user-interface">Graphical User Interface</a></li>
    <li><a href="#tutorials">Tutorials</a></li>
    <li><a href="#documentation">Documentation</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#citation">Citation</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

## About The Project

[![PyFock Screenshot][product-screenshot]](https://github.com/manassharma07/pyfock)

**PyFock** is a pure Python quantum chemistry package that enables efficient Kohn-Sham density functional theory (DFT) calculations for molecular systems. Unlike traditional quantum chemistry codes written in Fortran or C/C++, PyFock is written entirely in Python—including all performance-critical molecular integral evaluations—while achieving computational efficiency comparable to established codes like PySCF and Psi4.

### What Makes PyFock Different?

* **100% Pure Python**: All code, including computationally intensive molecular integrals, is written in Python
* **High Performance**: Achieves efficiency comparable to C/C++ backends through Numba JIT compilation, NumPy, NumExpr, SciPy, and CuPy
* **GPU Acceleration**: Leverages CUDA via Numba and CuPy for up to 14× speedup on large systems
* **Easy Installation**: Simple `pip install` on all major operating systems (Linux, macOS, Windows)
* **Accessible**: Designed for education, prototyping, and community development
* **Near-Quadratic Scaling**: ~O(N²·⁰⁵) scaling through density fitting with Cauchy-Schwarz screening
* **Gaussian-Type Orbitals**: Employs GTOs as basis functions for molecular calculations
* **Efficient Parallelization**: Multi-core CPU support and multi-GPU acceleration capabilities

### Performance Highlights

- **Numerical accuracy**: Consistent with PySCF (< 10⁻⁷ Ha)
- **Parallel efficiency**: Comparable to state-of-the-art C++ backends on multicore CPUs
- **GPU speedup**: Up to 14× faster than 4-core CPU execution for large systems
- **Scaling**: Near-quadratic ~O(N²·⁰⁵) for electron repulsion integrals (Coulomb term)
- **XC evaluation**: Sub-quadratic scaling ~O(N¹·²⁵⁻¹·⁵) for exchange-correlation contributions

## Key Features

- ✅ **Pure Python Implementation**: Including molecular integral evaluations (overlap, kinetic, nuclear attraction, electron repulsion integrals)
- ✅ **Density Fitting**: Efficient density fitting approximation with Cauchy-Schwarz screening
- ✅ **GPU Acceleration**: Full GPU support for integral evaluation, XC term, and matrix operations. `use_gpu=True` runs the XC term in single precision until the relative energy change drops below 5e-7 and in double precision from there on (`dynamic_precision`, on by default since it is ~1.5x faster for the same converged energy; automatically disabled for meta-GGAs, whose tau single precision resolves too poorly)
- ✅ **Multiple Integration Schemes**: 
  - Classical Taketa-Huzinaga-O-ohata scheme
  - Rys quadrature method (roots 1–10) for efficient ERI evaluation
  - Obara-Saika method for ERI evaluation
- ✅ **XC Functionals**: Support for LDA, GGA and meta-GGA functionals natively and optionally via LibXC integration, plus the Skala neural functional
- ✅ **DIIS Convergence**: Direct inversion of iterative subspace for SCF acceleration
- ✅ **Parallel Execution**: Multi-core CPU and multi-GPU support via Numba and Joblib
- ✅ **Modular Design**: Standalone integral modules for benchmarking and embedding
- ✅ **Web-based GUI**: Interactive interface for visualization and input generation
- ✅ **Cartesian and Spherical Basis**: Support for both CAO and SAO representations
- ✅ **Effective Core Potentials**: Support for evaluation of ECP integrals
- ✅ **Analytical gradients & forces**: Fast analytical nuclear gradients for density-fitted DFT — one-electron (overlap/kinetic/nuclear), DF Coulomb (3c2e + 2c2e), and XC for LDA, GGA and meta-GGA (native or LibXC) — matching PySCF forces and faster
- ✅ **ASE Calculator**: Optional ASE interface (geometry optimization and the wider ASE ecosystem), using analytical forces by default
- ✅ **XC integration grids**: Treutler-Ahlrichs radial and Lebedev angular grids with region-wise angular pruning and Becke partitioning (levels 0-9), built by a parallel Numba kernel or on the GPU (~6x faster, same grid); energies agree with PySCF to the SCF convergence threshold in our benchmarks; numgrid grids remain available as an alternative scheme
- ✅ **SANO initial guess**: superposition of atomic natural-orbital densities (ANO-RCC-MB natural orbitals projected onto the calculation basis, no atomic SCF needed) as the default SCF starting point; it roughly halves the number of SCF iterations relative to the core-Hamiltonian guess and prunes the XC grid accurately
- ✅ **Cross-Platform**: Works on Linux, macOS, and Windows

## Installation

### Basic Installation

PyFock can be easily installed via pip:

```bash
pip install pyfock
```

### Installing from GitHub (Latest Development Version)

To get the latest development version directly from GitHub:

```bash
pip install git+https://github.com/manassharma07/pyfock.git
```

Or clone the repository and install locally:

```bash
git clone https://github.com/manassharma07/pyfock.git
cd pyfock
pip install -e .
```

### Installing LibXC (Optional Dependency)

PyFock can use LibXC for exchange-correlation functionals not available natively in PyFock. The installation method depends on your system:

#### Using Conda (Recommended - Easiest Method)

```bash
conda install -c conda-forge pylibxc -y
```

#### On Ubuntu/Debian

```bash
sudo apt-get install libxc-dev
pip install pylibxc2
```

#### On macOS

```bash
brew install libxc
pip install pylibxc2
```

**Note**: The conda method is recommended as it works reliably across all platforms.

### Optional Dependencies

For GPU acceleration:
```bash
pip install cupy-cuda11x  # Replace 11x with your CUDA version
```

For the ASE calculator (geometry optimization and the ASE ecosystem):
```bash
pip install ase           # or: pip install pyfock[ase]
```
PyFock itself imports and runs without ASE installed; ASE is only required when you use `PyFockCalculator`.

For the **Skala** neural exchange-correlation functional:
```bash
pip install torch huggingface_hub
```
PyFock loads Skala's published TorchScript checkpoint directly with `torch.jit.load`, so the `skala`
package itself is **not** needed — and neither are its dependencies PySCF (which has no Windows wheels)
and e3nn. See [Skala: the neural exchange-correlation functional](#skala-the-neural-exchange-correlation-functional).

## Quick Start

Here's a minimal example to get you started with PyFock:

```python
from pyfock import Basis, Mol, DFT

# Define molecule from XYZ file
mol = Mol(coordfile='h2o.xyz')

# Set up basis sets
basis = Basis(mol, {'all': Basis.load(mol=mol, basis_name='def2-SVP')})
auxbasis = Basis(mol, {'all': Basis.load(mol=mol, basis_name='def2-universal-jfit')})

# Create DFT calculation object
dftObj = DFT(mol, basis, auxbasis, xc='PBE')

# Set calculation parameters
dftObj.conv_crit = 1e-7
dftObj.max_itr = 20
dftObj.ncores = 4

# Run SCF calculation
energy, dmat = dftObj.scf()
print(f"Total Energy: {energy} Ha")
```

## Usage

### Computing Molecular Integrals

PyFock provides standalone access to all molecular integrals:

```python
from pyfock import Integrals, Basis, Mol

mol = Mol(coordfile='h2o.xyz')
basis = Basis(mol, {'all': Basis.load(mol=mol, basis_name='def2-SVP')})

# One-electron integrals
S_ovlp = Integrals.overlap_mat_symm(basis)
V_kin = Integrals.kin_mat_symm(basis)
V_nuc = Integrals.nuc_mat_symm(basis, mol)

# Two-electron integrals (classical scheme)
ERI_slow = Integrals.conv_4c2e_symm(basis)

# Two-electron integrals (Rys quadrature - faster)
ERI_fast = Integrals.rys_4c2e_symm(basis)

# Three-center integrals for density fitting
ERI_3c2e = Integrals.rys_3c2e_symm(basis, auxbasis)

# Two-center integrals
ERI_2c2e = Integrals.rys_2c2e_symm(basis)
```

### GPU-Accelerated Integrals

```python
# GPU versions (returns CuPy arrays in device memory)
S_ovlp_gpu = Integrals.overlap_mat_symm_cupy(basis)
V_kin_gpu = Integrals.kin_mat_symm_cupy(basis)
V_nuc_gpu = Integrals.nuc_mat_symm_cupy(basis, mol)
ERI_3c2e_gpu = Integrals.rys_3c2e_symm_cupy(basis, auxbasis)
```

### Converting Between Cartesian and Spherical Basis

```python
# Convert from Cartesian to Spherical atomic orbitals
V_kin_CAO = Integrals.kin_mat_symm(basis)
c2sph_mat = basis.cart2sph_basis()
V_kin_SAO = np.dot(c2sph_mat, np.dot(V_kin_CAO, c2sph_mat.T))
```

### Subset Evaluation

```python
# Evaluate integrals for a subset of basis functions
S_ovlp_subset = Integrals.overlap_mat_symm(basis, slice=[0, 5, 0, 5])
# slice = [row_start, row_end, col_start, col_end]
```

### Full DFT Calculation Example

```python
from pyfock import Basis, Mol, DFT

# Initialize molecule
xyzFilename = 'benzene.xyz'
mol = Mol(coordfile=xyzFilename)

# Set up basis sets
basis_set_name = 'def2-SVP'
auxbasis_name = 'def2-universal-jfit'
basis = Basis(mol, {'all': Basis.load(mol=mol, basis_name=basis_set_name)})
auxbasis = Basis(mol, {'all': Basis.load(mol=mol, basis_name=auxbasis_name)})

# Configure XC functional (PBE)
funcx = 101  # Exchange
funcc = 130  # Correlation
funcidcrysx = [funcx, funcc]

# Initialize DFT object
dftObj = DFT(mol, basis, auxbasis, xc=funcidcrysx)

# Configure convergence and parallelization
dftObj.conv_crit = 1e-7
dftObj.max_itr = 20
dftObj.ncores = 4

# Run calculation
energyCrysX, dmat = dftObj.scf()
print(f"SCF Energy: {energyCrysX} Ha")
```

### Skala: the neural exchange-correlation functional

[Skala](https://github.com/microsoft/skala) is a machine-learned exchange-correlation functional from
Microsoft Research AI for Science that reaches hybrid-like accuracy at semilocal cost. Pass its name as
the `xc` argument and nothing else changes:

```python
from pyfock import Basis, DFT, Grids, Mol

mol      = Mol(coordfile='H2O.xyz')
basis    = Basis(mol, {'all': Basis.load(mol=mol, basis_name='def2-SVP')})
auxbasis = Basis(mol, {'all': Basis.load(mol=mol, basis_name='def2-universal-jfit')})

dftObj = DFT(mol, basis, auxbasis, xc='skala-1.1', grids=Grids(mol, level=3))
dftObj.conv_crit = 1e-8
energy, dmat = dftObj.scf()
```

Available names are `skala-1.1` (recommended), `skala-1.1-rev1`, `skala-1.1-rev0` and `skala-1.0`.

#### Installation

Only PyTorch is required at run time:

```bash
pip install torch huggingface_hub
```

Skala's own `pyproject.toml` lists PySCF and e3nn as dependencies, but those are needed only by
`skala.pyscf`, `skala.gpu4pyscf`, `skala.ase` and the *trainable* model definition. PyFock uses none of
them: it reads the published TorchScript checkpoint directly, so `pip install skala` is unnecessary. This
also removes the Linux/macOS restriction in Skala's packaging, which comes from PySCF — the checkpoints
are platform-independent and **Skala works on Windows through PyFock**.

The 2.4 MB checkpoint is downloaded from Hugging Face on first use and cached afterwards. Its SHA-256 is
verified against the digests published by Microsoft before it is loaded, because TorchScript
deserialization executes code from the file. To run without network access — or without
`huggingface_hub` at all — download `skala-1.1-rev1.fun` from
[huggingface.co/microsoft/skala-1.1](https://huggingface.co/microsoft/skala-1.1) and set:

```bash
export SKALA_LOCAL_MODEL_PATH=/path/to/skala-1.1-rev1.fun
```

Note that this path bypasses hash verification, so only point it at a file you trust.

#### What is different about it

Every other functional in PyFock is pointwise: the energy density at a grid point depends only on the
density at that point. Skala is not — its non-local layers aggregate over the points of each atomic grid,
so it consumes the whole grid at once and returns the total XC energy as a single number, and the
potential comes from automatic differentiation of that number. PyFock therefore evaluates it in three
passes (density over the whole grid → one model call → potential), rather than in the single fused
blocked loop used for LDA/GGA/meta-GGA. The derivatives the model returns are exactly the
`vrho`/`vsigma`/`vtau` intermediates the meta-GGA path already contracts with the AO values, so the
potential assembly itself is unchanged.

Consequences worth knowing:

- **Grids must be the native `'treutler'` ones** (the default). Skala needs both the Becke-partitioned
  weights and the raw single-atom weights.
- **Do not density-prune the grid.** The model's non-local features are integrals over each atom's full
  grid.
- **CPU only for now.** `use_gpu=True` raises a clear error: the GPU path needs a CUDA build of PyTorch
  plus a CuPy↔Torch bridge (zero-copy through DLPack) that is not wired up yet.
- **Closed-shell (restricted) only**, like the rest of PyFock's DFT.
- **Analytical nuclear gradients are not available.** Forces would need the Pulay and grid-weight
  derivative terms propagated through the network; use `DFT_NumGrad` for numerical forces.
- **Dispersion is not included.** Skala 1.1 expects an additive DFT-D3 correction with B3LYP5 parameters
  (`dftObj.skala.d3_settings()` returns `'b3lyp5'`). It does not enter the SCF, and PyFock does not add
  it automatically, so the energy reported is the bare Skala energy. Add it separately if you are
  comparing against published Skala numbers.
- **The model carries about 1e-9 Ha of its own numerical noise.** Presenting the network with differently
  shaped batches (a different `max_points_per_chunk`) shifts the energy at that level. Within one
  calculation the chunking is fixed, so the shift is systematic rather than random and SCF convergence to
  `1e-8` is unaffected — but do not expect two runs with different chunk sizes to agree bit for bit.

#### Cost

Skala is several times more expensive per SCF iteration than a semilocal functional. Measured with
`benchmarks_tests/benchmark_skala.py` (def2-SVP, level-3 grid, 4 CPU cores, density fitting):

| | H2O (3 atoms, 25 AOs, 34k grid points) | Caffeine (24 atoms, 260 AOs, 295k grid points) |
|---|---|---|
| PBE | -76.2740944860 Ha, 7 iter, 0.9 s | -679.1219071931 Ha, 15 iter, 20.1 s |
| r2SCAN | -76.3187302439 Ha, 7 iter, 1.1 s | -679.5341593969 Ha, 12 iter, 29.2 s |
| **skala-1.1** | **-76.3236300555 Ha, 7 iter, 10.0 s** | **-679.5460711592 Ha, 14 iter, 194.8 s** |

One XC evaluation on an identical density and grid, which isolates the functional from any difference
in iteration count:

| | H2O | Caffeine |
|---|---|---|
| PBE | 0.036 s | 1.17 s |
| r2SCAN | 0.109 s | 2.11 s |
| skala-1.1 | 0.791 s | 12.57 s |
| *of which the model itself* | *0.709 s (90%)* | *8.69 s (69%)* |
| **slowdown vs PBE / r2SCAN** | **21.7x / 7.3x** | **10.8x / 5.9x** |

Two things are worth reading out of this. First, the three-pass restructuring that non-locality forces
on PyFock is *not* where the time goes: the extra density and potential passes cost 0.07 s of H2O's
0.79 s, and the neural network is essentially the whole overhead. Second, the relative cost **falls** as
the system grows (21.7x to 10.8x against PBE), because the model's cost tracks the number of grid points
while the semilocal functionals also carry AO work that grows with the basis.


### Initial Guess for the SCF

The SCF starts by default from the **SANO** guess (superposition of atomic natural-orbital
densities): the spherically averaged density of every free atom, written in the ANO-RCC-MB
minimal basis whose contracted functions are the atomic natural orbitals of Roos and
co-workers, projected onto the calculation basis and renormalized shell by shell. It is the
PyFock analogue of PySCF's `minao` guess, costs a fraction of a second and needs no atomic
SCF. The core-Hamiltonian guess remains available:

```python
dft_obj = DFT(mol, basis, auxbasis, xc='PBE', dmat_guess_method='sano')   # default
dft_obj = DFT(mol, basis, auxbasis, xc='PBE', dmat_guess_method='core')   # core-Hamiltonian guess

# The guess density itself (CAO representation) and its references
from pyfock import Guess
dmat_guess, info = Guess.sano_dmat(mol, basis)
print(Guess.sano_citation_text(info['nuclear_charges']))
```

The SCF output lists the references to cite for the guess. Compared with the core guess it
saves 12-63 % of the SCF iterations, 41 % over a 22-case benchmark (e.g. decane/def2-SVP/PBE
18 -> 8, cholesterol 27 -> 10), and, because PyFock prunes the XC grid with the starting
density, it also removes a grid-pruning error of the too-compact core-Hamiltonian density
(3e-4 Ha for decane, more for heavy atoms). See [docs/sano_guess.md](docs/sano_guess.md) for
the method, the benchmark table and the tests.

### XC Integration Grids

The exchange-correlation term is integrated on Treutler-Ahlrichs radial grids combined with Lebedev
angular grids (tables from numgrid), pruned region by region and joined by Becke partitioning
with Treutler's atomic-size adjustment. `gridsLevel` runs from 0 (coarsest) to 9 (finest), default 3.
In our benchmarks the resulting SCF energies agree with PySCF's at the same level to the convergence
threshold, and the grid is built in about the time PySCF needs (decane: 0.11 s vs 0.14 s).

```python
dftObj = DFT(mol, basis, auxbasis, xc='PBE', gridsLevel=3)            # default: grids_scheme='treutler'
dftObj.grids_options = {'pruning': None, 'size_adjustment': 'becke'} # variants of the default scheme
dftObj.grids_options = {'points_per_element': {'C': (75, 302)}}      # (n_rad, n_ang) overrides per element
dftObj.grids_scheme = 'numgrid'; dftObj.grids_preset = 'compact'     # numgrid grids ('dense': the old PyFock grids)
dftObj.use_pyscf_grids = True                                        # or let PySCF build the grid (PySCF must be installed)
```

A GPU calculation (`use_gpu=True`) also builds the grid on the GPU: the points, their atom indices and
their box order are identical to the CPU build and the weights agree to ~1e-13, at about 6x the speed
from roughly 30 atoms on (decane 0.13 s -> 0.02 s, taxol 3.7 s -> 0.68 s). It falls back to the CPU with
a warning when CuPy or a CUDA device is missing; `grids_options={'use_gpu': False}` turns it off. The
density pruning that follows the build runs on the GPU too, with the same AO kernel the XC term uses:
it keeps exactly the same points as the CPU pruning at about 4.5x the speed (cholesterol 2.5 s -> 0.55 s).

Standalone grids come from `Grids(mol, level=3)` (`coords`, `weights` and the atom index of every
point, grouped into 1.2 Bohr boxes; `use_gpu=True` builds them on the GPU). See
[docs/xc_grids.md](docs/xc_grids.md) for the construction, its validation and a study of what the
numgrid grids can and cannot do.

### Density-Fitting Coulomb Algorithms and Memory Budget

The Coulomb term is evaluated with density fitting and Schwarz screening. Two CPU/GPU
algorithms are available through `DFT.DF_algo`:

- `11` (default): shell-blocked Rys evaluation with block-sparse storage. It honours a
  memory budget for the stored integrals:
- `10`: the previous default, per-function Rys evaluation with sparse triangular storage
  of the screened three-center integrals. It gives the same energies (identical to
  ~1e-10 Hartree in the CPU comparisons). It remains selectable for benchmarking.

```python
dft_obj.max_memory_ints3c2e = 2.0   # GB; None (default) = store everything, 0 = recompute every SCF iteration
```

With a budget smaller than the significant integrals, the most expensive shell-pair
blocks are kept in memory and the cheaper ones are re-evaluated in every SCF cycle.
With `use_gpu=True`, algorithm 11 evaluates and contracts shell blocks on the GPU
using CuPy and Numba-CUDA. The memory budget limits cached device values. Direct
pairs use an additional temporary buffer (up to 1 GB by default); the plan summary
reports its size. Cached blocks remain on the device even if
`keep_ints3c2e_in_gpu=False`; use `max_memory_ints3c2e` to control their storage.
The GPU integral builder supports orbital and auxiliary shells through i (`l=6`).
See [GPU design and validation](docs/df_algo11_gpu.md) for validation coverage,
benchmark commands, memory accounting, and limitations of the surrounding driver.

### Analytical Forces & Geometry Optimization

After a converged DFT calculation, analytical nuclear gradients (and forces)
are available directly via `DFT_Grad` (density fitting; LDA/GGA/meta-GGA; CPU):

```python
from pyfock import DFT_Grad

# dftObj must already be converged (dftObj.scf() called)
grad = DFT_Grad(dftObj)
result = grad.calculate()
forces = result["forces"]      # (natoms, 3) in Ha/Bohr
gradient = result["gradient"]  # = -forces
```

For geometry optimization, use the ASE calculator (requires `ase`). It uses
the analytical forces by default and falls back to finite differences only for
configurations the analytical gradients do not yet cover (e.g. HF, no DF):

```python
from ase import Atoms
from ase.optimize import BFGS
from pyfock import PyFockCalculator

water = Atoms("OHH", positions=[[0, 0, 0.119], [0, 0.763, -0.477], [0, -0.763, -0.477]])
water.calc = PyFockCalculator(functional="PBE", basis="def2-SVP",
                              auxbasis="def2-universal-jfit", ncores=4)
BFGS(water).run(fmax=0.02)
```

### Generating Visualization Files

```python
from pyfock import Utils

# Generate cube files for molecular orbitals and density
Utils.write_density_cube(dftObj, filename='benzene_density.cube')
```

## Graphical User Interface

PyFock includes a web-based GUI for interactive calculations and visualization:

🌐 **Try it online**: [https://pyfock-gui.bragitoff.com](https://pyfock-gui.bragitoff.com)

### GUI Features

- **Interactive 3D Visualization**: View molecules and molecular orbitals using Py3Dmol
- **Easy Configuration**: Select basis sets, functionals, and calculation parameters
- **Automatic Cube File Generation**: HOMO, LUMO, and density visualizations
- **Input Script Generator**: Export Python code for local execution
- **PySCF Validation**: Built-in comparison with PySCF for accuracy verification
- **Molecule Library**: Pre-loaded common molecules or custom XYZ input

### Running GUI Locally

The GUI source code is available on GitHub and can be run locally:

```bash
git clone https://github.com/manassharma07/PyFock-GUI.git
cd PyFock-GUI
pip install -r requirements.txt
streamlit run app.py
```

## Tutorials


### Interactive Jupyter Notebooks

🚀 **Coming Soon**: Interactive tutorials on Kaggle and Google Colab

- [x] [Kaggle Notebook: Introduction to PyFock](https://www.kaggle.com/code/ducktape07/pyfock-tutorial) 
- [x] [Kaggle Notebook: Advanced Features and GPU accelerated computations](https://www.kaggle.com/code/ducktape07/introduction-to-pyfock-tutorial)
- [x] [Google Colab Notebook: Introduction to PyFock](https://colab.research.google.com/drive/1d2QpcE2vLt7c6_s48r3GUXruMT7rB-ua?usp=sharing)
- [x] [Kaggle Notebook: Benchmarking PyFock against PySCF](https://www.kaggle.com/code/ducktape07/pyfock-vs-pyscf#10.-Timing-summary-table)
- [x] [Kaggle Notebook: PyFock GPU Dynamic Precision](https://www.kaggle.com/code/ducktape07/pyfock-gpu-dynamic-precision)
- [x] [Kaggle Notebook: PySCF vs. PyFock GPU Benchmark](https://www.kaggle.com/code/ducktape07/pyscf-vs-pyfock-gpu-benchmark)


## Documentation

📚 **Full Documentation**: [https://pyfock-docs.bragitoff.com](https://pyfock-docs.bragitoff.com)

- [Getting Started Guide](https://pyfock.bragitoff.com)
- [API Reference](https://pyfock-docs.bragitoff.com)
- [Examples and Tutorials](https://github.com/manassharma07/pyfock/tree/main/examples)

## Roadmap

- [x] Density Fitting with Cauchy-Schwarz screening
- [x] GPU acceleration for integrals and XC evaluation
- [x] DIIS convergence acceleration
- [x] Web-based GUI
- [x] Rys quadrature (roots 1–10)
- [x] Analytical nuclear gradients & forces (density fitting; LDA/GGA/meta-GGA; CPU)
- [x] ASE calculator & geometry optimization
- [ ] Analytical gradients on GPU and for non-DF / ECP calculations
- [ ] Electron dynamics & Excited state calculations (RT-TDDFT)
- [ ] Periodic boundary conditions
- [x] Hybrid functionals with exact exchange (native B3LYP/PBE0 and LibXC hybrids, RI-K via DF_algo=11; CPU)
- [x] Skala neural exchange-correlation functional (CPU)
- [ ] Multi-GPU parallelization
- [ ] Basis set optimization tools

See the [open issues](https://github.com/manassharma07/pyfock/issues) for a full list of proposed features and known issues.

## Contributing

Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make PyFock better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

Don't forget to give the project a star! ⭐ Thanks!

## License

Distributed under the **MIT License**. See `LICENSE` file for more information.


## Citation

If you use PyFock in your research, please cite:

```bibtex
@article{sharma2026pyfock,
  title        = {PyFock: A Just-In-Time Compiled Gaussian Basis DFT Python Code for CPU and GPU Architectures},
  author       = {Sharma, Manas and Sierka, Marek},
  journal = {The Journal of Physical Chemistry A},
  year = {2026},
  month = {08},
  issn = {1089-5639},
  doi = {10.1021/acs.jpca.6c03727},
  url = {https://doi.org/10.1021/acs.jpca.6c03727},
  eprint = {https://pubs.acs.org/jpcafh/article-pdf/doi/10.1021/acs.jpca.6c03727/67108190/acs.jpca.6c03727.pdf},
}
```
[Journal Article Link](https://pubs.acs.org/jpcafh/article/doi/10.1021/acs.jpca.6c03727/5298372/PyFock-A-Just-In-Time-Compiled-Gaussian-Basis-DFT)

**PyPI Package**: [https://pypi.org/project/pyfock/](https://pypi.org/project/pyfock/)

## Contact

**Manas Sharma**
- Email: manas.sharma@uni-jena.de
- Website: [manas.bragitoff.com](https://manas.bragitoff.com)
- LinkedIn: [linkedin.com/in/manassharma07](https://www.linkedin.com/in/manassharma07)
- Project Homepage: [https://pyfock.bragitoff.com](https://pyfock.bragitoff.com)
- Project Link: [https://github.com/manassharma07/pyfock](https://github.com/manassharma07/pyfock)


---

**Built With**

- [![Python][Python-badge]][Python-url]
- [![Numba][Numba-badge]][Numba-url]
- [![NumPy][NumPy-badge]][NumPy-url]
- [![CuPy][CuPy-badge]][CuPy-url]
- [![SciPy][SciPy-badge]][SciPy-url]

<!-- MARKDOWN LINKS & IMAGES -->
[contributors-shield]: https://img.shields.io/github/contributors/manassharma07/crysx_nn.svg?style=for-the-badge
[contributors-url]: https://github.com/manassharma07/PyFock/contributors
[forks-shield]: https://img.shields.io/github/forks/manassharma07/PyFock.svg?style=for-the-badge
[forks-url]: https://github.com/manassharma07/PyFock/network/members
[stars-shield]: https://img.shields.io/github/stars/manassharma07/PyFock.svg?style=for-the-badge
[stars-url]: https://github.com/manassharma07/PyFock/stargazers
[issues-shield]: https://img.shields.io/github/issues/manassharma07/PyFock.svg?style=for-the-badge
[issues-url]: https://github.com/manassharma07/PyFock/issues
[license-shield]: https://img.shields.io/github/license/manassharma07/PyFock.svg?style=for-the-badge
[license-url]: https://github.com/manassharma07/PyFock/blob/main/LICENSE
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://www.linkedin.com/in/manassharma07
[product-screenshot]: https://github.com/manassharma07/PyFock/blob/main/TOC_pyfock.webp
[documentation-url]: https://pyfock-docs.bragitoff.com
[docs-shield]: https://img.shields.io/badge/-docs-blue.svg?style=for-the-badge&logo=documentation&colorB=389
[Python-badge]: https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white
[Python-url]: https://python.org
[Numba-badge]: https://img.shields.io/badge/Numba-00A3E0?style=for-the-badge&logo=numba&logoColor=white
[Numba-url]: https://numba.pydata.org
[NumPy-badge]: https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white
[NumPy-url]: https://numpy.org
[CuPy-badge]: https://img.shields.io/badge/CuPy-76B900?style=for-the-badge&logo=nvidia&logoColor=white
[CuPy-url]: https://cupy.dev
[SciPy-badge]: https://img.shields.io/badge/SciPy-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white
[SciPy-url]: https://scipy.org
