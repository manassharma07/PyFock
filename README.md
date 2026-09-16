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
    <li><a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#what-makes-pyfock-different">What Makes PyFock Different?</a></li>
        <li><a href="#performance-highlights">Performance Highlights</a></li>
      </ul>
    </li>
    <li><a href="#key-features">Key Features</a></li>
    <li><a href="#installation">Installation</a>
      <ul>
        <li><a href="#basic-installation">Basic Installation</a></li>
        <li><a href="#installing-from-github-latest-development-version">Installing from GitHub</a></li>
        <li><a href="#installing-libxc-optional-dependency">Installing LibXC</a></li>
        <li><a href="#optional-dependencies">Optional Dependencies</a></li>
      </ul>
    </li>
    <li><a href="#quick-start">Quick Start</a></li>
    <li><a href="#usage">Usage</a>
      <ul>
        <li><a href="#computing-molecular-integrals">Computing Molecular Integrals</a></li>
        <li><a href="#gpu-accelerated-integrals">GPU-Accelerated Integrals</a></li>
        <li><a href="#converting-between-cartesian-and-spherical-basis">Cartesian &lt;-&gt; Spherical Basis</a></li>
        <li><a href="#subset-evaluation">Subset Evaluation</a></li>
        <li><a href="#skala-the-neural-exchange-correlation-functional">Skala: the neural XC functional</a>
          <ul>
            <li><a href="#getting-the-model">Getting the model</a></li>
            <li><a href="#what-is-different-about-it">What is different about it</a></li>
            <li><a href="#dispersion">Dispersion</a></li>
            <li><a href="#validating-against-the-published-reference-energies">Validating against the published reference energies</a></li>
            <li><a href="#cost">Cost</a></li>
            <li><a href="#skala-on-the-gpu">Skala on the GPU</a></li>
          </ul>
        </li>
        <li><a href="#initial-guess-for-the-scf">Initial Guess for the SCF</a></li>
        <li><a href="#xc-integration-grids">XC Integration Grids</a></li>
        <li><a href="#density-fitting-coulomb-algorithms-and-memory-budget">Density-Fitting Coulomb Algorithms and Memory Budget</a></li>
        <li><a href="#analytical-forces--geometry-optimization">Analytical Forces &amp; Geometry Optimization</a>
          <ul><li><a href="#grid-response">Grid response</a></li></ul>
        </li>
        <li><a href="#generating-visualization-files">Generating Visualization Files</a></li>
      </ul>
    </li>
    <li><a href="#graphical-user-interface">Graphical User Interface</a>
      <ul>
        <li><a href="#gui-features">GUI Features</a></li>
        <li><a href="#running-gui-locally">Running GUI Locally</a></li>
      </ul>
    </li>
    <li><a href="#tutorials">Tutorials</a>
      <ul>
        <li><a href="#interactive-jupyter-notebooks">Interactive Jupyter Notebooks</a></li>
      </ul>
    </li>
    <li><a href="#documentation">Documentation</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#citation">Citation</a></li>
    <li><a href="#contact">Contact</a></li>
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

None of these are needed to import or run PyFock. Each is available as a pip *extra*:

| Extra | Installs | Needed for |
|---|---|---|
| `pyfock[ase]` | `ase` | the `PyFockCalculator` ASE interface: geometry optimization, NEB, MD |
| `pyfock[dispersion]` | `dftd3` | DFT-D3 corrections, `DFT(..., dispersion=...)` |
| `pyfock[dispersion-gpu]` | `torch-dftd` | evaluating D3 on a GPU, through the ASE calculator |
| `pyfock[skala]` | `torch`, `huggingface_hub`, `dftd3` | the Skala neural functional |

Extras combine as usual:

```bash
pip install "pyfock[ase,skala]"
```

and work the same when installing from a clone:

```bash
pip install -e ".[ase,skala]"
```

GPU acceleration is kept separate, because the right wheel depends on your CUDA version:

```bash
pip install cupy-cuda11x  # Replace 11x with your CUDA version
```

Two notes:

- ASE is only required when you actually use `PyFockCalculator`; PyFock imports and runs without it.
- `pyfock[skala]` does **not** install the `skala` package. PyFock reads the published TorchScript
  checkpoint directly with `torch.jit.load`, which avoids that package's own dependencies on PySCF
  (no Windows wheels) and e3nn. `dftd3` is included because Skala is parametrised together with a
  DFT-D3 correction, and it is what reproduces Skala's published numbers. See
  [Skala: the neural exchange-correlation functional](#skala-the-neural-exchange-correlation-functional).

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

`xc` also accepts LibXC IDs instead of a name — `xc=[101, 130]` is the same PBE as `xc='PBE'`.

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
See [`examples/ex44_Skala_neural_functional.py`](examples/ex44_Skala_neural_functional.py) for a runnable
version, with and without the dispersion correction.

#### Getting the model

`pip install "pyfock[skala]"` is all that is needed (see
[Optional Dependencies](#optional-dependencies)) — notably *not* the `skala` package itself, which is
why this works on Windows despite Skala's own packaging being Linux/macOS only.

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
- **The GPU is supported**, either way round: `use_gpu=True` runs the whole SCF on the device, and
  `skala_gpu=True` moves only the neural functional there and leaves the rest on the CPU. See
  [Skala on the GPU](#skala-on-the-gpu). Forces are still CPU-only.
- **Closed-shell (restricted) only**, like the rest of PyFock's DFT.
- **Analytical nuclear gradients are supported**, so Skala works for geometry optimization like any
  other functional — see
  [Analytical Forces & Geometry Optimization](#analytical-forces--geometry-optimization). Unlike the
  semilocal path, the Skala gradient includes the full grid response (Becke weight derivatives and the
  grid-translation term); its features are integrals over each atomic grid, so a frozen grid would cost
  ~1e-2 Ha/Bohr rather than the ~1e-4 it costs a meta-GGA.
- **Dispersion is off by default** — pass `dispersion=True` to include it, see below.
- **The model carries about 1e-9 Ha of its own numerical noise.** Presenting the network with differently
  shaped batches (a different `max_points_per_chunk`) shifts the energy at that level. Within one
  calculation the chunking is fixed, so the shift is systematic rather than random and SCF convergence to
  `1e-8` is unaffected — but do not expect two runs with different chunk sizes to agree bit for bit.

#### Dispersion

Skala 1.1 is parametrised *together with* a DFT-D3 correction — the checkpoint declares it, and
`dftObj.skala.d3_settings()` returns `'b3lyp5'`, meaning D3(BJ) damping with B3LYP5 parameters and no
three-body term. Leaving it out is fine for comparisons against Skala's own reference energies (which
exclude it) but wrong for anything where dispersion matters, such as non-covalent interactions or
conformer ranking.

D3 depends only on the atomic numbers, the coordinates and those damping parameters — not on the
density. It never enters the Kohn-Sham matrix and cannot change the SCF, so PyFock evaluates it once
after convergence and adds it to the total energy:

```bash
pip install dftd3
```

```python
dftObj = DFT(mol, basis, auxbasis, xc='skala-1.1', dispersion=True)
energy, dmat = dftObj.scf()      # dispersion-corrected total energy
print(dftObj.Edisp)              # the correction on its own, in Hartree
```

`dispersion=True` uses whatever the functional declares. For any other functional, name the
parametrisation yourself — `dispersion='pbe'`, `dispersion='b3lyp'` — and tune the damping through
`dftObj.dispersion_version` (default `'d3bj'`) and `dftObj.dispersion_atm` (default `False`).
[`pyfock.Dispersion`](pyfock/Dispersion.py) also exposes `d3_energy` and `d3_energy_and_gradient`
directly, the latter giving `dE_disp/dR` for forces. See
[`examples/ex43_D3_dispersion_correction.py`](examples/ex43_D3_dispersion_correction.py).

The ASE calculator uses the same backend by default:

```python
PyFockCalculator(functional='PBE', dispersion=True, dispersion_kwargs={'xc': 'pbe'})
```

`torch-dftd` (`pip install pyfock[dispersion-gpu]`) remains available there for GPU runs, where it
evaluates the correction on the device. Select it with `backend='torch-dftd'`, or implicitly by asking
for a non-CPU device:

```python
dispersion_kwargs={'xc': 'pbe', 'backend': 'torch-dftd', 'device': 'cuda'}
```

Note that Skala's own `skala.dispersion` module routes D3 through PySCF (`pyscf.dispersion.dftd3` or
`dftd3.pyscf`), which is why PyFock does not use it. PyFock calls `dftd3.interface` from the same
`simple-dftd3` package instead — identical numbers, no PySCF, and Windows wheels are available.

#### Validating against the published reference energies

Skala ships the total energies behind its benchmark report as `benchmark/reference/measurements.json`
(a Git-LFS file, ~7.8 MB): 1221 converged runs over skala-1.1, r2scan, m06-2x and b3lyp5, each at
def2-SVP/TZVP/QZVP, for ~30 molecules from GMTKN55 and a conformer benchmark. The geometries are pulled
from `grimme-lab/GMTKN55` at a pinned commit.

One detail is easy to get wrong: the **Skala energies there include the D3 correction**, because the
runner builds the method through `SkalaKS`, whose constructor defaults to `with_dftd3=True`. The plain
PySCF functionals in the same set do not. Comparing a bare PyFock Skala energy against them leaves a
residual exactly equal to the dispersion energy.

`benchmarks_tests/validate_skala_reference.py` downloads both, runs PyFock under the matching protocol
(spherical orbitals via `dftObj.sao = True`, density fitting with `def2-universal-jkfit`, grid level 3,
`conv_crit = 5e-6`, `dispersion=True` for Skala only) and tabulates the differences:

```bash
python validate_skala_reference.py --molecules H2O --basis def2-tzvp
```

It runs r2SCAN alongside Skala as a control: PyFock's r2SCAN is independently validated, so whatever it
shows against the same reference measures the grid, basis and density-fitting baseline rather than
anything to do with Skala. Over H2, H2O, H2O2 and H3N at def2-SVP and def2-TZVP:

| | mean abs. difference | max |
|---|---|---|
| skala-1.1 | 3.2e-08 Ha | 1.1e-07 Ha |
| r2SCAN (control) | 7.0e-08 Ha | 2.2e-07 Ha |

PyFock reproduces the published Skala energies to within the reference data's own run-to-run spread.
`tests/test_skala.py` keeps one of these values (H2 at def2-SVP) as a permanent regression check.

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

#### Skala on the GPU

There are two ways to use a GPU, and they compose with the rest of PyFock exactly as they do for any
other functional. `use_gpu=True` runs the whole SCF on the device — integrals, Coulomb term, AO values,
density, the model and the Vxc assembly:

```python
dftObj = DFT(mol, basis, auxbasis, xc='skala-1.1', dispersion=True, use_gpu=True)
```

`skala_gpu=True` instead moves only the neural functional and leaves the rest of the SCF on the CPU.
That is useful when the GPU path is unavailable or unwanted, since the model alone is most of Skala's
cost:

```python
dftObj = DFT(mol, basis, auxbasis, xc='skala-1.1', dispersion=True, skala_gpu=True)
atoms.calc = PyFockCalculator(functional='skala-1.1', skala_gpu=True, dispersion=True)  # same in ASE
```

Both need a CUDA build of PyTorch (`pip install torch --index-url
https://download.pytorch.org/whl/cu128` for current cards); PyFock downloads the CUDA checkpoint, which
is a separate file from the CPU one. `use_gpu=True` additionally needs CuPy, as it does for every
functional. Skala is a float64 model, so `dynamic_precision` is switched off for it automatically.

Non-locality is the only thing that changes on the device: Skala cannot be folded into the per-block
loop of `eval_xc_3_cupy`, where each block calls the functional on its own points, so it gets the same
three-pass driver as on the CPU (`Integrals.eval_xc_skala_cupy`). Everything inside those passes is the
semilocal GPU code.

`benchmarks_tests/benchmark_skala_gpu.py` measures the speedup and the CPU/GPU agreement, for
single points and for geometry optimizations. **Forces are CPU-only**: `DFT_Grad` raises for
`use_gpu=True`, so geometry optimization still runs the gradient on the host.

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

The Coulomb term is evaluated with density fitting and Schwarz screening. Three
algorithms are available through `DFT.DF_algo`:

- `11` (default): shell-blocked Rys evaluation with block-sparse storage (CPU/GPU). It honours a
  memory budget for the stored integrals:
- `12`: multipole-accelerated density fitting (CPU, pure functionals). The near-field
  three-center integrals are evaluated and stored as in `11`; well-separated pairs of a
  basis-function-product distribution and an auxiliary function are handled through
  multipole expansions (exact finite moments of the primitive pair products and of the
  auxiliary functions, box-level expansions truncated at `lmax`). Energies agree with `11`
  to ~1e-8 Hartree. Both the speed-up and the memory saving grow with the extent of the
  molecule, since the far field is what they come from (def2-SVP, 4 cores, three-center
  build plus all per-iteration Coulomb work):

  | System | far field | time vs `11` | stored integrals + moments |
  |---|---|---|---|
  | Caffeine (14 bohr across) | 4.0% | 0.9x | 0.18 -> 0.19 GB |
  | Cholesterol (35 bohr) | 32.4% | 1.5x | 1.61 -> 1.28 GB |
  | Icosane C20H42 (48 bohr) | 52.6% | 1.4x | 0.58 -> 0.37 GB |
  | Tetracontane C40H82 (95 bohr) | 74.7% | 2.7x | 2.41 -> 0.80 GB |

  The crossover is around 25 to 30 bohr of molecular extent; below it a group of distributions is
  mostly not expanded at all, because it is only expanded when it has enough far-field auxiliary
  functions to pay for the expansion. `{'low_memory': True}` drops the pre-translated box-centred
  moments and re-derives them each iteration, which takes icosane from 0.37 to 0.28 GB for about
  20% slower Coulomb iterations and identical energies.
  Parameters are set through `dft_obj.multipole_options`, e.g.
  `{'precision': 1e-10, 'lmax': 12, 'box_size': 2.5, 'separation': 4.0}`; see
  [docs/df_algo12_multipoles.md](docs/df_algo12_multipoles.md).
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
are available directly via `DFT_Grad` (density fitting; LDA/GGA/meta-GGA and Skala; CPU):

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

By default each step runs in a fresh subprocess, which keeps a crashing or non-converging step from
taking the optimizer down and leaves a full PyFock output on disk for every step. The price is that
every step repeats the cold start — the imports, loading the Skala checkpoint and warming up
TorchScript, and a CUDA context when `skala_gpu` is on — which for small molecules costs more than the
step itself. (PyFock's own Numba kernels are compiled with `cache=True`, so they reload from disk and
are a small part of it.) `run_in_process=True` runs the steps here instead and keeps all of that alive
between them:

```python
PyFockCalculator(functional="skala-1.1", run_in_process=True)
```

It is the faster choice for Skala by a wide margin; keep the subprocess for long unattended runs. BLAS
reads its thread count from the environment when numpy is first imported, so with `run_in_process=True`
set `OMP_NUM_THREADS` at the top of your script rather than relying on `ncores` alone. See
[`examples/ex45_ASE_geometry_optimization_with_Skala.py`](examples/ex45_ASE_geometry_optimization_with_Skala.py).

#### Grid response

By default the XC gradient treats the quadrature grid as fixed — its dependence on the nuclear
positions is not differentiated. Passing `grid_response=True` adds the two terms
that removes: the grid points of an atom translating with it, and the Becke weights depending on every
nucleus. The forces then become exactly translationally invariant:

```python
DFT_Grad(dftObj, grid_response=True).calculate()
PyFockCalculator(functional="PBE", basis="def2-SVP", grid_response=True)   # same flag in ASE
```

| net force (H2O / def2-SVP) | fixed grid | with grid response |
|---|---|---|
| LDA | 1.3e-05 | **2.9e-14** |
| PBE | 7.8e-06 | **1.5e-14** |
| r2SCAN | 8.9e-05 | **1.9e-14** |

It costs one extra pass over the partitioning and needs the native (`'treutler'`) grids. For semilocal
functionals it is a small correction and off by default, so existing results are unchanged. For Skala it
is **mandatory** and on by default — its features are integrals over each atomic grid, so a frozen grid
would be wrong by ~1e-2 Ha/Bohr, the size of the forces themselves.

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
- [x] Skala neural exchange-correlation functional, on CPU and GPU, with analytical gradients (CPU)
- [x] Grid-response XC gradients (Becke weight derivatives, `Grids.becke_weight_gradient`)
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
