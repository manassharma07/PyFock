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
- ✅ **GPU Acceleration**: Full GPU support for integral evaluation, XC term, and matrix operations
- ✅ **Multiple Integration Schemes**: 
  - Classical Taketa-Huzinaga-O-ohata scheme
  - Rys quadrature method (roots 1–10) for efficient ERI evaluation
- ✅ **XC Functionals**: Support for LDA and GGA functionals via LibXC integration
- ✅ **DIIS Convergence**: Direct inversion of iterative subspace for SCF acceleration
- ✅ **Parallel Execution**: Multi-core CPU and multi-GPU support via Numba and Joblib
- ✅ **Modular Design**: Standalone integral modules for benchmarking and embedding
- ✅ **Web-based GUI**: Interactive interface for visualization and input generation
- ✅ **Cartesian and Spherical Basis**: Support for both CAO and SAO representations
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

### Installing LibXC (Required Dependency)

PyFock requires LibXC for exchange-correlation functionals. The installation method depends on your system:

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

## Quick Start

Here's a minimal example to get you started with PyFock:

```python
from pyfock import Basis, Mol, DFT

# Define molecule from XYZ file
mol = Mol(coordfile='h2o.xyz')

# Set up basis sets
basis = Basis(mol, {'all': Basis.load(mol=mol, basis_name='def2-SVP')})
auxbasis = Basis(mol, {'all': Basis.load(mol=mol, basis_name='def2-universal-jfit')})

# Configure DFT calculation (PBE functional)
funcx = 101  # PBE exchange (LibXC code)
funcc = 130  # PBE correlation (LibXC code)
dftObj = DFT(mol, basis, auxbasis, xc=[funcx, funcc])

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
ERI_3c2e = Integrals.rys_3c2e_symm(basis)

# Two-center integrals
ERI_2c2e = Integrals.rys_2c2e_symm(basis)
```

### GPU-Accelerated Integrals

```python
# GPU versions (returns CuPy arrays in device memory)
S_ovlp_gpu = Integrals.overlap_mat_symm_cupy(basis)
V_kin_gpu = Integrals.kin_mat_symm_cupy(basis)
V_nuc_gpu = Integrals.nuc_mat_symm_cupy(basis, mol)
ERI_3c2e_gpu = Integrals.rys_3c2e_symm_cupy(basis)
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
- [ ] Google Colab: GPU-Accelerated DFT with PyFock
- [ ] Kaggle Notebook: Molecular Orbital Visualization
- [ ] Google Colab: Advanced Features and Benchmarking

## Documentation

📚 **Full Documentation**: [https://pyfock-docs.bragitoff.com](https://pyfock-docs.bragitoff.com)

- [Getting Started Guide](https://pyfock-docs.bragitoff.com)
- [API Reference](https://pyfock-docs.bragitoff.com)
- [Theory and Implementation](https://github.com/manassharma07/pyfock/blob/main/Documentation.md)
- [Examples and Tutorials](https://github.com/manassharma07/pyfock/tree/main/examples)

## Roadmap

- [x] Density Fitting with Cauchy-Schwarz screening
- [x] GPU acceleration for integrals and XC evaluation
- [x] DIIS convergence acceleration
- [x] Web-based GUI
- [x] Rys quadrature (roots 1–10)
- [ ] Analytical gradients (in progress)
- [ ] Electron dynamics & Excited state calculations (RT-TDDFT)
- [ ] Periodic boundary conditions
- [ ] Hybrid functionals with exact exchange
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

### Third-Party Code Acknowledgments

PyFock incorporates code adapted from several open-source projects:

- **Rys quadrature (roots 1–5)**: Adapted from [MolecularIntegrals.jl](https://github.com/rpmuller/MolecularIntegrals.jl) (MIT License)
- **Rys quadrature (roots 6–10)**: Ported from [PySCF's libcint](https://github.com/sunqm/libcint) (Apache v2 License)
- **DIIS-based SCF**: Adapted from [McMurchie-Davidson](https://github.com/jjgoings/McMurchie-Davidson) (BSD 3-Clause License)
- **Dipole matrix integrals**: Based on [McMurchie-Davidson](https://github.com/jjgoings/McMurchie-Davidson) (BSD 3-Clause License)
- **Boys Function**: Adapted from [PyBoys](https://github.com/peter-reinholdt/pyboys) (BSD 3-Clause License)
- **XC functionals**: Implementation based on [eminus](https://github.com/wangenau/eminus) (Apache v2 License)

## Citation

If you use PyFock in your research, please cite:

```bibtex
@article{sharma2025pyfock,
  title={PyFock: A Pure Python Gaussian Basis DFT Code for CPU and GPU},
  author={Sharma, Manas},
  journal={[Journal Name]},
  year={2025},
  note={Manuscript in preparation}
}
```

**PyPI Package**: [https://pypi.org/project/pyfock/](https://pypi.org/project/pyfock/)

## Contact

**Manas Sharma**
- Email: manassharma07@live.com
- Website: [manas.bragitoff.com](https://manas.bragitoff.com)
- LinkedIn: [linkedin.com/in/manassharma07](https://www.linkedin.com/in/manassharma07)
- Project Homepage: [https://pyfock.bragitoff.com](https://pyfock.bragitoff.com)
- Project Link: [https://github.com/manassharma07/pyfock](https://github.com/manassharma07/pyfock)

## Acknowledgments

* Prof. Dr. Marek Sierka and Friedrich Schiller University Jena for HPC access
* The open-source quantum chemistry community

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
[product-screenshot]: https://github.com/manassharma07/PyFock/blob/main/logo_crysx_pyfock.png
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