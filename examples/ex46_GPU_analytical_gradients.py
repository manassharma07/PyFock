"""Analytical nuclear gradients on the GPU.

``DFT_Grad`` follows the SCF: converge with ``use_gpu=True`` and the gradient is evaluated on the
device as well, by a port of every CPU term except the ECP one. Pass ``use_gpu`` explicitly to the
constructor to override that -- which is what this example does, so the two devices can be compared
from a single converged SCF.

The device results agree with the host ones to round-off; what changes is the wall clock, and only
once the molecule is big enough to fill the GPU. Water is not: run it on caffeine or larger to see
the gradient speed up. ``benchmarks_tests/benchmark_DFT_gradients_gpu.py`` does that systematically.
"""
import numpy as np
from timeit import default_timer as timer

from pyfock import Basis, DFT, DFT_Grad, Mol

ncores = 4

mol = Mol(coordfile="h2o.xyz")
basis = Basis(mol, {"all": Basis.load(mol=mol, basis_name="def2-SVP")})
auxbasis = Basis(mol, {"all": Basis.load(mol=mol, basis_name="def2-universal-jfit")})

dft_obj = DFT(mol, basis, auxbasis, xc="PBE")
dft_obj.conv_crit = 1e-9
dft_obj.max_itr = 30
dft_obj.ncores = ncores
dft_obj.save_ao_values = True
dft_obj.use_gpu = True

dft_obj.scf()

# The first call pays for compiling the CUDA kernels, so time the second one.
DFT_Grad(dft_obj, verbose=False, use_gpu=True).calculate()
start = timer()
gpu = DFT_Grad(dft_obj, verbose=False, use_gpu=True).calculate()
t_gpu = timer() - start

start = timer()
cpu = DFT_Grad(dft_obj, verbose=False, use_gpu=False).calculate()
t_cpu = timer() - start

print("Total energy (Ha):", gpu["energy"])
print("Forces (Ha/Bohr):")
print(np.array2string(gpu["forces"], precision=8, suppress_small=False))
print("gradient: CPU %.3f s, GPU %.3f s" % (t_cpu, t_gpu))
print("max |GPU - CPU| (Ha/Bohr): %.3e" % np.abs(gpu["gradient"] - cpu["gradient"]).max())
