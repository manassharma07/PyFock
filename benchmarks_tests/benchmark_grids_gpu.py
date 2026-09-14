"""Compare the CPU and the GPU generation of PyFock's native ('treutler') XC grids.

For every molecule the grid of a level is built twice, once with the parallel Numba kernel and once on
the GPU (``Grids(..., use_gpu=True)``), and the script reports the build times, the speed-up and how far
the two grids differ. The points, their atom indices and their box order have to be identical; the
weights differ only by the floating-point contraction of the two compilers, so the maximum absolute
difference and the difference of the summed weights are reported.

The GPU build is timed after a warm-up, so the reported time excludes the one-off CUDA compilation of
the partition kernel (about 0.5 s the first time a molecule of a new size class is seen; the kernels are
cached on disk by Numba afterwards).

Usage:
    python3 benchmark_grids_gpu.py [molecule ...] [--levels 1,3,5] [--ncores 4]

The molecules are xyz names in this directory; the default set spans 3 to 453 atoms.
"""
import argparse
import os
import sys
from timeit import default_timer as timer

DEFAULT_MOLECULES = ['H2O', 'SnCl4', 'Decane_C10H22', 'C60', 'Cholesterol', 'Taxol', 'Olestra']

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('molecules', nargs='*', default=DEFAULT_MOLECULES,
                    help='names of xyz files in benchmarks_tests (without .xyz)')
parser.add_argument('--levels', default='3', help='grid levels to build')
parser.add_argument('--ncores', type=int, default=4, help='threads of the CPU build')
parser.add_argument('--repeats', type=int, default=1,
                    help='timed repetitions after a warm-up build (the best is reported); 0 times a single cold build')
args = parser.parse_args()

os.environ['OMP_NUM_THREADS'] = str(args.ncores)
import numpy as np
from pyfock import Mol, Grids
from pyfock import Grids_cupy

if not Grids_cupy.gpu_available():
    print('No CUDA device available (CuPy installed: %s).' % (Grids_cupy.cp is not None))
    sys.exit(1)

here = os.path.dirname(os.path.abspath(__file__))
levels = [int(x) for x in args.levels.split(',')]


def build(mol, level, use_gpu):
    best = float('inf')
    grids = None
    for _ in range(max(args.repeats + 1, 1)):   # the first pass warms up the JIT of either back end
        start = timer()
        grids = Grids(mol, level=level, ncores=args.ncores, verbose=False, use_gpu=use_gpu)
        best = min(best, timer() - start)
    return grids, best


print('%-22s %5s %6s %10s %9s %9s %8s %11s %11s' % (
    'molecule', 'atoms', 'level', 'points', 'CPU (s)', 'GPU (s)', 'speed-up', 'max |dw|', 'd(sum w)'))
for name in args.molecules:
    xyz = os.path.join(here, name + '.xyz')
    if not os.path.exists(xyz):
        print('%-22s missing %s' % (name, xyz))
        continue
    mol = Mol(coordfile=xyz)
    for level in levels:
        cpu, t_cpu = build(mol, level, False)
        gpu, t_gpu = build(mol, level, True)
        if not gpu.use_gpu:
            print('%-22s %5d %6d   the GPU build fell back to the CPU' % (name, mol.natoms, level))
            continue
        assert np.array_equal(cpu.coords, gpu.coords), 'the GPU points differ from the CPU ones'
        assert np.array_equal(cpu.atom_idx, gpu.atom_idx), 'the GPU atom indices differ from the CPU ones'
        print('%-22s %5d %6d %10d %9.3f %9.3f %8.1fx %11.2e %11.2e' % (
            name, mol.natoms, level, cpu.size, t_cpu, t_gpu, t_cpu / max(t_gpu, 1e-12),
            np.abs(cpu.weights - gpu.weights).max(), abs(cpu.weights.sum() - gpu.weights.sum())))
