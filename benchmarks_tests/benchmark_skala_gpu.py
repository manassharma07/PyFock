"""Skala on the GPU versus the CPU, in PyFock.

``skala_gpu=True`` moves only the neural functional to the device; the integrals, the Coulomb term and
the assembly of Vxc from the model's cotangents all stay on the CPU. That is where the time is -- Skala
is a PyTorch model and dominates the exchange-correlation cost -- so the device does the part that
matters without needing PyFock's full CuPy SCF path.

    python benchmark_skala_gpu.py single     wall time and CPU/GPU agreement for single points
    python benchmark_skala_gpu.py opt        two geometry optimizations, in both calculator modes

Run the two sections separately. Caffeine leaves a large heap behind, and the optimizations spawn
subprocesses that would then have nothing left to allocate from.
"""

import contextlib
import io
import os
import sys
import time

section = sys.argv[1] if len(sys.argv) > 1 else 'single'
if section not in ('single', 'opt'):
    raise SystemExit(__doc__)

ncores = int(os.environ.get('PYFOCK_NCORES', 4))
for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(variable, str(ncores))

import numpy as np
import torch
from pyfock import Basis, DFT, DFT_Grad, Mol

HERE = os.path.dirname(os.path.abspath(__file__))
EXAMPLES = os.path.join(HERE, '..', 'examples')
BASIS, AUXBASIS, CONV = 'def2-SVP', 'def2-universal-jfit', 1e-8

# No standalone benzene in the repo, so the usual D6h geometry goes here.
BENZENE = [['C', 0.0, 1.3970, 0.0], ['C', 1.2098, 0.6985, 0.0], ['C', 1.2098, -0.6985, 0.0],
           ['C', 0.0, -1.3970, 0.0], ['C', -1.2098, -0.6985, 0.0], ['C', -1.2098, 0.6985, 0.0],
           ['H', 0.0, 2.4810, 0.0], ['H', 2.1486, 1.2405, 0.0], ['H', 2.1486, -1.2405, 0.0],
           ['H', 0.0, -2.4810, 0.0], ['H', -2.1486, -1.2405, 0.0], ['H', -2.1486, 1.2405, 0.0]]


def single_point(mol, skala_gpu, save_ao_values, use_gpu=False):
    basis = Basis(mol, {'all': Basis.load(mol=mol, basis_name=BASIS)})
    auxbasis = Basis(mol, {'all': Basis.load(mol=mol, basis_name=AUXBASIS)})
    dft = DFT(mol, basis, auxbasis, xc='skala-1.1', dispersion=True, skala_gpu=skala_gpu,
              use_gpu=use_gpu)
    dft.conv_crit, dft.ncores, dft.save_ao_values = CONV, ncores, save_ao_values

    with contextlib.redirect_stdout(io.StringIO()):
        start = time.perf_counter()
        energy, _ = dft.scf()
        t_scf = time.perf_counter() - start
        forces, t_forces = None, 0.0
        if not use_gpu:      # DFT_Grad is CPU-only, so a full-GPU run reports the SCF alone
            start = time.perf_counter()
            forces = np.asarray(DFT_Grad(dft, verbose=False).calculate()['forces'])
            t_forces = time.perf_counter() - start
    return float(energy), forces, t_scf, t_forces, dft.niter


print('GPU: %s' % torch.cuda.get_device_name(0))
print('%d CPU cores, %s/%s, level-3 grid, conv_crit %g\n' % (ncores, BASIS, AUXBASIS, CONV))

# Warm up: load the Skala checkpoint, let TorchScript settle, build the CUDA context and pull the
# cached Numba kernels in, so none of that one-time cost shows up in the timings below.
water = Mol(coordfile=os.path.join(EXAMPLES, 'h2o.xyz'))
single_point(water, skala_gpu=False, save_ao_values=True)
single_point(water, skala_gpu=True, save_ao_values=True)
single_point(water, skala_gpu=False, save_ao_values=True, use_gpu=True)
print('warm-up done\n')


if section == 'single':
    print('=' * 100)
    print('Single point: energy + analytical forces')
    print('=' * 100)
    print('%-9s %5s %7s   %-22s %-22s %9s %9s'
          % ('system', 'atoms', 'iter', 'SCF cpu -> gpu (s)', 'forces cpu -> gpu (s)',
             'dE (Ha)', 'max|dF|'))
    print('-' * 100)

    systems = [('water', water, True),
               ('benzene', Mol(atoms=[list(atom) for atom in BENZENE]), True),
               # Caffeine's AO table would be several GB, so that cache stays off for it.
               ('caffeine', Mol(coordfile=os.path.join(HERE, 'Caffeine.xyz')), False)]

    for name, mol, save_ao_values in systems:
        e_cpu, f_cpu, scf_cpu, grad_cpu, iters = single_point(mol, False, save_ao_values)
        e_gpu, f_gpu, scf_gpu, grad_gpu, iters_gpu = single_point(mol, True, save_ao_values)
        print('%-9s %5d %3d/%-3d   %7.1f -> %6.1f (%5.2fx) %7.1f -> %6.1f (%5.2fx) %9.2e %9.2e'
              % (name, mol.natoms, iters, iters_gpu,
                 scf_cpu, scf_gpu, scf_cpu / scf_gpu, grad_cpu, grad_gpu, grad_cpu / grad_gpu,
                 abs(e_cpu - e_gpu), np.abs(f_cpu - f_gpu).max()))

    # The whole SCF on the device, Skala included (Integrals.eval_xc_skala_cupy).
    print()
    print('=' * 100)
    print('Single point: the whole SCF on the GPU (use_gpu=True); forces stay on the CPU')
    print('=' * 100)
    print('%-9s %5s %7s   %-28s %10s' % ('system', 'atoms', 'iter', 'SCF cpu -> gpu (s)', 'dE (Ha)'))
    print('-' * 100)

    for name, mol, save_ao_values in systems:
        e_cpu, _, scf_cpu, _, iters = single_point(mol, False, save_ao_values)
        e_gpu, _, scf_gpu, _, iters_gpu = single_point(mol, False, save_ao_values, use_gpu=True)
        print('%-9s %5d %3d/%-3d   %9.1f -> %7.1f (%5.2fx) %10.2e'
              % (name, mol.natoms, iters, iters_gpu, scf_cpu, scf_gpu, scf_cpu / scf_gpu,
                 abs(e_cpu - e_gpu)))


if section == 'opt':
    # Both calculator modes, because they differ by more than the GPU does. PyFockCalculator runs every step
    # in a fresh subprocess by default, which redoes the cold start -- the imports, loading the Skala
    # checkpoint, warming TorchScript up -- that a long-lived process pays once. run_in_process=True keeps
    # all of it, and the SCF then converges from the previous step's density in well under a second.
    from ase.io import read
    from ase.optimize import LBFGSLineSearch
    from pyfock import PyFockCalculator

    print()
    print('=' * 100)
    print('Geometry optimization (ASE LBFGSLineSearch, fmax = 0.02 eV/A, started 3% stretched)')
    print('=' * 100)
    print('%-9s %-12s %7s   %-24s   %11s %11s'
          % ('system', 'mode', 'steps', 'time cpu -> gpu (s)', 'bond cpu', 'bond gpu'))
    print('-' * 100)

    for name, filename, pair in [('water', 'h2o.xyz', (0, 1)), ('ethane', 'Ethane.xyz', (0, 1))]:
        for in_process in (False, True):
            result = {}
            for skala_gpu in (False, True):
                atoms = read(os.path.join(EXAMPLES, filename))
                atoms.positions *= 1.03
                atoms.calc = PyFockCalculator(
                    functional='skala-1.1', basis=BASIS, auxbasis=AUXBASIS, ncores=ncores,
                    conv_crit=CONV, save_ao_values=True, skala_gpu=skala_gpu, dispersion=True,
                    dispersion_kwargs={'xc': 'b3lyp5'}, run_in_process=in_process,
                    directory='opt_%s_%s_%s' % (name, 'inproc' if in_process else 'subproc',
                                                'gpu' if skala_gpu else 'cpu'))
                optimizer = LBFGSLineSearch(atoms, logfile=None)
                with contextlib.redirect_stdout(io.StringIO()):
                    start = time.perf_counter()
                    optimizer.run(fmax=0.02)
                    elapsed = time.perf_counter() - start
                result[skala_gpu] = (elapsed, atoms.get_distance(*pair), optimizer.get_number_of_steps())
            (t_cpu, d_cpu, steps), (t_gpu, d_gpu, steps_gpu) = result[False], result[True]
            print('%-9s %-12s %3d/%-3d %9.1f -> %6.1f (%5.2fx)   %9.4f A %9.4f A'
                  % (name, 'in-process' if in_process else 'subprocess', steps, steps_gpu,
                     t_cpu, t_gpu, t_cpu / t_gpu, d_cpu, d_gpu))

    print()
    print('The in-process rows start warm, since the warm-up above already ran; a cold start adds about')
    print('6 s once. The subprocess rows pay that warm-up on every step and cannot avoid it.')
