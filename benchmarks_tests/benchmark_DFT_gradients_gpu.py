"""GPU analytical nuclear gradients against the CPU ones: accuracy and speed.

Every term of ``DFT_Grad`` except the ECP one now has a device implementation, and ``use_gpu``
selects between them. The CPU routines are validated elsewhere against PySCF and finite differences
(``benchmark_DFT_analytical_gradients.py``, ``benchmark_skala_gradients.py``), so the question this
script answers is whether the device port reproduces them and what it buys.

Both gradients are taken from the *same* converged SCF, so the timings and the differences are those
of the gradient alone -- nothing here is contaminated by the SCF having run differently.

Three accuracy numbers are reported:

* ``max |GPU-CPU|`` -- the port check. It should sit at the round-off floor of the contraction, a few
  1e-12 Ha/Bohr, not at the size of any physical term.
* ``net force`` -- translational invariance, which needs no reference at all. With the fixed-grid
  approximation (the default for semilocal functionals, as in PySCF) it is ~1e-4 Ha/Bohr and is a
  property of the approximation, not of the implementation; with ``--grid-response`` it drops to
  ~1e-13 and any deviation is a real bug. It must come out the same on both devices.
* ``max |analytical-numerical|`` -- only with ``--numerical``. The reference rebuilds an explicit,
  unpruned grid at every displaced geometry, so it differentiates the moving grid and is only
  comparable to a gradient that carries the grid response; ``--numerical`` therefore implies
  ``--grid-response``. It costs 6N converged SCFs, so it is meant for the small cases.

Run it as::

    python benchmark_DFT_gradients_gpu.py
    python benchmark_DFT_gradients_gpu.py --molecules H2O Benzene --basis def2-TZVP --xc PBE
    python benchmark_DFT_gradients_gpu.py --molecules H2O --numerical
    python benchmark_DFT_gradients_gpu.py --xc skala-1.1
"""

import argparse
import contextlib
import io
import os
import sys

ncores = int(os.environ.get('PYFOCK_NCORES', 8))
for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(variable, str(ncores))

import numpy as np
from timeit import default_timer as timer

from pyfock import Basis, Data, DFT, DFT_Grad, Grids, Mol

DEFAULT_MOLECULES = ['H2O', 'Benzene', 'Caffeine', 'Serotonin']
AUX_BASIS = 'def2-universal-jfit'

# Native PyFock functional ids, the same spellings benchmark_DFT_analytical_gradients.py accepts.
FUNCTIONALS = {
    'LDA': [1, 7],
    'PBE': [101, 130],
    'TPSS': [202, 231],
    'R2SCAN': [497, 498],
}


def resolve_xc(name):
    """Map a command line functional name onto what DFT expects."""
    key = name.upper()
    if key in FUNCTIONALS:
        return FUNCTIONALS[key]
    return name           # Skala and anything else DFT resolves itself


def build(xyz, basis_name, level):
    mol = Mol(coordfile=xyz)
    basis = Basis(mol, {'all': Basis.load(mol=mol, basis_name=basis_name)})
    auxbasis = Basis(mol, {'all': Basis.load(mol=mol, basis_name=AUX_BASIS)})
    # Built exactly as numerical_gradient() builds the displaced ones, so the finite difference
    # differentiates the same functional on both sides.
    grids = Grids(mol, level=level, verbose=False)
    return mol, basis, auxbasis, grids


def converged_scf(xc, xyz, basis_name, level, conv_crit, use_gpu):
    mol, basis, auxbasis, grids = build(xyz, basis_name, level)
    dft = DFT(mol, basis, auxbasis, xc=xc, grids=grids)
    dft.conv_crit = conv_crit
    dft.max_itr = 50
    dft.ncores = ncores
    dft.use_gpu = use_gpu
    dft.save_ao_values = True
    dft.isDF = True
    start = timer()
    with contextlib.redirect_stdout(io.StringIO()):
        energy, _ = dft.scf()
    return dft, float(energy), timer() - start


def timed_gradient(dft, use_gpu, grid_response, repeats=1):
    """Warm up, then time the gradient and return the *fastest* run with its own term breakdown.

    Two warm-up calls, not one: the first compiles the CUDA kernels, and the second is still slow
    because CuPy's pool is growing its segments, which on a device that PyTorch is already holding
    several GB of means falling back to cudaMalloc. Returning the fastest run's own timings (rather
    than the last run's) keeps the per-term table consistent with the wall clock beside it.
    """
    kwargs = dict(verbose=False, use_gpu=use_gpu)
    if grid_response is not None:
        kwargs['grid_response'] = grid_response
    with contextlib.redirect_stdout(io.StringIO()):
        DFT_Grad(dft, **kwargs).calculate()
        DFT_Grad(dft, **kwargs).calculate()
        best = None
        result = None
        for _ in range(repeats):
            start = timer()
            candidate = DFT_Grad(dft, **kwargs).calculate()
            elapsed = timer() - start
            if best is None or elapsed < best:
                best, result = elapsed, candidate
    return result, best


def numerical_gradient(xc, mol, basis_name, level, conv_crit, use_gpu, step_bohr=1e-3):
    """Central differences with an explicit, unpruned grid rebuilt at every displacement."""
    step_ang = step_bohr / Data.Angs2BohrFactor
    atoms = [[symbol, *coords] for symbol, coords in zip(mol.atomicSpecies, mol.coords)]

    def energy(geometry):
        displaced = Mol(atoms=[list(a) for a in geometry])
        basis = Basis(displaced, {'all': Basis.load(mol=displaced, basis_name=basis_name)})
        auxbasis = Basis(displaced, {'all': Basis.load(mol=displaced, basis_name=AUX_BASIS)})
        dft = DFT(displaced, basis, auxbasis, xc=xc,
                  grids=Grids(displaced, level=level, verbose=False))
        dft.conv_crit, dft.max_itr, dft.ncores = conv_crit, 50, ncores
        dft.use_gpu = use_gpu
        with contextlib.redirect_stdout(io.StringIO()):
            value, _ = dft.scf()
        return float(value)

    gradient = np.zeros((len(atoms), 3))
    for atom in range(len(atoms)):
        for direction in range(3):
            plus = [list(a) for a in atoms]
            minus = [list(a) for a in atoms]
            plus[atom][1 + direction] += step_ang
            minus[atom][1 + direction] -= step_ang
            gradient[atom, direction] = (energy(plus) - energy(minus)) / (2.0 * step_bohr)
    return gradient


def print_timings(cpu_timings, gpu_timings):
    keys = list(dict.fromkeys(list(gpu_timings) + list(cpu_timings)))
    print('    %-24s %10s %10s %9s' % ('term', 'CPU (s)', 'GPU (s)', 'speed-up'))
    for key in keys:
        cpu_t = cpu_timings.get(key)
        gpu_t = gpu_timings.get(key)
        ratio = ('%8.1fx' % (cpu_t / gpu_t)) if cpu_t and gpu_t else '        -'
        print('    %-24s %10s %10s %9s'
              % (key,
                 '-' if cpu_t is None else '%10.3f' % cpu_t,
                 '-' if gpu_t is None else '%10.3f' % gpu_t,
                 ratio))
    cpu_total = sum(cpu_timings.values())
    gpu_total = sum(gpu_timings.values())
    print('    %-24s %10.3f %10.3f %8.1fx'
          % ('TOTAL', cpu_total, gpu_total, cpu_total / max(gpu_total, 1e-12)))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--molecules', nargs='+', default=DEFAULT_MOLECULES,
                        help='xyz basenames in this directory (default: %(default)s)')
    parser.add_argument('--basis', default='def2-SVP')
    parser.add_argument('--xc', default='PBE',
                        help='LDA, PBE, TPSS, R2SCAN, or a name DFT resolves (e.g. skala-1.1)')
    parser.add_argument('--level', type=int, default=3, help='grid level')
    parser.add_argument('--conv-crit', type=float, default=1e-9)
    parser.add_argument('--repeats', type=int, default=1,
                        help='time this many gradients and keep the fastest. Worth raising for '
                             'Skala: its model call is a PyTorch forward/backward whose cost varies '
                             'by an order of magnitude between identical calls, on either device')
    parser.add_argument('--grid-response', action='store_true',
                        help='differentiate the quadrature grid too (always on for Skala)')
    parser.add_argument('--numerical', action='store_true',
                        help='also take a finite-difference gradient (6N SCFs); implies '
                             '--grid-response')
    parser.add_argument('--scf-on-cpu', action='store_true',
                        help='converge the SCF on the CPU (the gradients are compared either way)')
    args = parser.parse_args(argv)

    if args.numerical:
        args.grid_response = True
    xc = resolve_xc(args.xc)
    grid_response = True if args.grid_response else None    # None: let DFT_Grad pick the default
    scf_gpu = not args.scf_on_cpu

    print('=' * 100)
    print('PyFock analytical nuclear gradients: GPU vs CPU | %s/%s | grid level %d | %d CPU cores'
          % (args.xc, args.basis, args.level, ncores))
    print('=' * 100)

    summary = []
    for name in args.molecules:
        xyz = name if name.endswith('.xyz') else name + '.xyz'
        if not os.path.exists(xyz):
            print('\n%s: no such file, skipped' % xyz)
            continue

        dft, energy, t_scf = converged_scf(xc, xyz, args.basis, args.level, args.conv_crit, scf_gpu)
        mol, basis, auxbasis = dft.mol, dft.basis, dft.auxbasis
        # After a GPU SCF the grid arrays live on the device, so ask for the shape, not the values.
        print('\n%s  |  %d atoms, %d AOs, %d aux AOs, %d grid points'
              % (xyz, mol.natoms, basis.bfs_nao, auxbasis.bfs_nao, dft.grids.coords.shape[0]))
        print('  SCF (%s): %.2f s, E = %.10f Ha' % ('GPU' if scf_gpu else 'CPU', t_scf, energy))

        gpu_result, t_gpu = timed_gradient(dft, True, grid_response, args.repeats)
        cpu_result, t_cpu = timed_gradient(dft, False, grid_response, args.repeats)
        print_timings(cpu_result['timings'], gpu_result['timings'])

        gpu_grad = np.asarray(gpu_result['gradient'])
        cpu_grad = np.asarray(cpu_result['gradient'])
        difference = np.abs(gpu_grad - cpu_grad)
        scale = float(np.abs(cpu_grad).max())
        print('  wall clock        CPU %.3f s   GPU %.3f s   speed-up %.1fx'
              % (t_cpu, t_gpu, t_cpu / max(t_gpu, 1e-12)))
        print('  max |GPU-CPU|     %.3e Ha/Bohr   (rms %.3e, largest force %.3e)'
              % (difference.max(), float(np.sqrt(np.mean((gpu_grad - cpu_grad) ** 2))), scale))
        print('  net force         CPU %.3e   GPU %.3e Ha/Bohr'
              % (float(np.abs(cpu_grad.sum(axis=0)).max()),
                 float(np.abs(gpu_grad.sum(axis=0)).max())))
        # np.max, not the builtin: `max` silently keeps the running value when it meets a NaN, which
        # would hide exactly the kind of failure this line exists to catch.
        per_term = {term: float(np.abs(np.asarray(gpu_result['gradient_components'][term])
                                       - np.asarray(cpu_result['gradient_components'][term])).max())
                    for term in cpu_result['gradient_components']}
        worst_term = max(per_term, key=lambda t: (np.isnan(per_term[t]), per_term[t]))
        print('  worst term        %s, |GPU-CPU| %.3e Ha/Bohr' % (worst_term, per_term[worst_term]))
        if not np.isfinite(gpu_grad).all() or not np.isfinite(cpu_grad).all():
            print('  *** NON-FINITE GRADIENT ***')

        row = dict(name=xyz, natoms=mol.natoms, nbf=basis.bfs_nao, t_cpu=t_cpu, t_gpu=t_gpu,
                   diff=float(difference.max()), scale=scale,
                   net=float(np.abs(gpu_grad.sum(axis=0)).max()), numerical=None)

        if args.numerical:
            start = timer()
            reference = numerical_gradient(xc, mol, args.basis, args.level, args.conv_crit, scf_gpu)
            t_numerical = timer() - start
            row['numerical'] = float(np.abs(gpu_grad - reference).max())
            print('  max |GPU-numerical| %.3e Ha/Bohr   (%d SCFs in %.1f s, %.0fx the GPU gradient)'
                  % (row['numerical'], 6 * mol.natoms, t_numerical, t_numerical / max(t_gpu, 1e-12)))
        summary.append(row)

    if not summary:
        return 1

    print('\n' + '=' * 100)
    print('Summary | %s/%s%s' % (args.xc, args.basis,
                                 ', grid response' if args.grid_response else ', fixed grid'))
    print('=' * 100)
    header = '%-18s %6s %6s %10s %10s %9s %12s %11s' % (
        'molecule', 'atoms', 'AOs', 'CPU (s)', 'GPU (s)', 'speed-up', 'max|GPU-CPU|', 'net force')
    print(header)
    print('-' * len(header))
    for row in summary:
        print('%-18s %6d %6d %10.3f %10.3f %8.1fx %12.2e %11.2e'
              % (row['name'], row['natoms'], row['nbf'], row['t_cpu'], row['t_gpu'],
                 row['t_cpu'] / max(row['t_gpu'], 1e-12), row['diff'], row['net']))
    if any(row['numerical'] is not None for row in summary):
        print('\n%-18s %14s %14s' % ('molecule', 'max|ana-num|', 'largest force'))
        for row in summary:
            if row['numerical'] is not None:
                print('%-18s %14.2e %14.2e' % (row['name'], row['numerical'], row['scale']))
    return 0


if __name__ == '__main__':
    sys.exit(main())
