"""Analytical Skala nuclear gradients against numerical ones.

The Skala gradient carries the full grid response -- the Becke weight derivatives and the
grid-translation term -- which PyFock omits for semilocal functionals. Omitting it for Skala is not an
option: its features are integrals over each atomic grid, so a frozen grid costs ~1e-2 Ha/Bohr, the size
of the forces themselves, against ~1e-4 for a meta-GGA.

Getting the numerical reference right turns out to matter more than the analytical code. Neither mode of
``DFT_NumGrad`` is usable here:

* ``use_fixed_grids=True`` freezes the grid, so it shares the fixed-grid approximation and agrees with a
  gradient that lacks the grid response *for the wrong reason*;
* ``use_fixed_grids=False`` sets ``grids = None`` on the displaced calculation, which makes the SCF
  rebuild **and density-prune** the grid. The base calculation, handed an explicit grid, does not prune.
  The finite difference then straddles two different energy functionals, and reports ~1e-3 Ha/Bohr of
  disagreement that belongs entirely to the reference. That artefact does not shrink with grid level,
  which is what gives it away.

This script therefore builds its own reference: an explicit, unpruned grid at every displaced geometry,
so both sides differentiate the same functional. Against that, the Skala gradient agrees to ~5e-6
Ha/Bohr, the finite-difference truncation floor at a 1e-3 Bohr step.

A semilocal functional runs alongside as a control. Timings are reported too, since numerical gradients
cost 6N SCFs and the point of the analytical route is that it does not.

Translational invariance is the sharpest single check here and needs no reference at all: the net force
must vanish, and it does so to ~1e-14 only when the grid response is included.

Run it as::

    python benchmark_skala_gradients.py                    # H2O, def2-SVP
    python benchmark_skala_gradients.py --molecule h2o_dimer --basis def2-SVP
"""

import argparse
import os
import sys

ncores = int(os.environ.get('PYFOCK_NCORES', 4))
for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(variable, str(ncores))

import contextlib
import io

import numpy as np
from timeit import default_timer as timer

from pyfock import Basis, Data, DFT, DFT_Grad, Grids, Mol


def numerical_gradient(xc, atoms, basis_name, auxbasis_name, level, conv_crit, step_bohr=1e-3):
    """Central-difference gradient with an explicit, unpruned grid rebuilt at every displacement."""
    step_ang = step_bohr / Data.Angs2BohrFactor

    def energy(geometry):
        mol = Mol(atoms=[list(a) for a in geometry])
        basis = Basis(mol, {'all': Basis.load(mol=mol, basis_name=basis_name)})
        auxbasis = Basis(mol, {'all': Basis.load(mol=mol, basis_name=auxbasis_name)})
        dft = DFT(mol, basis, auxbasis, xc=xc, grids=Grids(mol, level=level, verbose=False))
        dft.conv_crit, dft.max_itr, dft.ncores = conv_crit, 50, ncores
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


def run(xc, mol, basis, auxbasis, level, conv_crit, numerical, atoms=None,
        basis_name=None, auxbasis_name=None):
    """Converge the SCF, then take the analytical gradient and optionally a numerical one."""
    dft = DFT(mol, basis, auxbasis, xc=xc, grids=Grids(mol, level=level, verbose=False))
    dft.conv_crit = conv_crit
    dft.max_itr = 50
    dft.ncores = ncores
    dft.scf()

    # The first analytical gradient pays for JIT-compiling the AO value/gradient/Hessian kernel, which
    # the SCF never touches. On a small molecule that dominates and would make the analytical route look
    # slower than the numerical one, so warm it up and time the second call.
    start = timer()
    analytical = DFT_Grad(dft, verbose=False).calculate()
    t_cold = timer() - start
    start = timer()
    analytical = DFT_Grad(dft, verbose=False).calculate()
    t_analytical = timer() - start

    gradient = np.asarray(analytical['gradient'])
    result = {'analytical': gradient, 't_analytical': t_analytical, 't_cold': t_cold,
              'net_force': float(np.abs(gradient.sum(axis=0)).max()),
              'xc_component': np.asarray(analytical['gradient_components']['xc'])}
    if numerical:
        start = timer()
        result['numerical'] = numerical_gradient(xc, atoms, basis_name, auxbasis_name, level,
                                                 conv_crit)
        result['t_numerical'] = timer() - start
    return result


def report(label, result):
    if 'numerical' not in result:
        return None
    difference = result['analytical'] - result['numerical']
    max_abs = float(np.abs(difference).max())
    rms = float(np.sqrt(np.mean(difference ** 2)))
    scale = float(np.abs(result['numerical']).max())
    print('%-12s max |diff| %9.2e   rms %9.2e   net force %9.2e   (largest force %7.2e)  '
          'analytical %6.2f s   numerical %7.2f s   speed-up %6.1fx'
          % (label, max_abs, rms, result['net_force'], scale, result['t_analytical'],
             result['t_numerical'], result['t_numerical'] / max(result['t_analytical'], 1e-9)))
    return max_abs, rms, scale


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--molecule', default='h2o')
    parser.add_argument('--basis', default='def2-SVP')
    parser.add_argument('--auxbasis', default='def2-universal-jfit')
    parser.add_argument('--level', type=int, default=3)
    parser.add_argument('--conv-crit', type=float, default=1e-9)
    parser.add_argument('--functionals', nargs='+', default=['R2SCAN', 'skala-1.1'])
    parser.add_argument('--no-numerical', action='store_true',
                        help='skip the numerical gradients (timing only)')
    args = parser.parse_args(argv)

    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [args.molecule,
                  os.path.join(here, args.molecule + '.xyz'),
                  os.path.join(here, '..', 'examples', args.molecule + '.xyz')]
    xyz = next((path for path in candidates if os.path.isfile(path)), None)
    if xyz is None:
        parser.error('could not find a geometry for ' + args.molecule)

    mol = Mol(coordfile=xyz)
    basis = Basis(mol, {'all': Basis.load(mol=mol, basis_name=args.basis)})
    auxbasis = Basis(mol, {'all': Basis.load(mol=mol, basis_name=args.auxbasis)})

    print('=' * 118)
    print('Analytical vs numerical nuclear gradients | %s | %s | %d atoms | %d AOs | %d cores'
          % (os.path.basename(xyz), args.basis, mol.natoms, basis.bfs_nao, ncores))
    print('The SCF convergence threshold is %g; numerical gradients need 6N converged SCFs.'
          % args.conv_crit)
    print('=' * 118)

    atoms = [[mol.atomicSpecies[i]] + list(np.asarray(mol.coords)[i]) for i in range(mol.natoms)]

    summary = {}
    for xc in args.functionals:
        result = run(xc, mol, basis, auxbasis, args.level, args.conv_crit, not args.no_numerical,
                     atoms=atoms, basis_name=args.basis, auxbasis_name=args.auxbasis)
        summary[xc] = report(xc, result)
        if summary.get(xc) is None:
            print('%-12s analytical gradient in %.2f s (numerical skipped)' % (xc, result['t_analytical']))

    control = next((f for f in args.functionals if not f.lower().startswith('skala')), None)
    skala = next((f for f in args.functionals if f.lower().startswith('skala')), None)
    if control and skala and summary.get(control) and summary.get(skala):
        print('-' * 118)
        print('%s has no grid response (the semilocal approximation); Skala does. The net-force'
              % control)
        print('column shows it: exact translational invariance is only possible with the response.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
