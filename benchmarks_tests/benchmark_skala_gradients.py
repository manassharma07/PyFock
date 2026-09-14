"""Analytical Skala nuclear gradients against numerical ones.

PyFock's analytical XC gradients are evaluated on a fixed grid: the dependence of the Becke partitioning
weights on the nuclear positions is not differentiated. That is a deliberate approximation, made for
every functional (see :func:`pyfock.Integrals.eval_xc_grad_2`, which follows PySCF's default), and it is
the only term missing from the Skala gradient as well.

So the question this script answers is not "is the analytical gradient right" in the abstract, but "is
Skala's analytical gradient as good as PyFock's already-accepted semilocal ones". It therefore runs a
semilocal functional alongside Skala as a control: whatever discrepancy r2SCAN shows against a numerical
gradient is the baseline cost of the fixed-grid approximation, and Skala should not be meaningfully
worse. Timings are reported too, since numerical gradients cost 6N SCF calculations and the whole point
of the analytical route is that it does not.

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

import numpy as np
from timeit import default_timer as timer

from pyfock import Basis, DFT, DFT_Grad, DFT_NumGrad, Grids, Mol


def run(xc, mol, basis, auxbasis, level, conv_crit, numerical):
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

    result = {'analytical': np.asarray(analytical['gradient']), 't_analytical': t_analytical,
              't_cold': t_cold,
              'xc_component': np.asarray(analytical['gradient_components']['xc'])}
    if numerical:
        start = timer()
        result['numerical'] = np.asarray(
            DFT_NumGrad(dft, verbose=False).calculate()['gradient'])
        result['t_numerical'] = timer() - start
    return result


def report(label, result):
    if 'numerical' not in result:
        return None
    difference = result['analytical'] - result['numerical']
    max_abs = float(np.abs(difference).max())
    rms = float(np.sqrt(np.mean(difference ** 2)))
    scale = float(np.abs(result['numerical']).max())
    print('%-12s max |diff| %9.2e   rms %9.2e   (largest force %7.2e)  '
          'analytical %6.2f s (%5.1f s cold)   numerical %7.2f s   speed-up %6.1fx'
          % (label, max_abs, rms, scale, result['t_analytical'], result['t_cold'],
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

    summary = {}
    for xc in args.functionals:
        result = run(xc, mol, basis, auxbasis, args.level, args.conv_crit, not args.no_numerical)
        summary[xc] = report(xc, result)
        if summary.get(xc) is None:
            print('%-12s analytical gradient in %.2f s (numerical skipped)' % (xc, result['t_analytical']))

    control = next((f for f in args.functionals if not f.lower().startswith('skala')), None)
    skala = next((f for f in args.functionals if f.lower().startswith('skala')), None)
    if control and skala and summary.get(control) and summary.get(skala):
        print('-' * 118)
        print('The %s row is the baseline: PyFock neglects the grid-weight response for every'
              % control)
        print('functional, so that is what the fixed-grid approximation costs on this system.')
        ratio = summary[skala][0] / max(summary[control][0], 1e-30)
        print('Skala max |diff| is %.2fx the %s baseline.' % (ratio, control))
    return 0


if __name__ == '__main__':
    sys.exit(main())
