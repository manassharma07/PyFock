"""How expensive is Skala compared with a GGA and a meta-GGA?

Skala is a neural exchange-correlation functional whose cost has a different shape from a semilocal
functional's. Two things are measured separately, because they answer different questions:

1. **One XC evaluation** on a fixed density and grid -- PBE, r2SCAN and Skala are handed exactly the same
   density matrix and the same grid, so the ratio is the pure functional-evaluation overhead, free of any
   difference in how many SCF iterations each functional happens to need. For Skala the three passes
   (density, model, potential) are reported individually.
2. **A full SCF** -- wall time and iteration count, which is what a user actually waits for.

Run it as::

    python benchmark_skala.py                  # H2O, def2-SVP
    python benchmark_skala.py Caffeine def2-SVP

Skala needs PyTorch and the native ('treutler') grids; see the README section "Skala: the neural
exchange-correlation functional".
"""

import os
import sys

ncores = int(os.environ.get('PYFOCK_NCORES', 4))
for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(variable, str(ncores))

import numpy as np
from timeit import default_timer as timer

from pyfock import Basis, DFT, Grids, Integrals, Mol, XC


def bench(xyz_name='H2O', basis_name='def2-SVP', auxbasis_name='def2-universal-jfit', level=3,
          repeats=3, conv_crit=1e-8, max_itr=50):
    here = os.path.dirname(os.path.abspath(__file__))
    xyz = xyz_name if os.path.isfile(xyz_name) else os.path.join(here, xyz_name + '.xyz')

    mol = Mol(coordfile=xyz)
    basis = Basis(mol, {'all': Basis.load(mol=mol, basis_name=basis_name)})
    auxbasis = Basis(mol, {'all': Basis.load(mol=mol, basis_name=auxbasis_name)})
    grids = Grids(mol, level=level, verbose=False)

    print('=' * 78)
    print('%s | %s | %d atoms | %d AOs | %d grid points | %d cores'
          % (os.path.basename(xyz), basis_name, mol.natoms, basis.bfs_nao, grids.size, ncores))
    print('=' * 78)

    # ---------------------------------------------------------------- full SCF runs
    print('\nFull SCF (energy, iterations, wall time)')
    print('-' * 78)
    results = {}
    dmat = None
    for xc in ('PBE', 'R2SCAN', 'skala-1.1'):
        dft = DFT(mol, basis, auxbasis, xc=xc, grids=Grids(mol, level=level, verbose=False))
        dft.conv_crit = conv_crit
        dft.max_itr = max_itr
        dft.ncores = ncores
        start = timer()
        energy, converged_dmat = dft.scf()
        elapsed = timer() - start
        iterations = getattr(dft, 'niter', None)
        results[xc] = (energy, elapsed, iterations)
        if xc == 'PBE':
            dmat = converged_dmat  # a converged, physical density for the per-call timings
        print('%-12s E = %18.10f Ha   %7.2f s   %s iterations'
              % (xc, energy, elapsed, iterations if iterations is not None else '?'))

    # ---------------------------------------------------------------- one XC evaluation each
    print('\nOne XC evaluation on the same density and grid (best of %d)' % repeats)
    print('-' * 78)

    blocksize = 5000
    ngrids = grids.size
    nblocks = ngrids // blocksize
    list_nonzero_indices, count_nonzero_indices = Integrals.bf_val_helpers.nonzero_ao_indices(
        basis, grids.coords, blocksize, nblocks, ngrids)

    def time_it(function):
        best = float('inf')
        value = None
        for _ in range(repeats):
            start = timer()
            value = function()
            best = min(best, timer() - start)
        return best, value

    timings = {}
    for xc in ('PBE', 'R2SCAN'):
        funcid = XC.resolve_functional(xc)
        elapsed, (exc, _) = time_it(lambda funcid=funcid: Integrals.eval_xc_2(
            basis, dmat, grids.weights, grids.coords, funcid, False, ncores=ncores,
            blocksize=blocksize, list_nonzero_indices=list_nonzero_indices,
            count_nonzero_indices=count_nonzero_indices, print_nelec=False))
        timings[xc] = elapsed
        print('%-12s %7.3f s   Exc = %16.10f Ha' % (xc, elapsed, exc))

    skala = XC.load_skala('skala-1.1')
    elapsed, (exc, _) = time_it(lambda: Integrals.eval_xc_skala(
        basis, dmat, grids, skala, ncores=ncores, blocksize=blocksize,
        list_nonzero_indices=list_nonzero_indices, count_nonzero_indices=count_nonzero_indices,
        print_nelec=False))
    timings['skala-1.1'] = elapsed
    print('%-12s %7.3f s   Exc = %16.10f Ha' % ('skala-1.1', elapsed, exc))

    # the three passes separately
    Integrals.eval_xc_skala(basis, dmat, grids, skala, ncores=ncores, blocksize=blocksize,
                            list_nonzero_indices=list_nonzero_indices,
                            count_nonzero_indices=count_nonzero_indices,
                            print_nelec=False, debug=True)

    print('\nSlowdown of one Skala XC evaluation')
    print('-' * 78)
    for xc in ('PBE', 'R2SCAN'):
        print('  vs %-8s %5.1fx' % (xc, timings['skala-1.1'] / timings[xc]))
    print('\nSlowdown of a full SCF')
    print('-' * 78)
    for xc in ('PBE', 'R2SCAN'):
        print('  vs %-8s %5.1fx wall time' % (xc, results['skala-1.1'][1] / results[xc][1]))
    return results, timings


if __name__ == '__main__':
    molecule = sys.argv[1] if len(sys.argv) > 1 else 'H2O'
    basis_set = sys.argv[2] if len(sys.argv) > 2 else 'def2-SVP'
    bench(molecule, basis_set)
