"""Compare PyFock's XC grid schemes with PySCF's grids at a fixed converged density.

For a molecule (xyz file in this directory) the script converges a PySCF DF-RKS calculation once and
then evaluates, on every grid, the electron count and the XC energy of that density. The errors are
taken against a fine PySCF grid (level 8 by default), so the table separates the grid error from
everything else in an SCF. It reports for each grid the number of points before and after PyFock's
density pruning (|rho * w| >= 1e-11), the errors and the generation time.

Usage:
    python3 benchmark_grids.py [xyz_name] [--levels 1,2,3,4,5] [--schemes treutler,numgrid-compact,numgrid-dense]
                               [--basis def2-SVP] [--xc PBE] [--reference-level 8] [--ncores 4]

The default molecule is Decane_C10H22.
"""
import argparse
import os
import sys
from timeit import default_timer as timer

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('xyz', nargs='?', default='Decane_C10H22', help='name of the xyz file in benchmarks_tests (without .xyz)')
parser.add_argument('--levels', default='1,2,3,4,5', help='grid levels to compare')
parser.add_argument('--schemes', default='treutler,numgrid-compact,numgrid-dense', help="PyFock grid schemes to compare: 'treutler' (default scheme), 'numgrid-compact', 'numgrid-dense'")
parser.add_argument('--basis', default='def2-SVP')
parser.add_argument('--auxbasis', default='def2-universal-jfit')
parser.add_argument('--xc', default='PBE')
parser.add_argument('--reference-level', type=int, default=8)
parser.add_argument('--ncores', type=int, default=4)
args = parser.parse_args()

os.environ['OMP_NUM_THREADS'] = str(args.ncores)
os.environ['RAYON_NUM_THREADS'] = str(args.ncores)
import numpy as np
from pyscf import gto, dft
from pyscf.dft import numint, gen_grid
import pyscf
pyscf.lib.num_threads(args.ncores)
from pyfock import Mol, Basis, Grids

here = os.path.dirname(os.path.abspath(__file__))
xyz = os.path.join(here, args.xyz + '.xyz')
levels = [int(x) for x in args.levels.split(',')]
schemes = [x.strip() for x in args.schemes.split(',')]

# ---- converged density (PySCF, DF-RKS)
molP = gto.Mole()
molP.atom = xyz
molP.basis = args.basis
molP.ecp = args.basis
molP.verbose = 0
molP.build()
mf = dft.RKS(molP).density_fit(auxbasis=args.auxbasis)
mf.xc = args.xc
mf.grids.level = 3
mf.conv_tol = 1e-9
mf.kernel()
dm = mf.make_rdm1()
xctype = dft.libxc.xc_type(args.xc)
print('%s: %d atoms, %d electrons, %s/%s, converged E(PySCF, level 3) = %.10f' % (args.xyz, molP.natm, molP.nelectron, args.xc, args.basis, mf.e_tot))

ni = numint.NumInt()


def integrate(coords, weights, blocksize=40000):
    """Electron count, XC energy and rho*w on a grid for the converged density."""
    nelec = 0.0
    exc_tot = 0.0
    rho_w = np.empty(weights.shape[0])
    deriv = 0 if xctype == 'LDA' else 1
    for i0 in range(0, coords.shape[0], blocksize):
        c = coords[i0:i0 + blocksize]
        w = weights[i0:i0 + blocksize]
        ao = ni.eval_ao(molP, c, deriv=deriv)
        rho = ni.eval_rho(molP, ao, dm, xctype=xctype)
        exc = ni.eval_xc_eff(args.xc, rho, deriv=0)[0]
        rho0 = rho if xctype == 'LDA' else rho[0]
        nelec += np.dot(rho0, w)
        exc_tot += np.dot(rho0 * w, exc)
        rho_w[i0:i0 + blocksize] = rho0 * w
    return nelec, exc_tot, rho_w


def pyscf_grid(level):
    g = gen_grid.Grids(molP)
    g.level = level
    g.alignment = 0
    t = timer()
    g.build(sort_grids=False)
    return g.coords, g.weights, timer() - t


coords, weights, _ = pyscf_grid(args.reference_level)
n_ref, exc_ref, _ = integrate(coords, weights)
print('reference: PySCF level %d, %d points, Nelec = %.9f, Exc = %.9f\n' % (args.reference_level, weights.shape[0], n_ref, exc_ref))
print('%-34s %10s %10s %11s %11s %8s' % ('grid', 'points', 'pruned', 'dNelec', 'dExc (Ha)', 'time (s)'))

mol = Mol(coordfile=xyz)
for level in levels:
    coords, weights, dt = pyscf_grid(level)
    n, exc, rho_w = integrate(coords, weights)
    print('%-34s %10d %10d %11.2e %11.2e %8.2f' % ('PySCF level %d' % level, weights.shape[0], (np.abs(rho_w) >= 1e-11).sum(), n - n_ref, exc - exc_ref, dt))
    for scheme in schemes:
        if scheme == 'numgrid-dense' and not 3 <= level <= 8:
            continue
        kwargs = {'scheme': 'treutler'} if scheme == 'treutler' else {'scheme': 'numgrid', 'preset': scheme.split('-')[1]}
        t = timer()
        g = Grids(mol, level=level, ncores=args.ncores, verbose=False, **kwargs)
        dt = timer() - t
        n, exc, rho_w = integrate(g.coords, g.weights)
        label = 'PyFock %s level %d' % (scheme, level)
        print('%-34s %10d %10d %11.2e %11.2e %8.2f' % (label, g.size, (np.abs(rho_w) >= 1e-11).sum(), n - n_ref, exc - exc_ref, dt))
    print()
