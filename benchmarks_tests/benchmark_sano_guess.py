"""
Benchmark: SCF convergence from the SANO initial guess versus the core-Hamiltonian guess.

Every case converges the same molecule / basis / functional twice with identical settings
(DF_algo=11, native level-3 grids, energy convergence criterion) and only the initial guess
changed: ``dmat_guess_method='core'`` and ``'sano'``. Reported per case: number of SCF
iterations, wall time of the whole SCF, energy difference, electrons in the SANO guess.

    python3 benchmarks_tests/benchmark_sano_guess.py [--ncores 8] [--conv 1e-7] [--large]
                                                     [--cases H2O,Decane] [--out table.md]
"""
import argparse
import contextlib
import io
import os
import sys
import time

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--ncores', type=int, default=8)
parser.add_argument('--conv', type=float, default=1e-7, help='SCF energy convergence criterion (Hartree)')
parser.add_argument('--large', action='store_true', help='also run the large cases (TPP, Zn-TPP, icosane/TZVP)')
parser.add_argument('--cases', default=None, help='comma-separated list of case labels to run (default: all)')
parser.add_argument('--out', default=None, help='write the markdown table to this file')
args = parser.parse_args()

for var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[var] = str(args.ncores)

import numpy as np
from pyfock import Basis, DFT, Mol

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def find_xyz(name):
    for folder in (HERE, os.path.join(ROOT, 'examples')):
        path = os.path.join(folder, name)
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(name)


# label, xyz file, orbital basis, functional
CASES = [
    ('H2O', 'H2O.xyz', 'def2-SVP', 'PBE'),
    ('H2O', 'H2O.xyz', 'def2-TZVP', 'PBE'),
    ('H2O', 'H2O.xyz', 'def2-QZVP', 'PBE'),
    ('H2O', 'H2O.xyz', 'def2-SVP', 'B3LYP'),
    ('H2O', 'H2O.xyz', 'def2-SVP', 'HF'),
    ('Ethane', 'Ethane.xyz', 'def2-SVP', 'PBE'),
    ('Decane', 'Decane_C10H22.xyz', 'def2-SVP', 'PBE'),
    ('Decane', 'Decane_C10H22.xyz', 'def2-TZVP', 'PBE'),
    ('Decane', 'Decane_C10H22.xyz', 'def2-SVP', 'HF'),
    ('Caffeine', 'Caffeine.xyz', 'def2-SVP', 'PBE'),
    ('Caffeine', 'Caffeine.xyz', 'def2-SVP', 'B3LYP'),
    ('Serotonin', 'Serotonin.xyz', 'def2-SVP', 'PBE'),
    ('Benzene-fulvene dimer', 'Benzene-Fulvene_Dimer.xyz', 'def2-SVP', 'PBE'),
    ('Adenine-thymine', 'Adenine-Thymine.xyz', 'def2-SVP', 'PBE'),
    ('Cholesterol', 'Cholesterol.xyz', 'def2-SVP', 'PBE'),
    ('Zn dimer', 'Zn_dimer.xyz', 'def2-SVP', 'PBE'),
    ('Cd dimer (ECP)', 'Cd_dimer.xyz', 'def2-SVP', 'PBE'),
    ('AgCl (ECP)', 'AgCl.xyz', 'def2-SVP', 'PBE'),
    ('AuCl (ECP)', 'AuCl.xyz', 'def2-SVP', 'PBE'),
    ('HgCl2 (ECP)', 'HgCl2.xyz', 'def2-SVP', 'PBE'),
    ('I2 (ECP)', 'I2.xyz', 'def2-SVP', 'PBE'),
    ('BiH3 (ECP)', 'BiH3.xyz', 'def2-SVP', 'PBE'),
]
LARGE_CASES = [
    ('TPP', 'TPP.xyz', 'def2-SVP', 'PBE'),
    ('Zn-TPP', 'Zn_TPP.xyz', 'def2-SVP', 'PBE'),
    ('Icosane', 'Icosane_C20H42.xyz', 'def2-TZVP', 'PBE'),
]
if args.large:
    CASES += LARGE_CASES
if args.cases:
    wanted = [c.strip().lower() for c in args.cases.split(',')]
    CASES = [c for c in CASES if c[0].lower() in wanted]


def run_scf(mol, basis, aux, xc, guess, conv, ncores):
    dft = DFT(mol, basis, aux, xc=xc, dmat_guess_method=guess, ncores=ncores,
              save_ao_values=basis.bfs_nao <= 400)
    dft.conv_crit = conv
    dft.max_itr = 100
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        energy, _ = dft.scf()
    return {'E': float(energy), 'niter': dft.niter, 'converged': dft.converged,
            'wall': time.perf_counter() - t0, 'info': getattr(dft, 'guess_info', None)}


def run_case(label, xyz, basis_name, xc, conv, ncores):
    mol = Mol(coordfile=find_xyz(xyz))
    basis = Basis(mol, {'all': Basis.load(mol=mol, basis_name=basis_name)})
    aux_name = 'def2-universal-jkfit' if xc in ('HF', 'B3LYP', 'PBE0') else 'def2-universal-jfit'
    aux = Basis(mol, {'all': Basis.load(mol=mol, basis_name=aux_name)})
    core = run_scf(mol, basis, aux, xc, 'core', conv, ncores)
    sano = run_scf(mol, basis, aux, xc, 'sano', conv, ncores)
    return {'label': label, 'basis': basis_name, 'xc': xc, 'nbf': basis.bfs_nao, 'nelec': mol.nelectrons,
            'core': core, 'sano': sano}


def fmt_iters(res):
    return str(res['niter']) if res['converged'] else 'n.c. (%d)' % res['niter']


print('SANO vs core-Hamiltonian initial guess: ncores = %d, conv_crit = %g' % (args.ncores, args.conv), flush=True)
# warm-up so that Numba compilation does not enter the timings of the first case
_ = run_case('warm-up', 'H2O.xyz', 'def2-SVP', 'PBE', 1e-5, args.ncores)

rows = []
for label, xyz, basis_name, xc in CASES:
    t0 = time.perf_counter()
    try:
        r = run_case(label, xyz, basis_name, xc, args.conv, args.ncores)
    except Exception as exc:  # report and continue with the next case
        print('%-24s %-9s %-6s FAILED: %r' % (label, basis_name, xc, exc), flush=True)
        continue
    rows.append(r)
    info = r['sano']['info']
    frac = 100.0 * info['nelectrons_projected'] / info['nelectrons'] if info else float('nan')
    print('%-24s %-9s %-6s nbf=%4d  core: %3s it %7.1f s   sano: %3s it %7.1f s   dE=%+.1e Ha  ANOs represented: %.2f%%  (%.0f s)'
          % (label, basis_name, xc, r['nbf'], fmt_iters(r['core']), r['core']['wall'],
             fmt_iters(r['sano']), r['sano']['wall'], r['sano']['E'] - r['core']['E'], frac,
             time.perf_counter() - t0), flush=True)

lines = ['| System | Basis | XC | N_bf | Iterations core | Iterations SANO | SCF wall core (s) | SCF wall SANO (s) | E(SANO) - E(core) (Ha) | ANOs represented by the basis |',
         '|---|---|---|---|---|---|---|---|---|---|']
tot_core = tot_sano = 0
for r in rows:
    info = r['sano']['info']
    frac = '%.2f %%' % (100.0 * info['nelectrons_projected'] / info['nelectrons']) if info else 'n/a'
    lines.append('| %s | %s | %s | %d | %s | %s | %.1f | %.1f | %+.1e | %s |' % (
        r['label'], r['basis'], r['xc'], r['nbf'], fmt_iters(r['core']), fmt_iters(r['sano']),
        r['core']['wall'], r['sano']['wall'], r['sano']['E'] - r['core']['E'], frac))
    if r['core']['converged'] and r['sano']['converged']:
        tot_core += r['core']['niter']
        tot_sano += r['sano']['niter']
lines.append('')
lines.append('Total SCF iterations over the converged cases: core %d, SANO %d (%.0f %% fewer).'
             % (tot_core, tot_sano, 100.0 * (1 - tot_sano / tot_core) if tot_core else 0.0))
table = '\n'.join(lines)
print('\n' + table)
if args.out:
    with open(args.out, 'w', encoding='utf-8') as f:
        f.write(table + '\n')
    print('\nTable written to', args.out)
