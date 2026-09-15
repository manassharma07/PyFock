"""DF_algo=12 (multipole-accelerated DF-J) against DF_algo=11 and PySCF: speed and memory.

python benchmarks_tests/benchmark_DF_algo12_scf.py [--molecules H2O Benzene ...]
       [--functionals pbe r2scan] [--algos 11 12] [--pyscf] [--ncores 4]
       [--basis def2-SVP] [--max-iter 3] [--save-ao] [--json results.json]

Each calculation runs in its own subprocess, one at a time, so the peak resident memory
reported for it is its own. The SCF is cut off after `--max-iter` iterations (3 by default):
both algorithms then do exactly the same work from the same starting density, which keeps the
times and the energy difference comparable while keeping the memory footprint and the runtime
of the large cases modest. `save_ao_values` is off by default for the same reason; pass
--save-ao to turn it back on.

PyFock settings: def2-universal-jfit, SAO, native level-3 grids, conv_crit 1e-7, the default
(SANO) starting density. PySCF DF-RKS uses the settings of benchmark_DFT_LDA_DF.py:
def2-universal-jfit, MINAO guess, grids.level 3, conv_tol 1e-7, direct_scf off.

Reported per calculation: the three-center build time, the per-iteration Coulomb work, the
total wall time, the stored three-center integrals, the far-field multipole storage and the
peak resident memory of the process. Geometries come from benchmarks_tests/<name>.xyz.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

try:
    import resource          # POSIX only
except ImportError:
    resource = None
try:
    import ctypes            # Windows peak working set
except ImportError:
    ctypes = None

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FUNCTIONALS = {'pbe': ([101, 130], '101,130'), 'r2scan': ([497, 498], '497,498'), 'lda': ([1, 7], '1,7')}


class _PROCESS_MEMORY_COUNTERS(ctypes.Structure if ctypes else object):
    _fields_ = [('cb', ctypes.c_uint32), ('PageFaultCount', ctypes.c_uint32),
                ('PeakWorkingSetSize', ctypes.c_size_t), ('WorkingSetSize', ctypes.c_size_t),
                ('QuotaPeakPagedPoolUsage', ctypes.c_size_t), ('QuotaPagedPoolUsage', ctypes.c_size_t),
                ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t), ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
                ('PagefileUsage', ctypes.c_size_t), ('PeakPagefileUsage', ctypes.c_size_t)] if ctypes else []


def peak_rss_gb():
    """Peak resident set size of this process in GB, on POSIX and on Windows."""
    if resource is not None:
        raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return raw / 1e9 if sys.platform == 'darwin' else raw / 1e6
    if ctypes is not None and sys.platform == 'win32':
        counters = _PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(counters)
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        if ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
            return counters.PeakWorkingSetSize / 1e9
    return float('nan')


def worker_pyfock(args):
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS',
                 'NUMEXPR_NUM_THREADS', 'NUMBA_NUM_THREADS'):
        os.environ[name] = str(args.ncores)
    sys.path.insert(0, str(ROOT))
    import contextlib
    import io
    import re
    from pyfock import Basis, DFT, Mol
    mol = Mol(coordfile=args.xyz)
    basis = Basis(mol, {'all': Basis.load(mol=mol, basis_name=args.basis)})
    aux = Basis(mol, {'all': Basis.load(mol=mol, basis_name='def2-universal-jfit')})
    dft = DFT(mol, basis, aux, xc=FUNCTIONALS[args.functional][0],
              save_ao_values=args.save_ao, ncores=args.ncores)
    dft.DF_algo = args.algo
    dft.sao = True
    dft.conv_crit = 1e-7
    dft.max_itr = args.max_iter
    if args.options:
        dft.multipole_options = json.loads(args.options)
    buf = io.StringIO()
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(buf):
        E, _ = dft.scf()
    wall = time.perf_counter() - t0
    out = buf.getvalue()
    if args.verbose:
        sys.stdout.write(out)
    prof = {}
    lines = out.splitlines()
    for i, line in enumerate(lines):
        if line.startswith('Profiling (Wall times'):
            for l2 in lines[i + 2:i + 40]:
                m = re.match(r'^(\s*[A-Za-z].*?)\s{2,}([-\d.]+)\s*$', l2)
                if m:
                    prof[m.group(1).strip()] = float(m.group(2))

    def grab(pattern):
        m = re.search(pattern, out)
        return float(m.group(1)) if m else 0.0

    ints_gb = grab(r'Three Center Two electron ERI \(cached (?:shell-pair|near-field) blocks\) size in GB\s+([\d.e+-]+)')
    t_2c2e = grab(r'Time taken for two-centered two-electron integrals ([\d.]+) seconds')
    t_3c2e = grab(r'Time taken for the (?:shell-blocked three-center integrals|near-field three-center integrals and far-field moments):\s+([\d.]+)')
    far_gb = grab(r'Far-field multipole moments size in GB\s+([\d.e+-]+)')
    metric_gb = grab(r'Two Center Two electron ERI size in GB\s+([\d.e+-]+)')
    summary = next((l for l in lines if l.startswith('DF_algo=12:')), '')
    ff = {}
    m = re.search(r'far field: ([\d.]+)% of the (\d+) significant \(ij\|P\) and ([\d.]+)% of their Rys work', summary)
    if m:
        ff = dict(elements_pct=float(m.group(1)), n_significant=int(m.group(2)), work_pct=float(m.group(3)))
    m = re.search(r'\((\d+) branches in (\d+) boxes', summary)
    if m:
        ff.update(branches=int(m.group(1)), boxes=int(m.group(2)))
    res = dict(code='pyfock', xyz=Path(args.xyz).name, functional=args.functional, algo=args.algo, E=float(E),
               converged=bool(dft.converged), niter=int(dft.niter), max_iter=args.max_iter, wall=wall,
               nao=basis.bfs_nao, naux=aux.bfs_nao, prof=prof, far_field=ff,
               ints_gb=ints_gb, far_gb=far_gb, metric_gb=metric_gb, peak_rss_gb=peak_rss_gb(),
               t_2c2e=t_2c2e, t_3c2e=t_3c2e)
    print('RESULT ' + json.dumps(res), flush=True)


def worker_pyscf(args):
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS',
                 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = str(args.ncores)
    os.environ['PYSCF_MAX_MEMORY'] = '8000'
    from pyscf import dft, gto, lib
    lib.num_threads(args.ncores)
    mol = gto.Mole()
    mol.atom = args.xyz
    mol.basis = args.basis
    mol.ecp = args.basis
    mol.cart = False
    mol.verbose = 4 if args.verbose else 0
    mol.max_memory = 8000
    mol.build()
    t0 = time.perf_counter()
    mf = dft.rks.RKS(mol).density_fit(auxbasis='def2-universal-jfit')
    mf.xc = FUNCTIONALS[args.functional][1]
    mf.direct_scf = False
    mf.init_guess = 'minao'
    dm0 = mf.init_guess_by_minao(mol)
    mf.max_cycle = args.max_iter
    mf.conv_tol = 1e-7
    mf.grids.level = 3
    E = mf.kernel(dm0=dm0)
    wall = time.perf_counter() - t0
    nao = mol.nao_nr()
    naux = int(mf.with_df.auxmol.nao_nr()) if getattr(mf.with_df, 'auxmol', None) is not None else 0
    res = dict(code='pyscf', xyz=Path(args.xyz).name, functional=args.functional, algo='pyscf', E=float(E),
               converged=bool(mf.converged), niter=-1, max_iter=args.max_iter, wall=wall, nao=nao, naux=naux,
               prof={}, far_field={},
               # PySCF stores the fitted three-center tensor as a lower-triangular _cderi block
               ints_gb=naux * nao * (nao + 1) / 2 * 8 / 1e9, far_gb=0.0, metric_gb=0.0,
               peak_rss_gb=peak_rss_gb())
    print('RESULT ' + json.dumps(res), flush=True)


def run_subprocess(cmd):
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          encoding='utf-8', errors='replace')
    for line in proc.stdout.splitlines()[::-1]:
        if line.startswith('RESULT '):
            return json.loads(line[7:])
    sys.stderr.write(proc.stdout[-3000:] + '\n' + proc.stderr[-3000:] + '\n')
    return None


def build_time(res):
    """Everything before the SCF loop: 2c2e metric, Schwarz diagonal, near field, moments."""
    return res.get('prof', {}).get('Coulomb Integrals (2c2e + 3c2e)', 0.0)


def three_center_time(res):
    """Only the three-center part, which is what the far field replaces (the 2c2e metric is common)."""
    return res.get('t_3c2e', 0.0)


def iter_time(res):
    """Per-iteration Coulomb work summed over the iterations that ran."""
    p = res.get('prof', {})
    return p.get('DF (gamma)', 0.0) + p.get('DF (coeff)', 0.0) + p.get('DF (Jtri)', 0.0)


def df_memory(res):
    """Storage the Coulomb term needs: three-center integrals plus far-field moments."""
    return res['ints_gb'] + res['far_gb']


def main():
    # PyFock prints a logo with block characters; keep redirected logs from failing on Windows code pages.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(errors='replace')
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--molecules', nargs='+', default=['H2O', 'Benzene', 'Caffeine', 'Serotonin', 'Cholesterol'])
    parser.add_argument('--functionals', nargs='+', default=['pbe'], choices=sorted(FUNCTIONALS))
    parser.add_argument('--algos', nargs='+', type=int, default=[11, 12])
    parser.add_argument('--pyscf', action='store_true', help='also run PySCF DF-RKS with the benchmark settings')
    parser.add_argument('--ncores', type=int, default=4)
    parser.add_argument('--basis', default='def2-SVP')
    parser.add_argument('--max-iter', type=int, default=3, help='SCF iterations to run (timing/memory benchmark)')
    parser.add_argument('--repeat', type=int, default=1,
                        help='run each calculation this many times and keep the fastest (this machine varies '
                             'by up to a factor of two between runs of identical work)')
    parser.add_argument('--save-ao', action='store_true', help='cache the AO values on the grid (needs several GB)')
    parser.add_argument('--options', default='', help='JSON multipole_options for DF_algo=12')
    parser.add_argument('--json', default='', help='write all results to this file')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--worker', choices=['pyfock', 'pyscf'], help=argparse.SUPPRESS)
    parser.add_argument('--xyz', help=argparse.SUPPRESS)
    parser.add_argument('--functional', help=argparse.SUPPRESS)
    parser.add_argument('--algo', type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker == 'pyfock':
        return worker_pyfock(args)
    if args.worker == 'pyscf':
        return worker_pyscf(args)

    base = [sys.executable, str(Path(__file__).resolve()), '--ncores', str(args.ncores), '--basis', args.basis,
            '--max-iter', str(args.max_iter)]
    if args.save_ao:
        base += ['--save-ao']
    if args.options:
        base += ['--options', args.options]
    # warm the Numba compilation cache on the smallest system first
    for algo in args.algos:
        run_subprocess(base + ['--worker', 'pyfock', '--xyz', str(HERE / 'H2O.xyz'),
                               '--functional', args.functionals[0], '--algo', str(algo)])
    results = []
    for functional in args.functionals:
        for name in args.molecules:
            xyz = str(HERE / (name + '.xyz'))
            for algo in args.algos:
                runs = [run_subprocess(base + ['--worker', 'pyfock', '--xyz', xyz,
                                               '--functional', functional, '--algo', str(algo)])
                        for _ in range(args.repeat)]
                runs = [r for r in runs if r]
                # the fastest run is the least contaminated by whatever else the machine was doing
                res = min(runs, key=lambda r: r['wall']) if runs else None
                if res:
                    results.append(res)
                    extra = ''
                    if algo == 12:
                        extra = '  far field %.1f%% int / %.1f%% work' % (
                            res['far_field'].get('elements_pct', 0), res['far_field'].get('work_pct', 0))
                    print('done: %-18s %-7s DF_algo=%d  2c2e %.2f s  3c2e %.2f s  %d iters %.2f s  '
                          'total %.2f s  DF memory %.2f GB  peak %.2f GB%s' % (
                              name, functional, algo, res.get('t_2c2e', 0.0), three_center_time(res),
                              res['niter'], iter_time(res), res['wall'], df_memory(res),
                              res['peak_rss_gb'], extra), flush=True)
            if args.pyscf:
                runs = [run_subprocess(base + ['--worker', 'pyscf', '--xyz', xyz, '--functional', functional])
                        for _ in range(args.repeat)]
                runs = [r for r in runs if r]
                res = min(runs, key=lambda r: r['wall']) if runs else None
                if res:
                    results.append(res)
                    print('done: %-18s %-7s PySCF       total %.2f s  DF memory %.2f GB  peak %.2f GB' % (
                        name, functional, res['wall'], df_memory(res), res['peak_rss_gb']), flush=True)
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=1))
    print_table(results, args)


def print_table(results, args):
    def find(name, functional, algo):
        return next((r for r in results if r['xyz'] == name + '.xyz' and r['functional'] == functional
                     and r['algo'] == algo), None)

    for functional in args.functionals:
        print('\n%s / %s / def2-universal-jfit, %d cores, %d SCF iterations, save_ao_values=%s, '
              'fastest of %d run(s)'
              % (functional.upper(), args.basis, args.ncores, args.max_iter, args.save_ao, args.repeat))
        print('3c2e+J = three-center build + all per-iteration gamma/J work (the part the far field replaces); '
              'the two-center metric is a common fixed cost, listed separately. '
              'DF memory = stored three-center integrals + far-field moments.')
        head = ('| Molecule | nao/naux | far field int./work | 3c2e+J 11 | 3c2e+J 12 | speed-up '
                '| 2c2e metric | DF memory 11 | DF memory 12 | saved | peak RSS 11 | peak RSS 12 | E(12)-E(11)')
        if args.pyscf:
            head += ' | PySCF total | PySCF/12 | PySCF DF memory'
        print(head + ' |')
        print('|' + '---|' * (head.count('|')))
        for name in args.molecules:
            r11, r12, rp = find(name, functional, 11), find(name, functional, 12), find(name, functional, 'pyscf')
            if r11 is None or r12 is None:
                continue
            c11 = three_center_time(r11) + iter_time(r11)
            c12 = three_center_time(r12) + iter_time(r12)
            m11, m12 = df_memory(r11), df_memory(r12)
            row = ('| %s | %d/%d | %.1f%% / %.1f%% | %.2f | %.2f | %.2fx | %.2f | %.3f | %.3f | %.0f%% '
                   '| %.2f | %.2f | %+.1e'
                   % (name, r11['nao'], r11['naux'], r12['far_field'].get('elements_pct', 0),
                      r12['far_field'].get('work_pct', 0), c11, c12, c11 / max(c12, 1e-9),
                      r11.get('t_2c2e', 0.0), m11, m12,
                      100 * (1 - m12 / max(m11, 1e-12)), r11['peak_rss_gb'], r12['peak_rss_gb'],
                      r12['E'] - r11['E']))
            if rp is not None:
                row += ' | %.2f | %.2fx | %.3f' % (rp['wall'], rp['wall'] / r12['wall'], df_memory(rp))
            print(row + ' |')


if __name__ == '__main__':
    main()
