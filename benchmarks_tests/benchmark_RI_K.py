"""
RI-K (exact exchange) timing benchmark on the CPU.

HF / def2-SVP / def2-universal-jkfit, SAO orbital and fit space, density fitting for
J and K (DF_algo=11, the RI-K path), one subprocess per molecule so that every run
starts from a clean process (Numba kernels are cached on disk after the warm-up run).

Driver:   python benchmark_RI_K.py --ncores 8 --tag baseline [--warmup] [--molecules A.xyz B.xyz]
Single:   python benchmark_RI_K.py --single Decane_C10H22.xyz --ncores 8

The driver writes <outdir>/<tag>_<molecule>.log (complete output of the run) and
<outdir>/<tag>.json + <outdir>/<tag>.md (the extracted timings).
"""
import argparse
import json
import os
import re
import subprocess
import sys
from timeit import default_timer as timer

HERE = os.path.dirname(os.path.abspath(__file__))
MOLECULES = ['Benzene.xyz', 'Serotonin.xyz', 'Caffeine.xyz', 'Decane_C10H22.xyz', 'Icosane_C20H42.xyz']
BASIS = 'def2-SVP'
AUXBASIS = 'def2-universal-jkfit'


def set_threads(ncores):
    for var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[var] = str(ncores)


def setup(xyz, ncores, xc='HF', df_algo=None, conv_crit=1e-7):
    """Mol/Basis/DFT objects of one benchmark run (shared with the profiling scripts)."""
    from pyfock import Basis, DFT, Mol
    mol = Mol(coordfile=xyz if os.path.isabs(xyz) else os.path.join(HERE, xyz))
    basis = Basis(mol, {'all': Basis.load(mol=mol, basis_name=BASIS)})
    aux = Basis(mol, {'all': Basis.load(mol=mol, basis_name=AUXBASIS)})
    dft = DFT(mol, basis, aux, xc=xc, conv_crit=conv_crit, ncores=ncores, use_gpu=False)
    dft.sao = True
    dft.max_itr = 50
    dft.threshold_schwarz = 1e-9
    dft.strict_schwarz = False
    dft.cholesky = True
    dft.orthogonalize = True
    if df_algo is not None:
        dft.DF_algo = df_algo
    return mol, basis, aux, dft


def run_single(xyz, ncores, xc, df_algo):
    set_threads(ncores)
    mol, basis, aux, dft = setup(xyz, ncores, xc=xc, df_algo=df_algo)
    import pyfock
    print('BENCH molecule %s natoms %d nao_cart %d naux_cart %d nelec %d ncores %d xc %s'
          % (os.path.basename(xyz), mol.natoms, basis.bfs_nao, aux.bfs_nao, mol.nelectrons, ncores, xc), flush=True)
    print('BENCH code %s' % os.path.dirname(os.path.dirname(pyfock.__file__)), flush=True)
    t0 = timer()
    energy, dmat = dft.scf()
    print('BENCH wall_scf %.3f energy %r converged %s' % (timer() - t0, energy, dft.converged), flush=True)


PROFILE_KEYS = {
    'Preprocessing': 'preprocessing',
    'Density Fitting': 'df_J_total',
    'DF (gamma)': 'df_gamma',
    'DF (coeff)': 'df_coeff',
    'DF (Jtri)': 'df_J',
    'DF (Cholesky)': 'df_cholesky',
    'Exchange matrix (RI-K)': 'K_total',
    'DIIS': 'diis',
    'KS matrix diagonalization': 'diag',
    'One electron Integrals (S, T, Vnuc)': 'one_electron',
    'Coulomb Integrals (2c2e + 3c2e)': 'ints_2c3c_total',
    'Grids construction': 'grids',
    'Exchange-Correlation Term': 'xc',
    'Misc.': 'misc',
    'Complete SCF': 'scf_total',
}


def parse_log(text):
    out = {}
    m = re.search(r'BENCH molecule (\S+) natoms (\d+) nao_cart (\d+) naux_cart (\d+) nelec (\d+) ncores (\d+) xc (\S+)', text)
    if m:
        out.update(molecule=m.group(1), natoms=int(m.group(2)), nao_cart=int(m.group(3)), naux_cart=int(m.group(4)),
                   nelec=int(m.group(5)), ncores=int(m.group(6)), xc=m.group(7))
    m = re.search(r'BENCH code (\S+)', text)
    if m:
        out['code'] = m.group(1)
    m = re.search(r'BENCH wall_scf ([\d.]+) energy (\S+) converged (\S+)', text)
    if m:
        out.update(wall_scf=float(m.group(1)), energy=float(m.group(2)), converged=m.group(3) == 'True')
    m = re.search(r'SCF Converged after (\d+) iterations', text)
    if m:
        out['iterations'] = int(m.group(1))
    m = re.search(r'RI-HF \(DF_algo=\d+\): (\d+) function pairs x (\d+) (\w+) fit functions, orthonormalized rows ([\d.]+) GB', text)
    if m:
        out.update(nrows=int(m.group(1)), naux_fit=int(m.group(2)), fit_space=m.group(3), rows_gb=float(m.group(4)))
    m = re.search(r'Time taken to orthonormalize the three-center rows for RI-HF:\s+([\d.]+)', text)
    if m:
        out['build_exchange'] = float(m.group(1))
    m = re.search(r'Time taken for the shell-blocked three-center integrals:\s+([\d.]+)', text)
    if m:
        out['build_3c2e'] = float(m.group(1))
    m = re.search(r'Time taken for the near-field three-center integrals and far-field moments:\s+([\d.]+)', text)
    if m:
        out['build_3c2e'] = float(m.group(1))
    m = re.search(r'Time taken for two-centered two-electron integrals ([\d.]+)', text)
    if m:
        out['build_2c2e'] = float(m.group(1))
    for m in re.finditer(r'BENCH_EXTRA (\S+) ([\d.]+)', text):
        out[m.group(1)] = float(m.group(2))
    prof = text.split('Profiling (Wall times in seconds)')
    if len(prof) > 1:
        for line in prof[-1].splitlines():
            mm = re.match(r'^\s*(.*?)\s{2,}([-\d.]+)\s*$', line)
            if mm and mm.group(1).strip() in PROFILE_KEYS:
                out[PROFILE_KEYS[mm.group(1).strip()]] = float(mm.group(2))
    if 'K_total' in out and out.get('iterations'):
        out['K_per_iter'] = out['K_total'] / out['iterations']
    if 'df_J_total' in out and out.get('iterations'):
        out['J_per_iter'] = out['df_J_total'] / out['iterations']
    return out


def markdown_table(results):
    cols = [('molecule', 'molecule'), ('natoms', 'atoms'), ('nao_cart', 'nao(cart)'), ('naux_fit', 'naux(fit)'),
            ('nrows', 'pair rows'), ('iterations', 'iters'), ('build_3c2e', '3c2e build [s]'),
            ('build_exchange', 'orthonormalize rows [s]'), ('K_per_iter', 'K per iter [s]'), ('K_total', 'K total [s]'),
            ('J_per_iter', 'J per iter [s]'), ('scf_total', 'SCF total [s]'), ('energy', 'energy [Ha]')]
    lines = ['| ' + ' | '.join(h for _, h in cols) + ' |', '|' + '---|' * len(cols)]
    for r in results:
        cells = []
        for key, _ in cols:
            v = r.get(key, '')
            if isinstance(v, float):
                v = ('%.4f' % v) if key == 'energy' else ('%.2f' % v)
            cells.append(str(v))
        lines.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


def run_driver(args):
    outdir = args.outdir if os.path.isabs(args.outdir) else os.path.join(HERE, args.outdir)
    os.makedirs(outdir, exist_ok=True)
    env = dict(os.environ, PYTHONUTF8='1', CUPY_ACCELERATORS='')
    mols = list(args.molecules)
    if args.warmup:
        mols = [mols[0]] + mols
    results = []
    for k, xyz in enumerate(mols):
        warm = args.warmup and k == 0
        tag = ('warmup_' if warm else '') + args.tag
        log = os.path.join(outdir, '%s_%s.log' % (tag, os.path.splitext(os.path.basename(xyz))[0]))
        cmd = [sys.executable, os.path.abspath(__file__), '--single', xyz, '--ncores', str(args.ncores),
               '--xc', args.xc]
        if args.df_algo is not None:
            cmd += ['--df-algo', str(args.df_algo)]
        print('[%d/%d] %s -> %s' % (k + 1, len(mols), ' '.join(cmd), log), flush=True)
        t0 = timer()
        with open(log, 'w', encoding='utf-8') as fh:
            proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env, cwd=HERE)
        wall = timer() - t0
        with open(log, encoding='utf-8', errors='replace') as fh:
            res = parse_log(fh.read())
        res['wall_process'] = wall
        res['returncode'] = proc.returncode
        res['log'] = os.path.basename(log)
        print('    done in %.1f s: ' % wall
              + ', '.join('%s=%s' % (k2, res[k2]) for k2 in ('iterations', 'K_total', 'scf_total', 'energy') if k2 in res),
              flush=True)
        if not warm:
            results.append(res)
    with open(os.path.join(outdir, args.tag + '.json'), 'w') as fh:
        json.dump(results, fh, indent=1)
    table = markdown_table(results)
    with open(os.path.join(outdir, args.tag + '.md'), 'w') as fh:
        fh.write('# RI-K benchmark `%s`: xc=%s, %s/%s, SAO, ncores=%d\n\n%s\n'
                 % (args.tag, args.xc, BASIS, AUXBASIS, args.ncores, table))
    print('\n' + table, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--single', default=None, help='run one molecule in this process (used by the driver)')
    ap.add_argument('--molecules', nargs='*', default=MOLECULES)
    ap.add_argument('--ncores', type=int, default=8)
    ap.add_argument('--xc', default='HF')
    ap.add_argument('--df-algo', type=int, default=None)
    ap.add_argument('--tag', default='baseline')
    ap.add_argument('--outdir', default='RI_K_def2-SVP')
    ap.add_argument('--warmup', action='store_true', help='run the first molecule twice and discard the first run')
    args = ap.parse_args()
    if args.single:
        run_single(args.single, args.ncores, args.xc, args.df_algo)
    else:
        run_driver(args)


if __name__ == '__main__':
    main()
