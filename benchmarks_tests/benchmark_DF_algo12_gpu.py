"""DF_algo=11 and DF_algo=12 on the CPU and on CUDA: three-centre build and per-iteration Coulomb.

    python benchmarks_tests/benchmark_DF_algo12_gpu.py [--molecules Caffeine Icosane_C20H42 ...]
        [--basis def2-SVP] [--algos 11 12] [--devices cpu gpu] [--mem 4.0] [--repeats 3]
        [--iters 12] [--threshold 1e-9] [--json results.json] [--ncores N]

Every calculation runs in its own subprocess, one at a time, so the CUDA context, the device
memory and the Numba thread pool of one case never affect another (and two CUDA processes never
run at once).  What is measured is exactly the part of the SCF these algorithms change:

* ``build``      the screening/classification metadata plus the three-centre integrals that are
                 stored (for DF_algo=12 also the far-field moments and their branch translations),
* ``gamma``      ``gamma_P = sum_ij D_ij (ij|P)`` for a fixed model density,
* ``J``          ``J_ij = sum_P (ij|P) c_P`` for a fixed model fitting vector,
* ``total``      ``build + iters * (gamma + J)``, the Coulomb cost of a whole SCF.

The model density and fitting vector are the same in every configuration, so the reported
``checksum`` values must agree across CPU/GPU for one algorithm (to rounding) and between the two
algorithms to the accuracy of the multipole approximation.

``--mem`` is the budget (GB) for the stored three-centre blocks; it applies identically to the CPU
and the GPU runs so the comparison is like for like.  Geometries come from benchmarks_tests/<name>.xyz.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

DEFAULT_MOLECULES = ['Ethane', 'Decane_C10H22', 'Icosane_C20H42', 'Tetracontane_C40H82',
                     'Pentacontane_C50H102', 'Octacontane_C80H162',
                     'Caffeine', 'Serotonin', 'Cholesterol', 'Taxol']


def worker(args):
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                 'NUMEXPR_NUM_THREADS', 'NUMBA_NUM_THREADS'):
        if args.ncores:
            os.environ[name] = str(args.ncores)
    sys.path.insert(0, str(ROOT))
    import numpy as np
    from pyfock import Basis, Integrals, Mol
    from pyfock.Integrals.schwarz_helpers import eri_4c2e_diag

    gpu = args.device == 'gpu'
    xp = np
    if gpu:
        import cupy as cp
        xp = cp

    mol = Mol(coordfile=str(HERE / (args.molecule + '.xyz')))
    basis = Basis(mol, {'all': Basis.load(mol=mol, basis_name=args.basis)})
    aux = Basis(mol, {'all': Basis.load(mol=mol, basis_name='def2-universal-jfit')})

    # The fit metric and the Schwarz bounds exactly as pyfock.DFT_Helper_Coulomb builds them
    # in SAO mode (projected, eps-regularized pseudo-Cartesian metric).
    proj = aux.sph2cart_basis() @ aux.cart2sph_basis()
    metric = proj @ Integrals.rys_2c2e_symm(aux) @ proj.T + 1e-12 * np.eye(aux.bfs_nao)
    sqrt4 = np.sqrt(np.abs(eri_4c2e_diag(basis)))
    sqrt2 = np.sqrt(np.abs(np.diag(metric)))
    # A smooth, symmetric model density with a physical trace, and a model fitting vector;
    # both are identical in every configuration, so the checksums are comparable.
    S = Integrals.overlap_mat_symm(basis)
    dmat = S + 0.3 * S @ S
    dmat *= 10.0 * mol.natoms / 3.0 / np.trace(dmat @ S)
    coeff = np.cos(np.arange(aux.bfs_nao) * 0.37)

    if args.algo == 11:
        mod = (__import__('pyfock.Integrals.df_algo11_helpers_cupy', fromlist=['x']) if gpu
               else __import__('pyfock.Integrals.df_algo11_helpers', fromlist=['x']))
        build = mod.build_plan_cupy if gpu else mod.build_plan
        gamma_fn = mod.gamma_from_plan_cupy if gpu else mod.gamma_from_plan
        j_fn = mod.J_from_plan_cupy if gpu else mod.J_from_plan
        extra = {}
    else:
        mod = (__import__('pyfock.Integrals.df_algo12_helpers_cupy', fromlist=['x']) if gpu
               else __import__('pyfock.Integrals.df_algo12_helpers', fromlist=['x']))
        build = mod.build_plan_cupy if gpu else mod.build_plan
        gamma_fn = mod.gamma_from_plan_cupy if gpu else mod.gamma_from_plan
        j_fn = mod.J_from_plan_cupy if gpu else mod.J_from_plan
        extra = dict(options=json.loads(args.options) if args.options else None)

    sq4, sq2 = (xp.asarray(sqrt4), xp.asarray(sqrt2)) if gpu else (sqrt4, sqrt2)
    dm, cf = (xp.asarray(dmat), xp.asarray(coeff)) if gpu else (dmat, coeff)

    t0 = time.perf_counter()
    plan = build(basis, aux, sq4, sq2, args.threshold, True, sao=True,
                 max_memory_gb=args.mem, **extra)
    if gpu:
        xp.cuda.get_current_stream().synchronize()
    t_build = time.perf_counter() - t0

    gamma_fn(plan, dm)          # warm up the JIT / the caches
    j_fn(plan, cf)
    t_gamma = t_j = float('inf')
    for _ in range(args.repeats):
        t0 = time.perf_counter()
        g = gamma_fn(plan, dm)
        t_gamma = min(t_gamma, time.perf_counter() - t0)
        t0 = time.perf_counter()
        j = j_fn(plan, cf)
        t_j = min(t_j, time.perf_counter() - t0)
    g = xp.asnumpy(g) if gpu else g
    j = xp.asnumpy(j) if gpu else j

    result = dict(molecule=args.molecule, natoms=int(mol.natoms), nao=int(basis.bfs_nao),
                  naux=int(aux.bfs_nao), algo=args.algo, device=args.device,
                  build=t_build, gamma=t_gamma, J=t_j,
                  stored_gb=float(plan.memory_gb), fraction_cached=float(plan.fraction_cached),
                  checksum_gamma=float(g @ coeff), checksum_J=float(np.linalg.norm(j)))
    if args.algo == 12:
        result.update(moments_gb=float(plan.moments_gb),
                      far_field=float(plan.fraction_far_field),
                      far_field_work=float(plan.fraction_far_field_work),
                      n_branches=int(plan.n_branches))
    if gpu:
        stats = plan.memory_stats()
        result.update(peak_pool_gb=stats['peak_pool_reserved_bytes'] / 1e9,
                      workspace_gb=float(plan.temporary_memory_gb))
    print('@@RESULT@@' + json.dumps(result), flush=True)


def run_case(molecule, algo, device, args):
    cmd = [sys.executable, str(Path(__file__).resolve()), '--worker', '--molecule', molecule,
           '--algo', str(algo), '--device', device, '--basis', args.basis,
           '--mem', repr(args.mem) if args.mem is not None else 'none',
           '--threshold', repr(args.threshold), '--repeats', str(args.repeats)]
    if args.ncores:
        cmd += ['--ncores', str(args.ncores)]
    if args.options:
        cmd += ['--options', args.options]
    env = dict(os.environ, PYTHONUTF8='1', CUPY_ACCELERATORS='')
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    for line in proc.stdout.splitlines():
        if line.startswith('@@RESULT@@'):
            return json.loads(line[len('@@RESULT@@'):])
    tail = (proc.stderr or proc.stdout).strip().splitlines()[-4:]
    print(f'  !! {molecule} algo={algo} {device} failed:\n     ' + '\n     '.join(tail), flush=True)
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    ap.add_argument('--molecule')
    ap.add_argument('--molecules', nargs='+', default=DEFAULT_MOLECULES)
    ap.add_argument('--basis', default='def2-SVP')
    ap.add_argument('--algo', type=int)
    ap.add_argument('--algos', nargs='+', type=int, default=[11, 12])
    ap.add_argument('--device')
    ap.add_argument('--devices', nargs='+', default=['cpu', 'gpu'])
    ap.add_argument('--mem', default='4.0', help='GB budget for the stored blocks, or "none"')
    ap.add_argument('--threshold', type=float, default=1e-9)
    ap.add_argument('--repeats', type=int, default=3)
    ap.add_argument('--iters', type=int, default=12, help='SCF iterations assumed in the totals')
    ap.add_argument('--ncores', type=int, default=None)
    ap.add_argument('--options', default=None, help='JSON dict of multipole options (DF_algo=12)')
    ap.add_argument('--json', default=None)
    args = ap.parse_args()
    args.mem = None if str(args.mem).lower() in ('none', 'null', '') else float(args.mem)

    if args.worker:
        worker(args)
        return

    print(f'basis {args.basis} / def2-universal-jfit, SAO, threshold {args.threshold:g}, '
          f'block budget {args.mem} GB, {args.repeats} timed repeats, totals for {args.iters} iterations\n')
    results = []
    for molecule in args.molecules:
        if not (HERE / (molecule + '.xyz')).exists():
            print(f'  !! no geometry for {molecule}', flush=True)
            continue
        print(f'{molecule}', flush=True)
        for algo in args.algos:
            for device in args.devices:
                r = run_case(molecule, algo, device, args)
                if r is None:
                    continue
                results.append(r)
                total = r['build'] + args.iters * (r['gamma'] + r['J'])
                r['total'] = total
                print(f'  algo {algo} {device:3s}: build {r["build"]:8.2f} s   '
                      f'gamma {r["gamma"]:7.3f} s   J {r["J"]:7.3f} s   '
                      f'total {total:8.2f} s   stored {r["stored_gb"]:.2f} GB'
                      + (f'   moments {r["moments_gb"]:.2f} GB' if algo == 12 else ''),
                      flush=True)

    print('\n' + '=' * 108)
    print(f'{"molecule":<22}{"nao":>6}{"naux":>7}   '
          f'{"11 CPU":>9}{"11 GPU":>9}{"x":>7}   {"12 CPU":>9}{"12 GPU":>9}{"x":>7}   {"12GPU/11CPU":>12}')
    print('=' * 108)
    index = {(r['molecule'], r['algo'], r['device']): r for r in results}
    for molecule in args.molecules:
        rows = [r for r in results if r['molecule'] == molecule]
        if not rows:
            continue

        def tot(algo, device):
            r = index.get((molecule, algo, device))
            return r['total'] if r else None

        def ratio(a, b):
            return f'{a / b:6.1f}x' if a and b else '     -'

        t = {k: tot(*k) for k in ((11, 'cpu'), (11, 'gpu'), (12, 'cpu'), (12, 'gpu'))}
        fmt = lambda v: f'{v:9.2f}' if v else '        -'  # noqa: E731
        print(f'{molecule:<22}{rows[0]["nao"]:>6}{rows[0]["naux"]:>7}   '
              f'{fmt(t[11, "cpu"])}{fmt(t[11, "gpu"])}{ratio(t[11, "cpu"], t[11, "gpu"]):>7}   '
              f'{fmt(t[12, "cpu"])}{fmt(t[12, "gpu"])}{ratio(t[12, "cpu"], t[12, "gpu"]):>7}   '
              f'{ratio(t[11, "cpu"], t[12, "gpu"]):>12}')
    print('=' * 108)
    print('x = speed-up of the GPU over the CPU for the same algorithm; the last column is the '
          'speed-up of\nthe multipole-accelerated GPU algorithm over the shell-blocked CPU default.')

    worst = 0.0
    for molecule in args.molecules:
        a = index.get((molecule, 11, 'cpu'))
        b = index.get((molecule, 12, 'cpu'))
        for r in (index.get((molecule, 11, 'gpu')), index.get((molecule, 12, 'gpu')), b):
            if a and r:
                worst = max(worst, abs(r['checksum_gamma'] - a['checksum_gamma'])
                            / max(abs(a['checksum_gamma']), 1e-30))
    print(f'\nlargest relative deviation of the gamma checksum from DF_algo=11 on the CPU: {worst:.2e}')

    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=1), encoding='utf-8')
        print(f'wrote {args.json}')


if __name__ == '__main__':
    main()
