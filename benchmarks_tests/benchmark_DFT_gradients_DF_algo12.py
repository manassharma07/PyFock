"""
Analytical-gradient timings with the DF_algo=12 Coulomb terms against the previous implementation.

One SCF (DF_algo=12, the default) is converged, and its density is then differentiated by

* ``12``        the default: near-field Rys derivatives plus far-field multipole derivatives
                (``pyfock.Integrals.df_algo12_grad``), the fitting coefficients taken from the SCF's
                last iteration;
* ``12-nofit``  the same, with the fitting redone from the plan (what a gradient pays when the SCF
                used another DF algorithm);
* ``10``        the previous implementation: every significant derivative integral, and the fitting
                redone from a full three-center pass.

Every kernel is first run on a single water molecule, so no timing includes compilation. The
per-term timings of every gradient and the largest force difference against ``10`` are printed and
written to ``--json``.

    python benchmark_DFT_gradients_DF_algo12.py --xyz water_cluster_30.xyz --ncores 16
    python benchmark_DFT_gradients_DF_algo12.py --xyz water_cluster_30.xyz --gpu --repeat 3
"""
import argparse
import json
import os
import sys
import time


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument('--xyz', required=True)
    ap.add_argument('--basis', default='def2-TZVPD')
    ap.add_argument('--auxbasis', default='def2-universal-jkfit')
    ap.add_argument('--xc', default='PBE')
    ap.add_argument('--ncores', type=int, default=16)
    ap.add_argument('--cap', type=float, default=4.0,
                    help='GB budget for the SCF\'s cached near-field 3c2e blocks (0 = direct)')
    ap.add_argument('--conv', type=float, default=1e-7)
    ap.add_argument('--algos', default='12,12-nofit,10')
    ap.add_argument('--gpu', action='store_true')
    ap.add_argument('--repeat', type=int, default=1,
                    help='gradients per algorithm; the fastest is reported (use 3 on the GPU, whose '
                         'memory pool grows during the first large calls)')
    ap.add_argument('--json', default=None)
    args = ap.parse_args()
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMBA_NUM_THREADS'):
        os.environ[name] = str(args.ncores)
    os.environ.setdefault("OPENBLAS_THREAD_TIMEOUT", "4")  # idle OpenBLAS threads sleep instead of spinning (read when numpy loads)

    import numpy as np
    from pyfock import Basis, DFT, DFT_Grad, Mol

    algos = [a.strip() for a in args.algos.split(',') if a.strip()]

    def converge(mol):
        basis = Basis(mol, {'all': Basis.load(mol=mol, basis_name=args.basis)})
        aux = Basis(mol, {'all': Basis.load(mol=mol, basis_name=args.auxbasis)})
        dft = DFT(mol, basis, aux, xc=args.xc, use_gpu=args.gpu)
        dft.sao = True
        dft.DF_algo = 12
        dft.conv_crit = args.conv
        dft.ncores = args.ncores
        dft.max_memory_ints3c2e = args.cap
        energy, _ = dft.scf()
        return dft, energy

    def gradient(dft, algo):
        saved = dft.df_fit
        if algo == '12-nofit':
            dft.df_fit = None
        try:
            start = time.perf_counter()
            result = DFT_Grad(dft, DF_algo=int(algo.split('-')[0])).calculate()
            result['wall'] = time.perf_counter() - start
        finally:
            dft.df_fit = saved
        return result

    # warm-up: load or compile every kernel on one water molecule
    water = Mol(atoms=[['O', 0.0, 0.0, 0.117], ['H', 0.0, 0.757, -0.467], ['H', 0.0, -0.757, -0.467]])
    dft, _ = converge(water)
    for algo in algos:
        gradient(dft, algo)

    mol = Mol(coordfile=args.xyz)
    start = time.perf_counter()
    dft, energy = converge(mol)
    t_scf = time.perf_counter() - start
    if not dft.converged:
        sys.exit('SCF did not converge')

    results = {}
    for algo in algos:
        runs = [gradient(dft, algo) for _ in range(max(1, args.repeat))]
        results[algo] = min(runs, key=lambda res: res['wall'])

    ref = results.get('10')
    summary = {'xyz': os.path.abspath(args.xyz), 'natoms': mol.natoms, 'basis': args.basis,
               'auxbasis': args.auxbasis, 'xc': args.xc, 'ncores': args.ncores, 'gpu': args.gpu,
               'nao': dft.basis.bfs_nao, 'naux': dft.auxbasis.bfs_nao, 'energy': energy,
               'scf_seconds': t_scf, 'scf_iterations': dft.niter, 'gradients': {}}
    for algo, res in results.items():
        g = np.asarray(res['gradient'])
        entry = {'wall': res['wall'], 'timings': {k: float(v) for k, v in res['timings'].items()},
                 'net_force': float(np.abs(g.sum(axis=0)).max()), 'max_abs_gradient': float(np.abs(g).max())}
        if ref is not None:
            entry['max_abs_diff_vs_10'] = float(np.abs(g - np.asarray(ref['gradient'])).max())
            dJ = (np.asarray(res['gradient_components']['coulomb_df'])
                  - np.asarray(ref['gradient_components']['coulomb_df']))
            entry['max_abs_coulomb_diff_vs_10'] = float(np.abs(dJ).max())
        summary['gradients'][algo] = entry

    keys = []
    for res in results.values():
        keys += [k for k in res['timings'] if k not in keys]
    print('\n' + '=' * 78)
    print(f"{mol.natoms} atoms, {args.basis}/{args.auxbasis}, {args.xc}, {args.ncores} threads"
          f"{' (GPU)' if args.gpu else ''}: nao {summary['nao']}, naux {summary['naux']}, "
          f"SCF {t_scf:.1f} s in {dft.niter} iterations")
    print(f"{'term (s)':<26}" + ''.join(f'{a:>14}' for a in results))
    for k in keys:
        print(f'{k:<26}' + ''.join(f"{res['timings'].get(k, float('nan')):>14.2f}" for res in results.values()))
    print(f"{'wall':<26}" + ''.join(f"{res['wall']:>14.2f}" for res in results.values()))
    if ref is not None:
        print(f"{'max |dG| vs 10 (Ha/Bohr)':<26}"
              + ''.join(f"{summary['gradients'][a]['max_abs_diff_vs_10']:>14.2e}" for a in results))
    print(f"{'net force (Ha/Bohr)':<26}" + ''.join(f"{summary['gradients'][a]['net_force']:>14.2e}" for a in results))
    print('=' * 78)
    if args.json:
        with open(args.json, 'w', encoding='utf-8') as handle:
            json.dump(summary, handle, indent=2)


if __name__ == '__main__':
    main()
