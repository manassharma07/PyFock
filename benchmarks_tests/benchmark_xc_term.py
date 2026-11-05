"""
Time the exchange-correlation term on a fixed density: algorithm 3 (``Integrals.eval_xc_3``, the
CPU default) against algorithm 2 (``Integrals.eval_xc_2``), with and without the AO cache.

The SCF is run once (to the usual 1e-7) to obtain the pruned, sorted grid and the converged density
matrix; both are kept in an ``.npz`` in the working directory so that later runs skip the SCF. Then each
algorithm evaluates the XC energy and potential matrix ``nrep`` times on ``ncores`` threads.

    python benchmark_xc_term.py water_cluster_47.xyz --functional PBE --ncores 24 --nrep 3
    python benchmark_xc_term.py water_cluster_47.xyz --functional R2SCAN --ncores 1 --cache --algo 3

The reported numbers are the best of ``nrep`` evaluations, the energy, and the resident memory before
and after the evaluations (the AO cache shows up there).
"""
import argparse
import os
import sys
import time


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('xyz')
    ap.add_argument('--basis', default='def2-SVP')
    ap.add_argument('--auxbasis', default='def2-universal-jfit')
    ap.add_argument('--functional', default='PBE')
    ap.add_argument('--ncores', type=int, default=4)
    ap.add_argument('--nrep', type=int, default=3)
    ap.add_argument('--algo', type=int, nargs='+', default=[3, 2], help='XC algorithms to time (3 and/or 2)')
    ap.add_argument('--cache', action='store_true', help='keep the basis functions on the grid (save_ao_values)')
    ap.add_argument('--blocksize', type=int, default=5000, help='batch size of algorithm 2')
    ap.add_argument('--leaf-size', type=int, default=256, help='leaf size of algorithm 3')
    ap.add_argument('--state', default=None, help='.npz with the grid and density (default: <geometry>_<functional>_<basis>_state.npz in the working directory)')
    args = ap.parse_args()

    for var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[var] = str(args.ncores)
    import numpy as np
    import numba
    import psutil
    from threadpoolctl import threadpool_limits
    from pyfock import Basis, Mol, DFT, XC, Integrals
    from pyfock.Integrals import eval_xc_3_module as m3

    proc = psutil.Process()
    mol = Mol(coordfile=args.xyz)
    basis = Basis(mol, {'all': Basis.load(mol=mol, basis_name=args.basis)})
    funcid = XC.resolve_functional(args.functional)
    state = args.state or (os.path.splitext(os.path.basename(args.xyz))[0] + '_' + args.functional + '_' + args.basis + '_state.npz')
    if not os.path.exists(state):
        auxbasis = Basis(mol, {'all': Basis.load(mol=mol, basis_name=args.auxbasis)})
        dft = DFT(mol, basis, auxbasis, xc=args.functional)
        dft.conv_crit = 1e-7
        dft.max_itr = 20
        dft.ncores = args.ncores
        energy, dmat = dft.scf()
        np.savez(state, coords=np.asarray(dft.grids.coords), weights=np.asarray(dft.grids.weights), dmat=dmat, energy=energy)
        print('\nSaved the grid and the converged density to', state)
    st = np.load(state)
    coords, weights, dmat = st['coords'], st['weights'], st['dmat']
    ngrids = coords.shape[0]
    numba.set_num_threads(args.ncores)
    print('\n%s: %d basis functions, %d grid points, functional %s (%s), %d threads, AO cache %s'
          % (args.xyz, basis.bfs_nao, ngrids, args.functional, funcid, args.ncores, 'on' if args.cache else 'off'))
    fam = XC.xc_semilocal_family(funcid)
    nd = 1 if fam == 1 else 4

    results = {}
    with threadpool_limits(limits=args.ncores, user_api='blas'):
        for algo in args.algo:
            rss0 = proc.memory_info().rss
            t0 = time.perf_counter()
            if algo == 3:
                plan = m3.build_xc_plan(basis, coords, weights, ncores=args.ncores, leaf_size=args.leaf_size)
                cache = plan.build_ao_cache(nd, args.ncores) if args.cache else None
                print('algorithm 3:', plan.describe())

                def run():
                    return Integrals.eval_xc_3(basis, dmat, weights, coords, funcid, ncores=args.ncores, plan=plan,
                                               list_ao_values=cache, print_nelec=False)
            else:
                nblocks = ngrids // args.blocksize
                lni, cni = Integrals.bf_val_helpers.nonzero_ao_indices(basis, coords, args.blocksize, nblocks, ngrids)
                lao = lgr = None
                if args.cache:
                    bfs_data = Integrals.bf_val_helpers.pack_bfs_data(basis)
                    lao, lgr = [], []
                    for ib in range(nblocks + 1):
                        cb = coords[ib * args.blocksize: min((ib + 1) * args.blocksize, ngrids)]
                        idx = lni[ib][:cni[ib]]
                        if nd == 1:
                            lao.append(Integrals.bf_val_helpers.eval_bfs(basis, cb, parallel=True, non_zero_indices=idx, bfs_data=bfs_data))
                        else:
                            a, g = Integrals.bf_val_helpers.eval_bfs_and_grad(basis, cb, parallel=True, non_zero_indices=idx, bfs_data=bfs_data)
                            lao.append(a)
                            lgr.append(g)
                    if nd == 1:
                        lgr = None

                def run():
                    return Integrals.eval_xc_2(basis, dmat, weights, coords, funcid, ncores=args.ncores, blocksize=args.blocksize,
                                               list_nonzero_indices=lni, count_nonzero_indices=cni, list_ao_values=lao,
                                               list_ao_grad_values=lgr, print_nelec=False)
            setup = time.perf_counter() - t0
            e, v = run()     # compilation / warm-up
            times = []
            for _ in range(args.nrep):
                t0 = time.perf_counter()
                e, v = run()
                times.append(time.perf_counter() - t0)
            rss1 = proc.memory_info().rss
            results[algo] = (min(times), e, rss1 - rss0)
            print('algorithm %d: setup %.2f s, XC term best of %d: %.3f s (all: %s), Exc = %.10f, memory +%.2f GB'
                  % (algo, setup, args.nrep, min(times), ', '.join('%.3f' % t for t in times), e, (rss1 - rss0) / 1e9), flush=True)
    if 2 in results and 3 in results:
        print('speedup of algorithm 3 over algorithm 2: %.1fx ; |Exc3 - Exc2| = %.1e Ha'
              % (results[2][0] / results[3][0], abs(results[3][1] - results[2][1])))


if __name__ == '__main__':
    main()
