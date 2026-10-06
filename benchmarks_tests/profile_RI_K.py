"""Time the library K_from_exchange for one molecule (stored partner rows vs gathered), outside the SCF."""
import argparse
import os
import sys
from timeit import default_timer as timer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from benchmark_RI_K import set_threads, setup  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--xyz', default='Icosane_C20H42.xyz')
    ap.add_argument('--ncores', type=int, default=8)
    ap.add_argument('--repeat', type=int, default=5)
    args = ap.parse_args()
    set_threads(args.ncores)
    import numpy as np
    import numba
    from threadpoolctl import threadpool_limits
    numba.set_num_threads(args.ncores)
    from pyfock import Integrals
    from pyfock.Integrals import df_algo11_helpers as algo11
    from pyfock.Integrals import df_algo11_exchange as algo11x
    from pyfock.Integrals.schwarz_helpers import eri_4c2e_diag
    from pyfock.DFT_Helper_Coulomb import _pseudo_cartesian_metric_diagonal

    mol, basis, aux, dft = setup(args.xyz, args.ncores)
    nao = basis.bfs_nao
    nocc = mol.nelectrons // 2
    with threadpool_limits(limits=args.ncores, user_api='blas'):
        metric_cart = Integrals.rys_2c2e_symm(aux)
        metric_sph = aux.cart2sph_operator_blockwise(metric_cart)
        diag_pc = _pseudo_cartesian_metric_diagonal(aux, metric_sph, aux.sph2cart_basis()) + 1e-12
        sqrt4 = np.sqrt(np.abs(eri_4c2e_diag(basis)))
        sqrt2 = np.sqrt(np.abs(diag_pc))
        plan = algo11.build_plan(basis, aux, sqrt4, sqrt2, 1e-9, False, sao=True)
        t0 = timer()
        ex = algo11x.build_exchange(plan, basis, aux, metric_sph, sao=True, release_plan_values=True)
        print('build_exchange %.2f s; %s' % (timer() - t0, ex.summary()), flush=True)
        rng = np.random.default_rng(1)
        factor = np.linalg.qr(rng.standard_normal((nao, nocc)))[0] * np.sqrt(2.0)
        for label in ('stored partner rows', 'gathered partner rows'):
            times = []
            for rep in range(args.repeat):
                t0 = timer()
                K = algo11x.K_from_exchange(ex, factor)
                times.append(timer() - t0)
            print('%s: K_from_exchange %s -> best %.3f s' % (label, ' '.join('%.3f' % t for t in times), min(times)), flush=True)
            ex.P = None
        for bq in (256, 512, 1024, 2048, 4096):
            budget = bq * 8 * (nao * nocc + args.ncores * (nocc + ex.max_partner_rows))
            times = []
            for rep in range(3):
                t0 = timer()
                algo11x.K_from_exchange(ex, factor, block_memory_bytes=budget)
                times.append(timer() - t0)
            print('gathered, block %d: best %.3f s' % (min(bq, ex.naux), min(times)), flush=True)


if __name__ == '__main__':
    main()
