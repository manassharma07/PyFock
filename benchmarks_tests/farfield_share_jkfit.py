"""Far-field share of the (ij|P) tensor and of the Rys work for DF_algo=12 with the JK-fitting basis
(what a DF_algo=12 RI-K build could reconstruct from multipoles instead of evaluating)."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from benchmark_RI_K import MOLECULES, set_threads, setup  # noqa: E402


def main():
    set_threads(8)
    import numpy as np
    import numba
    numba.set_num_threads(8)
    from pyfock import Integrals
    from pyfock.Integrals import df_algo12_helpers as algo12
    from pyfock.Integrals.schwarz_helpers import eri_4c2e_diag
    from pyfock.DFT_Helper_Coulomb import _pseudo_cartesian_metric_diagonal
    for xyz in MOLECULES + ['Tetracontane_C40H82.xyz']:
        if not os.path.exists(os.path.join(HERE, xyz)):
            print(xyz, 'not found')
            continue
        mol, basis, aux, dft = setup(xyz, 8)
        metric_cart = Integrals.rys_2c2e_symm(aux)
        metric_sph = aux.cart2sph_operator_blockwise(metric_cart)
        diag_pc = _pseudo_cartesian_metric_diagonal(aux, metric_sph, aux.sph2cart_basis()) + 1e-12
        sqrt4 = np.sqrt(np.abs(eri_4c2e_diag(basis)))
        sqrt2 = np.sqrt(np.abs(diag_pc))
        plan = algo12._plan_metadata(basis, aux, sqrt4, sqrt2, 1e-9, False, sao=True)
        print('%-24s natoms %3d: far field %5.1f%% of the significant (ij|P), %5.1f%% of the Rys work; '
              '%d branches, %d entries, moments %.3f GB; metadata %.2f s'
              % (xyz, mol.natoms, 100 * plan.fraction_far_field, 100 * plan.fraction_far_field_work,
                 plan.n_branches, plan.n_entries, plan.moments_gb, plan.timings['metadata_total']), flush=True)


if __name__ == '__main__':
    main()
