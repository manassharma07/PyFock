"""End-to-end sanity run of a global hybrid (B3LYP) through the RI-K path on the CPU."""
import os
import sys
from timeit import default_timer as timer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from benchmark_RI_K import set_threads, setup  # noqa: E402

xyz = sys.argv[1] if len(sys.argv) > 1 else 'Benzene.xyz'
set_threads(8)
mol, basis, aux, dft = setup(xyz, 8, xc='B3LYP')
dft.XC_algo = 2
dft.blocksize = 5000
t0 = timer()
energy, dmat = dft.scf()
print('SANITY %s B3LYP/def2-SVP energy %r converged %s exx %r wall %.1f s' % (xyz, energy, dft.converged, dft.Exx_energy, timer() - t0))
