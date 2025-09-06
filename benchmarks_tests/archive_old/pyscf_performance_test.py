import os

ncores = 8
os.environ['OMP_NUM_THREADS'] = str(ncores)
os.environ["OPENBLAS_NUM_THREADS"] = str(ncores) # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = str(ncores) # export MKL_NUM_THREADS=4
os.environ["VECLIB_MAXIMUM_THREADS"] = str(ncores) # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = str(ncores) # export NUMEXPR_NUM_THREADS=4
# Set the max memory for PySCF
os.environ["PYSCF_MAX_MEMORY"] = str(25000) 

import psi4
import pyscf
from pyscf import dft
# pyscf.lib.misc.num_threads(n=8)


# a molecule
geo_txt = """
C -1.561000 -4.140600 -0.976800
C -0.281700 -3.352100 -0.725900
O 0.833900 -4.207300 -0.637200
C -0.325500 -2.399200 0.483200
C -0.326500 -1.185200 -0.466500
C 0.724100 -0.136500 -0.243900
C 1.188600 0.175800 1.030200
C 2.082700 1.210900 1.214500
C 2.532100 1.967100 0.142200
C 2.063800 1.655100 -1.120000
O 2.436700 2.345100 -2.293400
C 3.106200 3.471800 -2.216100
F 4.358400 3.350000 -1.731500
F 2.522300 4.417300 -1.454900
F 3.228000 3.988100 -3.436000
C 1.172300 0.614500 -1.321100
C -1.713300 -0.462900 -0.574100
O -1.897200 0.080400 -1.848300
C -1.786800 0.612100 0.483400
C -2.096200 0.351700 1.812600
C -2.054900 1.400000 2.716000
C -1.710600 2.665100 2.265200
C -1.438900 2.825200 0.916400
N -1.477400 1.829200 0.051200
C -0.092600 -2.134400 -1.661600
H -1.488800 -4.680700 -1.920300
H -1.715000 -4.859800 -0.174700
H -2.417100 -3.474700 -1.027500
H 0.882400 -4.741900 -1.437500
H -1.189600 -2.508500 1.133500
H 0.586900 -2.489000 1.066500
H 0.837600 -0.383000 1.883700
H 2.436700 1.446600 2.206400
H 3.225400 2.774300 0.313600
H 0.826000 0.411700 -2.321000
H -2.503700 -1.207300 -0.424400
H -1.573700 0.994600 -1.802600
H -2.366200 -0.643400 2.129400
H -2.289200 1.233100 3.757500
H -1.663600 3.504900 2.940000
H -1.177900 3.790200 0.502400
H -0.815300 -2.035100 -2.467500
H 0.919100 -2.073000 -2.056600
symmetry C1
"""
psi4_geo = psi4.geometry(geo_txt)

# run psi4 calculation with 8 cores
psi4.core.set_num_threads(8)
psi4.set_options(
    {
        "scf__reference": "rhf",
        "scf__maxiter": 50,
        'basis': 'def2-svp',
    }
)
energy, wfn = psi4.energy("WB97X-D", molecule=psi4_geo, return_wfn=True)

# run PySCF calculation
mol = pyscf.M(
    atom=geo_txt.replace("\nsymmetry C1", ""),  # just removing the last line that was only for psi4
    basis="def2-svp",
    symmetry=True,
    verbose=5,
)
# mol.max_memory = 10_000
mf = dft.rks.RKS(mol, xc="WB97X-D").density_fit()
mf.kernel()