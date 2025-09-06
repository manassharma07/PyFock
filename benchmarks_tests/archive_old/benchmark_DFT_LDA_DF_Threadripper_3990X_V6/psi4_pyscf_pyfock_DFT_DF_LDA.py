#DFT SCF benchmark and comparison with PySCF and Psi4
# Benchmarking accuracy and performance using various techniques and different softwares
import os

ncores = 4
max_memory = 120000

os.environ['OMP_NUM_THREADS'] = str(ncores)
os.environ["OPENBLAS_NUM_THREADS"] = str(ncores) # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = str(ncores) # export MKL_NUM_THREADS=4
os.environ["VECLIB_MAXIMUM_THREADS"] = str(ncores) # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = str(ncores) # export NUMEXPR_NUM_THREADS=4
# Set the max memory for PySCF
os.environ["PYSCF_MAX_MEMORY"] = str(max_memory)

# Print system information 
from pyfock import Utils
Utils.print_sys_info()

# Check if the environment variables are properly set
print("Number of cores being actually used/requested for the benchmark:", ncores)
print('Confirming that the environment variables are properly set...')
print('OMP_NUM_THREADS =', os.environ.get('OMP_NUM_THREADS', None))
print('OPENBLAS_NUM_THREADS =', os.environ.get('OPENBLAS_NUM_THREADS', None))
print('MKL_NUM_THREADS =', os.environ.get('MKL_NUM_THREADS', None))
print('VECLIB_MAXIMUM_THREADS =', os.environ.get('VECLIB_MAXIMUM_THREADS', None))
print('NUMEXPR_NUM_THREADS =', os.environ.get('NUMEXPR_NUM_THREADS', None))
print('PYSCF_MAX_MEMORY =', os.environ.get('PYSCF_MAX_MEMORY', None))

from timeit import default_timer as timer

radial_points = 80
spherical_points = 590

#funcpsi4 = "svwn"
funcpyscf = "LDA_X, LDA_C_VWN" #'1,7'
funcx = 1 #LDA_X
funcc = 7 #LDA_C_VWN
funccrysx = [funcx, funcc]
funcpsi4 = {
    "name": "LDA,VWN5",
    "x_functionals": {"LDA_X":{}},
    "c_functionals": {"LDA_C_VWN": {}}
}

# basis_set_name = 'sto-2g'
# basis_set_name = 'sto-3g'
# basis_set_name = 'sto-6g'
# basis_set_name = '6-31G'
basis_set_name = 'def2-SVP'
# basis_set_name = 'def2-DZVP'
# basis_set_name = 'def2-TZVP'
# basis_set_name = 'def2-TZVPP'
# basis_set_name = 'def2-TZVPPD'
# basis_set_name = 'def2-QZVPPD'
# basis_set_name = 'cc-pVDZ'
# basis_set_name = 'ano-rcc'

auxbasis_name = 'def2-universal-jfit'
# auxbasis_name = 'sto-3g'
# auxbasis_name = 'def2-SVP'
# auxbasis_name = '6-31G'

# xyzFilename = 'Benzene-Fulvene_Dimer.xyz'
# xyzFilename = 'Adenine-Thymine.xyz'
# xyzFilename = 'Zn.xyz'
# xyzFilename = 'Zn_dimer.xyz'
# xyzFilename = 'TPP.xyz'
# xyzFilename = 'Zn_TPP.xyz'
# xyzFilename = 'h2o.xyz'
# xyzFilename = 'Ethane.xyz'
# xyzFilename = 'Cholesterol.xyz'
# xyzFilename = 'Serotonin.xyz'
# xyzFilename = 'Decane_C10H22.xyz'
xyzFilename = 'Icosane_C20H42.xyz'
# xyzFilename = 'Tetracontane_C40H82.xyz'
# xyzFilename = 'Pentacontane_C50H102.xyz'
# xyzFilename = 'Octacontane_C80H162.xyz'
# xyzFilename = 'Hectane_C100H202.xyz'
# xyzFilename = 'Icosahectane_C120H242.xyz'

###############-----------------PSI4-----------------###############

import psi4
psi4.set_memory(int(max_memory*1e6))
numpy_memory = 80
import numpy as np

#psi4_geo = psi4.geometry(geo_txt)
with open(xyzFilename,"r") as f:
    xyz_string = f.read()

psi4_geo = psi4.core.Molecule.from_string(xyz_string, dtype='xyz',fix_symmetry='c1', fix_orientation=True,fix_com=True)

# run psi4 calculation
psi4.core.set_num_threads(ncores)
psi4.set_options(
    {
        "scf__reference": "rks",
        "scf_type": "df",
        "scf__maxiter": 50,
        "basis": basis_set_name,
        "df_basis_scf": auxbasis_name,
        "puream":False,
        "guess": "sad",
        "SCF_INITIAL_ACCELERATOR":"none",
        "dft_spherical_points":spherical_points,
        "dft_radial_points" :radial_points,
        "dft_pruning_scheme": "robust",
        "diis_max_vecs" : 8,
        #"DFT_NUCLEAR_SCHEME":"becke",
        "E_CONVERGENCE":1e-7,
    }
)
energy_psi4, wfn = psi4.energy(name="scf",dft_functional=funcpsi4, molecule=psi4_geo, return_wfn=True)
basis = wfn.basisset()

# Get grid points that were used by psi4 calculation
functional = psi4.driver.dft.build_superfunctional(funcpsi4, True)[0] # True states that we're working with a restricted system
Vpot       = psi4.core.VBase.build(basis, functional, "RV")         # This object contains different methods associated with DFT methods and the grid.
                                                                    # "RV" states that this is a restricted system consystent with 'functional'
Vpot.initialize() # We initialize the object

# The grid (and weights) can then be extracted from Vpot.
x, y, z, weights_psi4 = Vpot.get_np_xyzw()
Vpot = 0 # Get rid of unnecessary stuff
#print(x.shape)
#print(y.shape)
#print(z.shape)
#print(weights_psi4.shape)

# Combine the arrays into a single 2D array
points = np.column_stack((x, y, z))
# Get rid of unnecessary stuff
x=0 
y=0
z=0
wfn = 0
print('Psi4 Grid Size: ', weights_psi4.shape)



###############-----------------PySCF-----------------###############
import pyscf
from pyscf import dft, gto
# run PySCF calculation
mol = gto.M(
    atom=xyzFilename,  # just removing the last line that was only for psi4
    basis=basis_set_name,
    symmetry=False,
    verbose=4,
    cart=True,
)


mf = dft.rks.RKS(mol, xc=funcpyscf).density_fit(auxbasis=auxbasis_name)
mf.init_guess = 'minao'
dmat_init = mf.init_guess_by_minao(mol)
mf.grids.level = 5
mf.verbose = 4
# mf.grids.prune = None
# mf.grids.atom_grid = (radial_points, spherical_points)
# mf.grids.build()
# mf.grids.becke_scheme = dft.gen_grid.stratmann
# mf.small_rho_cutoff = 1e-15
#mf.grids.coords = points
#print('Check to see if using psi4 grids worked or not?', mf.grids.coords.shape)
#mf.grids.weights = weights_psi4
#print('Check to see if using psi4 grids worked or not?', mf.grids.weights.shape)
mf.conv_tol = 1e-7
start=timer()
energy_pyscf = mf.kernel()
duration = timer()-start
print('PySCF time: ', duration)
pyscfGrids = mf.grids
print(mf.grids.coords.shape)
print('Nuc-Nuc PySCF= ', mol.energy_nuc())
print('One electron integrals energy',mf.scf_summary['e1'])
print('Coulomb energy ',mf.scf_summary['coul'])
print('EXC ',mf.scf_summary['exc'])
print('Energy diff (Pyscf-Psi4)', abs(energy_pyscf-energy_psi4))
print('PySCF Grid Size: ', pyscfGrids.weights.shape)
# Get rid of unnecessary stuff
mf = 0
weights_psi4 = 0




###############-----------------PyFOCK-----------------###############
from pyfock import Basis
from pyfock import Mol
from pyfock import Integrals
from pyfock import DFT



#Initialize a Mol object with somewhat large geometry
molCrysX = Mol(coordfile=xyzFilename)


#Initialize a Basis object with a very large basis set
basis = Basis(molCrysX, {'all':Basis.load(mol=molCrysX, basis_name=basis_set_name)})
print('\n\nNAO :',basis.bfs_nao)

auxbasis = Basis(molCrysX, {'all':Basis.load(mol=molCrysX, basis_name=auxbasis_name)})
print('\n\naux NAO :',auxbasis.bfs_nao)

dftObj = DFT(molCrysX, basis, xc=funccrysx)
#dftObj.diisSpace = 8
# Using PySCF/Psi4 grids to compare the energies
energy_crysX, dmat = dftObj.scf(max_itr=30, ncores=ncores, dmat=dmat_init, grids=pyscfGrids, isDF=True, auxbasis=auxbasis, rys=True, DF_algo=6, blocksize=5000, XC_algo=2, debug=False, sortGrids=False, save_ao_values=False, threshold_schwarz=1e-09)
# Using CrysX grids 
# To get the same energies as PySCF (level=5) upto 1e-7 au, use the following settings
# radial_precision=1.0e-13
# level=3
# pruning by density with threshold = 1e-011
# alpha_min and alpha_max corresponding to QZVP
# energyCrysX, dmat = dftObj.scf(max_itr=30, ncores=ncores, dmat=dmat_init, grids=None, gridsLevel=3, isDF=True, auxbasis=auxbasis, rys=True, DF_algo=6, blocksize=5000, XC_algo=2, debug=False, sortGrids=True, save_ao_values=False)



print('Energy diff (PySCF-CrysX)', abs(energy_crysX-energy_pyscf))
print('Energy diff (Psi4-CrysX)', abs(energy_crysX-energy_psi4))



#Print package versions
import joblib
import scipy
import numba
import threadpoolctl
import opt_einsum
import pylibxc
import llvmlite 
print('\n\n\n Package versions')
print('pyscf version', pyscf.__version__)
print('psi4 version', psi4.__version__)
print('np version', np.__version__)
print('joblib version', joblib.__version__)
print('numba version', numba.__version__)
print('threadpoolctl version', threadpoolctl.__version__)
print('opt_einsum version', opt_einsum.__version__)
# print('pylibxc version', pylibxc.__version__)
print('llvmlite version', llvmlite.__version__)



print('\n\n\n Numpy Config')
print(np.show_config())
