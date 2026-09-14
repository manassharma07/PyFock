"""DFT-D3 dispersion corrections in PyFock.

Semilocal functionals miss London dispersion, so a D3 correction is usually added on top. D3 depends
only on the atomic numbers, the nuclear coordinates and a set of functional-specific damping parameters
-- not on the electron density. It therefore never enters the Kohn-Sham matrix and cannot change the
SCF: PyFock evaluates it once after convergence and adds it to the total energy.

Needs the simple-dftd3 package:   pip install dftd3
(or:  pip install pyfock[dispersion])
"""

import os

ncores = 4
os.environ['OMP_NUM_THREADS'] = str(ncores)
os.environ["OPENBLAS_NUM_THREADS"] = str(ncores)
os.environ["MKL_NUM_THREADS"] = str(ncores)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(ncores)
os.environ["NUMEXPR_NUM_THREADS"] = str(ncores)

from pyfock import Basis, DFT, Dispersion, Mol

# A water dimer: held together largely by a hydrogen bond, so dispersion is small but not negligible.
mol = Mol(coordfile='h2o_dimer.xyz')


# ----------------------------------------------------------------------------------------------
# 1. The correction on its own -- no SCF needed, so this is instant.
# ----------------------------------------------------------------------------------------------
print('D3 dispersion energies of the water dimer (Hartree)')
print('-' * 60)

# The first argument after the molecule is the functional whose D3 parameters to use.
print('PBE,   D3(BJ)            %12.8f' % Dispersion.d3_energy(mol, 'pbe'))
print('B3LYP, D3(BJ)            %12.8f' % Dispersion.d3_energy(mol, 'b3lyp'))
print('B3LYP5,D3(BJ)            %12.8f' % Dispersion.d3_energy(mol, 'b3lyp5'))  # what Skala uses

# 'd3bj' (Becke-Johnson damping) is the default and the usual choice. 'd3zero' is the original
# zero-damping form; see Dispersion.DAMPING_VERSIONS for the full list.
print('PBE,   D3(0)   zero damp %12.8f' % Dispersion.d3_energy(mol, 'pbe', version='d3zero'))

# The three-body Axilrod-Teller-Muto term is off by default. It is small for a dimer and grows for
# larger, denser systems.
print('PBE,   D3(BJ) + ATM      %12.8f' % Dispersion.d3_energy(mol, 'pbe', atm=True))

# Damping parameters can also be given explicitly instead of looked up by functional name. These are
# PBE's own D3(BJ) parameters, so this reproduces the first line exactly.
print('PBE,   explicit params   %12.8f'
      % Dispersion.d3_energy(mol, None, param={'s6': 1.0, 's8': 0.7875, 'a1': 0.4289, 'a2': 4.4407}))

# The nuclear gradient comes back with the energy, ready to be added to the SCF forces.
energy, gradient = Dispersion.d3_energy_and_gradient(mol, 'pbe')
print('\ngradient shape %s, largest component %.3e Ha/Bohr' % (gradient.shape, abs(gradient).max()))


# ----------------------------------------------------------------------------------------------
# 2. Inside an SCF: pass `dispersion` to DFT and the corrected energy is what you get back.
# ----------------------------------------------------------------------------------------------
basis = Basis(mol, {'all': Basis.load(mol=mol, basis_name='def2-SVP')})
auxbasis = Basis(mol, {'all': Basis.load(mol=mol, basis_name='def2-universal-jfit')})

dftObj = DFT(mol, basis, auxbasis, xc='PBE', dispersion='pbe')
dftObj.conv_crit = 1e-7
dftObj.ncores = ncores

# Optional knobs, both shown here at their defaults:
dftObj.dispersion_version = 'd3bj'   # damping function
dftObj.dispersion_atm = False        # three-body ATM term

energy_corrected, dmat = dftObj.scf()

print('\n' + '-' * 60)
print('E(SCF) + E(D3)  = %.10f Ha' % energy_corrected)
print('E(D3) alone     = %.10f Ha' % dftObj.Edisp)
print('E(SCF) alone    = %.10f Ha' % (energy_corrected - dftObj.Edisp))

# The ASE calculator uses the same backend by default:
#     PyFockCalculator(dispersion=True, dispersion_kwargs={'xc': 'pbe'})
# For a GPU run you can switch it to torch-dftd, which evaluates the correction on the device:
#     dispersion_kwargs={'xc': 'pbe', 'backend': 'torch-dftd', 'device': 'cuda'}
