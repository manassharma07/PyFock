# Geometry optimization with extrapolated starting densities (ASE).
# By default every SCF starts from the converged density of the previous step (density_guess='previous').
# density_guess='extrapolate' carries the densities of the last steps along with the atoms and
# extrapolates them to the new geometry, so the SCFs need fewer iterations along the same path
# (the number of SCF iterations is printed after every BFGS step).
# Builds on density projection/extrapolation code contributed by Prof. Vincenzo Barone.
from ase.build import molecule
from ase.optimize import BFGS

from pyfock import PyFockCalculator

for guess in ('previous', 'extrapolate'):
    print('Using density_guess= ', guess)
    atoms = molecule('CH3CH2OH')
    atoms.rattle(stdev=0.05, seed=7)
    atoms.calc = PyFockCalculator(functional='PBE', basis='def2-SVP', ncores=4,
                                  density_guess=guess, directory='ase_' + guess)
    opt = BFGS(atoms)
    opt.attach(lambda: print('    SCF iterations:', atoms.calc.pyfock_results['niter']))
    opt.run(fmax=0.04)
