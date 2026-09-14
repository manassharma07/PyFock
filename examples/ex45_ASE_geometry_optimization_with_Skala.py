"""Geometry optimization with Skala, through PyFock's ASE calculator.

Skala has analytical nuclear gradients in PyFock -- grid response included -- so any ASE optimizer
drives it like a conventional functional. The one choice worth making is how the calculator runs each
step:

  run_in_process=True   (used here) every step runs in this process, so the Skala checkpoint stays
                        loaded, TorchScript stays warmed up and, with skala_gpu=True, the CUDA context
                        stays alive. Each step then converges from the previous step's density in well
                        under a second.

  run_in_process=False  (the default) every step runs in a fresh subprocess. That is the more robust
                        arrangement -- a crash or a step that refuses to converge cannot take the
                        optimizer down, and every step leaves a complete PyFock output file on disk --
                        but it repeats the cold start every time: about a second of imports and a
                        couple of seconds of loading Skala and letting TorchScript warm up, plus a CUDA
                        context when the GPU is in use. PyFock's own Numba kernels are compiled with
                        cache=True and reload from disk, so they are a small part of it.

For Skala, run in process: it is several times faster, because that cold start costs more than a
converged step does. Keep the subprocess default for long unattended runs where robustness matters
more than speed.

One catch with run_in_process=True: BLAS reads its thread count from the environment when numpy is
first imported, so set OMP_NUM_THREADS and friends at the top of the script, as below, rather than
relying on ncores alone.

Needs:  pip install "pyfock[ase,skala]"
"""

import os

ncores = 4
os.environ['OMP_NUM_THREADS'] = str(ncores)
os.environ['OPENBLAS_NUM_THREADS'] = str(ncores)
os.environ['MKL_NUM_THREADS'] = str(ncores)
os.environ['VECLIB_MAXIMUM_THREADS'] = str(ncores)
os.environ['NUMEXPR_NUM_THREADS'] = str(ncores)

from ase import Atoms
from ase.optimize import LBFGSLineSearch

from pyfock import PyFockCalculator

# Water, started well off equilibrium so the optimizer has something to do.
water = Atoms('OHH', positions=[[0.000, 0.000, 0.150],
                                [0.000, 0.850, -0.500],
                                [0.000, -0.850, -0.500]])

water.calc = PyFockCalculator(
    functional='skala-1.1',
    basis='def2-SVP',
    auxbasis='def2-universal-jfit',
    dispersion=True,
    dispersion_kwargs={'xc': 'b3lyp5'},  # the D3 parametrisation Skala was fitted with
    ncores=ncores,
    conv_crit=1e-8,
    save_ao_values=True,
    run_in_process=True,                 # see the note at the top
    # skala_gpu=True,                    # run the neural functional on the GPU (needs CUDA PyTorch)
    directory='ase_skala_optimization',
)

opt = LBFGSLineSearch(water, trajectory='ase_skala_optimization/water.traj')
opt.run(fmax=0.02)

print('\nOptimized water with Skala 1.1 + D3, %d steps' % opt.get_number_of_steps())
print('energy      %.8f eV' % water.get_potential_energy())
print('O-H         %.4f A' % water.get_distance(0, 1))
print('H-O-H       %.2f deg' % water.get_angle(1, 0, 2))
print('max force   %.5f eV/A' % abs(water.get_forces()).max())
