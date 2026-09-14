"""Skala, the neural exchange-correlation functional, with and without its dispersion correction.

Skala is a machine-learned functional from Microsoft Research AI for Science. In PyFock you select it
the same way as any other functional -- by name -- and everything else stays the same.

One thing sets it apart: Skala is parametrised *together with* a DFT-D3 correction (D3(BJ) with B3LYP5
parameters). The correction is additive and does not enter the SCF, so PyFock leaves it out unless you
ask for it. Leaving it out is right when comparing against Skala's own published reference energies
minus dispersion; including it is right for real chemistry, and it is what reproduces the numbers in
Skala's benchmark report.

Needs:   pip install torch huggingface_hub dftd3
(or:     pip install pyfock[skala])

The ~2.4 MB checkpoint is downloaded from Hugging Face on first use and cached afterwards. PyFock reads
it directly with torch.jit.load, so the `skala` package itself -- and with it PySCF -- is not required,
which is also why this works on Windows.
"""

import os

ncores = 4
os.environ['OMP_NUM_THREADS'] = str(ncores)
os.environ["OPENBLAS_NUM_THREADS"] = str(ncores)
os.environ["MKL_NUM_THREADS"] = str(ncores)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(ncores)
os.environ["NUMEXPR_NUM_THREADS"] = str(ncores)

from pyfock import Basis, DFT, Grids, Mol

mol = Mol(coordfile='h2o.xyz')

basis = Basis(mol, {'all': Basis.load(mol=mol, basis_name='def2-SVP')})
auxbasis = Basis(mol, {'all': Basis.load(mol=mol, basis_name='def2-universal-jfit')})


def run(dispersion):
    # Skala needs the native ('treutler') grids, which are PyFock's default: it uses both the
    # Becke-partitioned weights and the raw single-atom ones, and the 'numgrid' scheme cannot supply
    # the latter. A fresh Grids object is built per run so the two are independent.
    dftObj = DFT(mol, basis, auxbasis, xc='skala-1.1',
                 grids=Grids(mol, level=3, verbose=False),
                 dispersion=dispersion)
    dftObj.conv_crit = 1e-8
    dftObj.ncores = ncores
    energy, dmat = dftObj.scf()
    return dftObj, float(energy)


# ----------------------------------------------------------------------------------------------
# 1. Plain Skala -- the bare SCF energy.
# ----------------------------------------------------------------------------------------------
plain, energy_plain = run(dispersion=None)

# `dftObj.skala` is the loaded model; it reports which D3 parametrisation the checkpoint expects.
print('\nfunctional     :', plain.skala.name)
print('D3 expected    :', plain.skala.d3_settings())
print('features used  :', plain.skala.features)


# ----------------------------------------------------------------------------------------------
# 2. Skala + D3. `dispersion=True` reads the parametrisation out of the checkpoint, so you do not
#    have to know that it is 'b3lyp5'. (Passing the name explicitly would also work.)
# ----------------------------------------------------------------------------------------------
corrected, energy_corrected = run(dispersion=True)

print('\n' + '=' * 64)
print('H2O / def2-SVP / level-3 grid / density fitting')
print('=' * 64)
print('Skala                       %.10f Ha  (%d iterations)' % (energy_plain, plain.niter))
print('Skala + D3(BJ)/b3lyp5       %.10f Ha' % energy_corrected)
print('dispersion contribution     %.10f Ha' % corrected.Edisp)
print('=' * 64)

# To reproduce the energies published in Skala's own benchmark report, also switch to spherical
# orbitals and the jkfit auxiliary basis, which is what their runs use:
#     dftObj.sao = True
#     auxbasis = Basis(mol, {'all': Basis.load(mol=mol, basis_name='def2-universal-jkfit')})
# and keep dispersion on -- SkalaKS attaches D3 by default, so their numbers include it.
# See benchmarks_tests/validate_skala_reference.py, which does exactly this and agrees to ~1e-8 Ha.
