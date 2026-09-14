"""Validate PyFock's Skala implementation against Microsoft's published reference energies.

The Skala repository ships the total energies behind its benchmark report as
``benchmark/reference/measurements.json`` (a Git-LFS file of about 7.8 MB): 1221 converged runs covering
skala-1.1, r2scan, m06-2x and b3lyp5, each at def2-SVP, def2-TZVP and def2-QZVP, for roughly 30 molecules
drawn from GMTKN55 and a conformer benchmark. The geometries themselves are not redistributed; they are
pulled from the upstream `grimme-lab/GMTKN55` repository at a pinned commit.

Matching their protocol matters, and it is not PyFock's default:

===================  ==========================================  ==============================
Setting              Reference                                   PyFock
===================  ==========================================  ==============================
orbitals             spherical (UKS, ``nAO`` = 24 for H2O/SVP)   Cartesian by default -> ``sao = True``
density fitting      on, ``def2-universal-jkfit``                 ``def2-universal-jfit`` in the examples
grid                 PySCF level 3                               native level 3
convergence          ``conv_tol = 5e-6`` on the energy            ``conv_crit``
dispersion           included for Skala, absent for the rest      off by default -> ``dispersion=True``
===================  ==========================================  ==============================

The dispersion row is easy to get wrong. ``benchmark/runner.py`` never mentions D3, but the energies it
records still contain it: the runner builds the method through ``SkalaKS``, whose constructor defaults
to ``with_dftd3=True``, so ``mf.kernel()`` returns the D3-corrected energy. The plain PySCF functionals
in the same reference set (r2scan, m06-2x, b3lyp5) get no such treatment. This script therefore enables
``dispersion=True`` for Skala and leaves it off for everything else.

The script runs **r2SCAN** alongside Skala as a control. Its PyFock implementation is independently
validated, so whatever disagreement it shows against the same reference measures the grid, basis and
density-fitting baseline rather than anything to do with Skala.

Usage::

    python validate_skala_reference.py                 # H2 and H2O, def2-SVP and def2-TZVP
    python validate_skala_reference.py --basis def2-tzvp --molecules H2O

Needs network access on the first run to fetch the reference file and the geometries; both are cached
next to this script under ``skala_reference_cache/``.
"""

import argparse
import json
import os
import sys
import urllib.request

import numpy as np
from timeit import default_timer as timer

from pyfock import Basis, Data, DFT, Grids, Mol

MEASUREMENTS_URL = ('https://media.githubusercontent.com/media/microsoft/skala/main/'
                    'benchmark/reference/measurements.json')
GMTKN55_REPO = 'grimme-lab/GMTKN55'
GMTKN55_COMMIT = '8d485b37a1ca8837e395042671ca5ba4e0714691'

#: Molecules this script knows how to fetch, as (reference mol_name, GMTKN55 path).
MOLECULES = {
    'H2':   'W4-11/h2/coord',
    'H2O':  'W4-11/h2o/coord',
    'H2O2': 'W4-11/hooh/coord',
    'H3N':  'W4-11/nh3/coord',
    'C2N2': 'W4-11/nccn/coord',
    'CHNO': 'W4-11/hnco/coord',
}



def cache_dir():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'skala_reference_cache')
    os.makedirs(path, exist_ok=True)
    return path


def _download(url, filename):
    """Fetch ``url`` once into the cache and return the local path."""
    path = os.path.join(cache_dir(), filename)
    if not os.path.isfile(path):
        print('downloading ' + url + ' ...', flush=True)
        with urllib.request.urlopen(url, timeout=120) as response:
            payload = response.read()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as handle:
            handle.write(payload)
    return path


def load_reference():
    """The published measurements, keyed by ``(mol_name, functional, basis)`` -> mean total energy."""
    path = _download(MEASUREMENTS_URL, 'measurements.json')
    with open(path, encoding='utf-8') as handle:
        records = json.load(handle)
    grouped = {}
    for record in records:
        if record.get('status') != 'ok' or record.get('total_energy') is None:
            continue
        key = (record['mol_name'], record['functional'], record['basis'])
        grouped.setdefault(key, []).append(float(record['total_energy']))
    # Each configuration was measured a few times; the spread is the reference's own noise.
    return {key: (float(np.mean(values)), float(np.ptp(values))) for key, values in grouped.items()}


def load_geometry(name):
    """A PyFock :class:`~pyfock.Mol.Mol` for one GMTKN55 entry (Turbomole ``coord``, Bohr)."""
    relative = MOLECULES[name]
    url = 'https://raw.githubusercontent.com/%s/%s/%s' % (GMTKN55_REPO, GMTKN55_COMMIT, relative)
    path = _download(url, relative.replace('/', '_'))
    atoms = []
    with open(path, encoding='utf-8') as handle:
        inside = False
        for line in handle:
            token = line.strip()
            if token.startswith('$coord'):
                inside = True
                continue
            if token.startswith('$'):
                inside = False
                continue
            if inside and token:
                x, y, z, symbol = token.split()[:4]
                # PyFock's Mol takes Angstrom; GMTKN55 coord files are in Bohr. Divide by
                # Angs2BohrFactor rather than multiplying by Bohr2AngsFactor (the repo convention,
                # see DFT_Grad): Mol multiplies by the same constant, so the round trip is exact,
                # which multiplying by the non-reciprocal Bohr2AngsFactor would not be.
                scale = 1.0 / Data.Angs2BohrFactor
                atoms.append([symbol.capitalize(), float(x) * scale,
                              float(y) * scale, float(z) * scale])
    return Mol(atoms=atoms)


def run_pyfock(mol, basis_name, xc, grid_level=3, conv_crit=5e-6, dispersion=None):
    """One PyFock energy under the reference protocol (spherical orbitals, DF with jkfit)."""
    basis = Basis(mol, {'all': Basis.load(mol=mol, basis_name=basis_name)})
    auxbasis = Basis(mol, {'all': Basis.load(mol=mol, basis_name='def2-universal-jkfit')})
    dft = DFT(mol, basis, auxbasis, xc=xc, grids=Grids(mol, level=grid_level, verbose=False),
              dispersion=dispersion)
    dft.sao = True            # the reference runs use spherical orbitals
    dft.conv_crit = conv_crit
    dft.max_itr = 50
    start = timer()
    energy, _ = dft.scf()
    return float(energy), timer() - start, dft.converged, dft.niter


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--molecules', nargs='+', default=['H2', 'H2O'], choices=sorted(MOLECULES))
    parser.add_argument('--basis', nargs='+', default=['def2-svp', 'def2-tzvp'])
    parser.add_argument('--functionals', nargs='+', default=['skala-1.1', 'r2scan'])
    parser.add_argument('--grid-level', type=int, default=3)
    args = parser.parse_args(argv)

    reference = load_reference()
    print('=' * 96)
    print('PyFock vs. the Skala reference measurements  (spherical orbitals, DF/def2-universal-jkfit, '
          'grid level %d)' % args.grid_level)
    print('=' * 96)
    print('%-6s %-11s %-10s %18s %18s %12s %9s' %
          ('mol', 'basis', 'functional', 'reference (Ha)', 'PyFock (Ha)', 'diff (Ha)', 'iter'))
    print('-' * 96)

    rows = []
    for name in args.molecules:
        mol = load_geometry(name)
        for basis_name in args.basis:
            for functional in args.functionals:
                key = (name, functional, basis_name)
                if key not in reference:
                    print('%-6s %-11s %-10s  -- not in the reference set --' % (name, basis_name, functional))
                    continue
                ref_energy, ref_spread = reference[key]
                is_skala = functional.startswith('skala')
                xc = functional if is_skala else functional.upper()
                # SkalaKS attaches DFT-D3 by default (with_dftd3=True), so the published Skala energies
                # include the correction; the plain PySCF functionals in the reference set do not.
                try:
                    energy, _, converged, iterations = run_pyfock(
                        mol, basis_name, xc, grid_level=args.grid_level,
                        dispersion=True if is_skala else None)
                except Exception as error:
                    print('%-6s %-11s %-10s  FAILED: %s' % (name, basis_name, functional, error))
                    continue
                difference = energy - ref_energy
                rows.append((name, basis_name, functional, difference))
                print('%-6s %-11s %-10s %18.10f %18.10f %12.2e %9s%s' %
                      (name, basis_name, functional, ref_energy, energy, difference, iterations,
                       '' if converged else '  NOT CONVERGED'))

    if rows:
        print('-' * 96)
        print('Mean |difference| by functional (the grid/basis/DF baseline is what r2SCAN shows):')
        for functional in args.functionals:
            differences = [abs(d) for _, _, f, d in rows if f == functional]
            if differences:
                print('  %-10s  mean %8.2e Ha   max %8.2e Ha   over %d systems'
                      % (functional, float(np.mean(differences)), float(np.max(differences)),
                         len(differences)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
