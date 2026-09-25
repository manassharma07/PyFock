"""DFT-D3 dispersion corrections.

The D3 correction is a purely geometric, additive term: it depends on the atomic numbers, the nuclear
coordinates and a set of functional-specific damping parameters, and on nothing else. It does not involve
the electron density, does not enter the Kohn-Sham matrix and does not change the SCF in any way. That is
why PyFock evaluates it once, after the SCF has converged, and simply adds it to the total energy.

Only real atoms take part, each with the atomic number of its element: ghost atoms (counterpoise
calculations) are left out, and an atom with an ECP enters as its element rather than with the effective
charge ``mol.Zcharges`` holds once an ECP basis is loaded. A counterpoise fragment with ghost partners
therefore gets exactly the dispersion energy of its real atoms.

The numbers come from `simple-dftd3 <https://github.com/dftd3/simple-dftd3>`_ (``pip install dftd3``),
through its ``dftd3.interface`` module. Note that the ``dftd3.pyscf`` submodule of the same package needs
PySCF, but ``dftd3.interface`` does not, so PyFock stays free of that dependency.

This module is the core-level dispersion API, used by ``DFT(..., dispersion=...)``. PyFock's ASE
calculator has a **separate** dispersion path of its own, reached through
``PyFockCalculator(dispersion=True, dispersion_kwargs=...)``, which is built on ``torch-dftd`` and works
in ASE's eV/Angstrom units. The two are independent; this one is the reference implementation from the
Grimme group and is what reproduces the published Skala energies.

The main consumer is the Skala neural functional, which is parametrised together with D3(BJ) using the
B3LYP5 damping parameters and no three-body term -- exactly the defaults of
``dftd3.interface.RationalDampingParam``. :func:`pyfock.XC.SkalaFunctional.d3_settings` reports the
parametrisation the checkpoint declares, and ``DFT(..., dispersion=True)`` picks it up automatically.

References
----------
S. Grimme, J. Antony, S. Ehrlich and H. Krieg, J. Chem. Phys. 132, 154104 (2010) [D3];
S. Grimme, S. Ehrlich and L. Goerigk, J. Comput. Chem. 32, 1456 (2011) [Becke-Johnson damping].
"""

import numpy as np

__all__ = ['d3_energy', 'd3_energy_and_gradient', 'DAMPING_VERSIONS']


# Damping function -> the parameter class of dftd3.interface that implements it.
DAMPING_VERSIONS = {
    'd3bj': 'RationalDampingParam',
    'd3zero': 'ZeroDampingParam',
    'd3bjm': 'ModifiedRationalDampingParam',
    'd3mbj': 'ModifiedRationalDampingParam',
    'd3zerom': 'ModifiedZeroDampingParam',
    'd3mzero': 'ModifiedZeroDampingParam',
    'd3op': 'OptimizedPowerDampingParam',
}

_CITATION = ('S. Grimme, J. Antony, S. Ehrlich and H. Krieg, J. Chem. Phys. 132, 154104 (2010); '
             'S. Grimme, S. Ehrlich and L. Goerigk, J. Comput. Chem. 32, 1456 (2011) [BJ damping]. '
             'Evaluated with simple-dftd3 (https://github.com/dftd3/simple-dftd3).')


def _damping_param(method, version, atm, param):
    """Build the ``dftd3.interface`` damping-parameter object for ``method``."""
    try:
        import dftd3.interface as interface
    except ImportError:
        raise ImportError(
            'The D3 dispersion correction needs the simple-dftd3 package: pip install dftd3. '
            "Only its 'dftd3.interface' module is used, which does not require PySCF.")

    version = str(version).lower()
    if version not in DAMPING_VERSIONS:
        raise ValueError("Unknown D3 damping '" + str(version) + "'. Available: "
                         + ', '.join(sorted(DAMPING_VERSIONS)) + '.')
    cls = getattr(interface, DAMPING_VERSIONS[version])
    if param is None:
        # Looked up by functional name: the library takes the ATM flag directly.
        return cls(method=str(method), atm=atm)
    # Explicit parameters go through a different constructor, which has no `atm` argument and instead
    # controls the three-body term through s9 -- defaulting it to 1.0, i.e. ATM *on*. Honour `atm`
    # unless the caller set s9 themselves.
    param = dict(param)
    param.setdefault('s9', 1.0 if atm else 0.0)
    return cls(**param)


def _geometry(mol):
    """The atoms D3 sees: ``(atomic_numbers, positions_in_bohr, their indices in mol, natoms of mol)``.

    ``mol`` is a PyFock :class:`~pyfock.Mol.Mol` or a bare ``(atomic_numbers, positions_in_bohr)`` pair,
    which is how callers that do not have a ``Mol`` -- the ASE calculator, for instance -- reach the same
    code. Two kinds of atom need care:

    * ghost atoms carry basis functions but no nucleus, so they have no dispersion. They are left out
      rather than handed to the library with Z = 0, where they would still count towards the coordination
      numbers of nearby atoms -- shifting those atoms' C6 coefficients -- and receive gradients of their
      own. In a bare pair, atomic number 0 marks a ghost;
    * an atom with an ECP enters with the atomic number of its element (``Mol.element_numbers``). Once an
      ECP basis is loaded ``mol.Zcharges`` holds the effective charge -- iodine with a 28-electron ECP
      reads 25, manganese -- and D3's parameters belong to the element.
    """
    if hasattr(mol, 'Zcharges'):
        numbers = mol.element_numbers() if hasattr(mol, 'element_numbers') else mol.Zcharges
        ghost = mol.ghost_mask() if hasattr(mol, 'ghost_mask') else None
        positions = mol.coordsBohrs
    else:
        numbers, positions = mol
        ghost = None
    numbers = np.asarray(numbers, dtype=np.int64).reshape(-1)
    positions = np.ascontiguousarray(np.asarray(positions, dtype=np.float64).reshape(-1, 3))
    if ghost is None:
        ghost = numbers == 0
    keep = np.nonzero(~np.asarray(ghost, dtype=bool))[0]
    return numbers[keep], np.ascontiguousarray(positions[keep]), keep, numbers.shape[0]


def _model(numbers, positions):
    """A ``DispersionModel`` for the atoms :func:`_geometry` selected."""
    from dftd3.interface import DispersionModel
    return DispersionModel(numbers, positions)


def d3_energy(mol, method, version='d3bj', atm=False, param=None):
    """DFT-D3 dispersion energy of ``mol`` in Hartree.

    Parameters
    ----------
    mol : Mol or tuple
        Molecule, or an ``(atomic_numbers, positions_in_bohr)`` pair. Ghost atoms are left out and atoms
        with an ECP enter as their element (see :func:`_geometry`), so the energy of a counterpoise
        fragment with ghost partners is exactly that of the fragment's real atoms.
    method : str
        Functional name whose D3 parameters to use, e.g. ``'b3lyp5'`` (what Skala 1.1 expects),
        ``'pbe'``, ``'b3lyp'``. Ignored when ``param`` is given.
    version : str
        Damping function; ``'d3bj'`` (Becke-Johnson, the usual choice and Skala's) by default.
        See :data:`DAMPING_VERSIONS`.
    atm : bool
        Include the three-body Axilrod-Teller-Muto term. Off by default, which is what Skala's own
        reference calculations use.
    param : dict or None
        Explicit damping parameters (``s6``, ``s8``, ``a1``, ``a2``, ...) instead of looking them up
        by functional name. ``s9`` scales the three-body term and is taken from ``atm`` when absent.

    Returns
    -------
    float
        Dispersion energy in Hartree (negative).
    """
    damping = _damping_param(method, version, atm, param)
    numbers, positions, _, _ = _geometry(mol)
    if numbers.shape[0] == 0:
        return 0.0      # nothing but ghost atoms
    result = _model(numbers, positions).get_dispersion(damping, grad=False)
    return float(np.asarray(result['energy']).item())


def d3_energy_and_gradient(mol, method, version='d3bj', atm=False, param=None):
    """:func:`d3_energy` together with its nuclear gradient.

    Returns
    -------
    energy : float
        Dispersion energy in Hartree.
    gradient : (natm, 3) ndarray
        dE_disp/dR in Hartree/Bohr, ready to be added to the SCF forces. One row per atom of ``mol``;
        the rows of ghost atoms are zero.
    """
    damping = _damping_param(method, version, atm, param)
    numbers, positions, keep, natm = _geometry(mol)
    gradient = np.zeros((natm, 3), dtype=np.float64)
    if numbers.shape[0] == 0:
        return 0.0, gradient      # nothing but ghost atoms
    result = _model(numbers, positions).get_dispersion(damping, grad=True)
    gradient[keep] = np.asarray(result['gradient'], dtype=np.float64).reshape(-1, 3)
    return float(np.asarray(result['energy']).item()), gradient


def citation():
    """Citation string for the D3 correction, printed in the SCF output."""
    return _CITATION
