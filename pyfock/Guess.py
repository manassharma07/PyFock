"""
Initial density-matrix guesses for the SCF.

SANO: Superposition of Atomic Natural-Orbital densities
--------------------------------------------------------
The SANO guess is PyFock's analogue of the ``minao`` guess of PySCF. The density of every
atom is written as a *diagonal* matrix in the ANO-RCC-MB minimal basis of Roos and
co-workers: the contracted functions of that basis are the natural orbitals of the free
atom, so two electrons go into every closed shell and the electrons of the open shell are
spread evenly over its 2l+1 orbitals. That is the spherically averaged, spin-averaged
atomic density, and no atomic SCF is required to obtain it. Every atomic natural orbital
``a`` is then projected onto the basis of the calculation and the guess density is the
superposition of the occupied projected orbitals::

    c_a = S^{-1} S_cross a,    D = sum_a occ_a f_a^2 c_a c_a^T

with ``S`` the overlap of the calculation basis, ``S_cross`` its overlap with the minimal
basis and ``f_a`` a renormalization factor. Without it (``renormalize=False``) this is
exactly the projected density ``P D_min P^T``. With it (default)
the 2l+1 components of every atomic shell share one factor that restores the norm of the
shell, ``f^2 = (2l+1) / sum_m c_m^T S c_m``, so the guess keeps the electron count of every
atom and stays rotationally invariant. The renormalization matters for atoms carrying an
effective core potential, whose nodeless valence functions can only represent about 93-97 %
of the all-electron valence natural orbitals, and is a no-op (0.1 %) for all-electron atoms.
For ECP atoms the core shells replaced by the potential are left empty.

The guess density is returned in the Cartesian (CAO) representation used inside the SCF.
It is symmetric and positive semidefinite, but not idempotent.
"""
import time

import numpy as np
import scipy.linalg

from pyfock.Data import Data
from pyfock.Basis import Basis
import pyfock.Integrals as Integrals


SANO_MINIMAL_BASIS = 'ano-rcc-mb'
"""Name of the shipped minimal atomic-natural-orbital basis used by the SANO guess."""

SANO_MAX_Z = 96
"""ANO-RCC(-MB) is available from H to Cm."""


# ---------------------------------------------------------------------------------------
# References
# ---------------------------------------------------------------------------------------
_REF_SAD_ALMLOF = ('J. Almlöf, K. Faegri Jr., K. Korsell, "Principles for a direct SCF approach to LCAO-MO '
                   'ab-initio calculations", J. Comput. Chem. 3, 385 (1982). doi:10.1002/jcc.540030314')
_REF_SAD_VANLENTHE = ('J. H. Van Lenthe, R. Zwaans, H. J. J. Van Dam, M. F. Guest, "Starting SCF calculations by '
                      'superposition of atomic densities", J. Comput. Chem. 27, 926 (2006). doi:10.1002/jcc.20393')
_REF_ANO_WIDMARK = ('P.-O. Widmark, P.-Å. Malmqvist, B. O. Roos, "Density matrix averaged atomic natural orbital (ANO) '
                    'basis sets for correlated molecular wave functions", Theor. Chim. Acta 77, 291 (1990). '
                    'doi:10.1007/BF01120130')
_REF_ANORCC_ALKALI = ('V. Veryazov, P.-O. Widmark, B. O. Roos, "Relativistic atomic natural orbital type basis sets for the '
                      'alkaline and alkaline-earth atoms ...", Theor. Chem. Acc. 111, 345 (2004). doi:10.1007/s00214-003-0537-0')
_REF_ANORCC_MAIN = ('B. O. Roos, R. Lindh, P.-Å. Malmqvist, V. Veryazov, P.-O. Widmark, "Main group atoms and dimers studied '
                    'with a new relativistic ANO basis set", J. Phys. Chem. A 108, 2851 (2004). doi:10.1021/jp031064+')
_REF_ANORCC_TM = ('B. O. Roos, R. Lindh, P.-Å. Malmqvist, V. Veryazov, P.-O. Widmark, "New relativistic ANO basis sets for '
                  'transition metal atoms", J. Phys. Chem. A 109, 6575 (2005). doi:10.1021/jp0581126')
_REF_ANORCC_LN = ('B. O. Roos, R. Lindh, P.-Å. Malmqvist, V. Veryazov, P.-O. Widmark, A. C. Borin, "New relativistic atomic '
                  'natural orbital basis sets for lanthanide atoms ...", J. Phys. Chem. A 112, 11431 (2008). doi:10.1021/jp803213j')
_REF_ANORCC_AN = ('B. O. Roos, R. Lindh, P.-Å. Malmqvist, V. Veryazov, P.-O. Widmark, "New relativistic ANO basis sets for '
                  'actinide atoms", Chem. Phys. Lett. 409, 295 (2005). doi:10.1016/j.cplett.2005.05.011')
_REF_BSE = ('B. P. Pritchard, D. Altarawy, B. Didier, T. D. Gibson, T. L. Windus, "A new Basis Set Exchange: an open, '
            'up-to-date resource for the molecular sciences community", J. Chem. Inf. Model. 59, 4814 (2019). '
            'doi:10.1021/acs.jcim.9b00725')
_REF_PYSCF = ('Q. Sun et al., "Recent developments in the PySCF program package", J. Chem. Phys. 153, 024109 (2020). '
              'doi:10.1063/5.0006074 (the ANO projection scheme of its "minao" guess is followed here)')

# Element ranges of the individual ANO-RCC papers (from the Basis Set Exchange reference file).
_ANORCC_PAPER_RANGES = (
    (_REF_ANORCC_ALKALI, {3, 4, 11, 12, 19, 20, 37, 38, 55, 56, 87, 88}),
    (_REF_ANORCC_MAIN, set(range(5, 11)) | set(range(13, 19)) | set(range(31, 37)) | set(range(49, 55)) | set(range(81, 87))),
    (_REF_ANORCC_TM, set(range(21, 31)) | set(range(39, 49)) | set(range(72, 81))),
    (_REF_ANORCC_LN, set(range(57, 72))),
    (_REF_ANORCC_AN, set(range(89, 97))),
)


def sano_references(nuclear_charges=None):
    """
    References for the SANO guess.

    Parameters
    ----------
    nuclear_charges : iterable of int, optional
        Nuclear charges of the atoms in the calculation. When given, only the ANO-RCC
        papers covering those elements are listed; otherwise all of them are.

    Returns
    -------
    list of str
    """
    refs = [_REF_SAD_ALMLOF, _REF_SAD_VANLENTHE, _REF_ANO_WIDMARK]
    if nuclear_charges is None:
        refs += [ref for ref, _ in _ANORCC_PAPER_RANGES]
    else:
        present = set(int(z) for z in nuclear_charges)
        refs += [ref for ref, elements in _ANORCC_PAPER_RANGES if present & elements]
    return refs


def sano_citation_text(nuclear_charges=None, indent='  '):
    """Human-readable, numbered citation block for the SANO guess."""
    lines = [indent + 'References for the SANO initial guess (please cite):']
    for i, ref in enumerate(sano_references(nuclear_charges), start=1):
        lines.append(indent + ' [%d] %s' % (i, ref))
    return '\n'.join(lines)


# ---------------------------------------------------------------------------------------
# Occupations
# ---------------------------------------------------------------------------------------
def shell_occupations(Z, ncore=0, configuration=None):
    """
    Spherically averaged shell occupations of a free atom.

    Parameters
    ----------
    Z : int
        Nuclear charge of the (all-electron) atom.
    ncore : int, optional
        Number of core electrons replaced by an ECP; the corresponding innermost shells get
        occupation zero.
    configuration : sequence, optional
        Electrons per angular momentum ``[s, p, d, f]`` indexed by Z; defaults to
        ``Data.ATOMIC_CONFIGURATION_NRSRHF``.

    Returns
    -------
    dict
        ``{l: [occupation per orbital of the 1st shell of that l, 2nd shell, ...]}`` in
        order of increasing principal quantum number. Closed shells have occupation 2,
        the open shell ``n_open / (2l+1)``, ECP core shells 0.

    Raises
    ------
    ValueError
        For elements beyond Cm or an ECP core size without a tabulated shell structure.
    """
    if configuration is None:
        configuration = Data.ATOMIC_CONFIGURATION_NRSRHF
    if Z < 1 or Z > SANO_MAX_Z:
        raise ValueError('SANO guess: no atomic natural orbitals for Z = %d (available for Z = 1..%d).' % (Z, SANO_MAX_Z))
    if ncore not in Data.ECP_CORE_SHELLS:
        raise ValueError('SANO guess: no core-shell structure tabulated for an ECP with %d core electrons.' % ncore)
    conf = configuration[Z]
    core = Data.ECP_CORE_SHELLS[ncore]
    occ = {}
    for l in range(4):
        nelec = conf[l]
        degen = 2 * l + 1
        ndocc = nelec // (2 * degen)
        nopen = nelec - 2 * degen * ndocc
        shells = [2.0] * ndocc + ([nopen / degen] if nopen > 0 else [])
        if core[l] > len(shells):
            # The ECP swallows a partially occupied shell (does not occur for standard ECPs).
            shells = [0.0] * len(shells)
        else:
            for j in range(core[l]):
                shells[j] = 0.0
        occ[l] = shells
    return occ


def _shell_atoms(basis):
    """Atom index of every shell of ``basis``."""
    return np.asarray([basis.bfs_atoms[o] for o in basis.shell_bfs_offset], dtype=np.int64)


def _atom_shells_by_l(basis, iatom, shell_atoms=None):
    """``{l: [shell indices]}`` of atom ``iatom`` in the order of the basis-set file."""
    if shell_atoms is None:
        shell_atoms = _shell_atoms(basis)
    out = {}
    for k in np.nonzero(shell_atoms == iatom)[0]:
        out.setdefault(int(basis.shells[k]) - 1, []).append(int(k))
    return out


def _shell_occupation_vector(basis, occ_by_atom, shell_atoms=None):
    """Occupation of every shell of ``basis`` (per orbital) from ``{iatom: {l: [...]}}``.

    Within each angular momentum the shells of an atom are filled in file order: the first
    shell gets the occupation of the innermost (most tightly bound) atomic orbital of that l.
    Shells beyond the tabulated ones stay unoccupied."""
    if shell_atoms is None:
        shell_atoms = _shell_atoms(basis)
    occ_shell = np.zeros(basis.nshells)
    for iatom, occ_l in occ_by_atom.items():
        for l, shells in _atom_shells_by_l(basis, iatom, shell_atoms).items():
            occs = occ_l.get(l, [])
            for j, k in enumerate(shells[:len(occs)]):
                occ_shell[k] = occs[j]
    return occ_shell


def _spherical_occupations(basis, occ_shell):
    """Expand per-shell occupations to one entry per real spherical function."""
    degen = 2 * (np.asarray(basis.shells, dtype=np.int64) - 1) + 1
    return np.repeat(occ_shell, degen)


def _solve_spd(S, B):
    """Solve S X = B for a symmetric positive (semi)definite overlap matrix."""
    try:
        c = scipy.linalg.cho_factor(S, lower=True, check_finite=False)
        return scipy.linalg.cho_solve(c, B, check_finite=False)
    except scipy.linalg.LinAlgError:
        # (Nearly) linearly dependent basis: pseudo-inverse with an eigenvalue cutoff.
        w, V = np.linalg.eigh(S)
        keep = w > 1e-10 * w.max()
        return (V[:, keep] / w[keep]) @ (V[:, keep].T @ B)


def _cartesian_to_spherical_columns(P, basis):
    """
    Given coefficients ``P`` (n x ncart) of the Cartesian functions of ``basis``, return the
    coefficients (n x nsph) of its real spherical functions, shell by shell:
    ``Q[:, sph_k] = P[:, cart_k] @ T_l^T`` with ``T_l`` the Cartesian -> spherical transform.
    """
    shells_l = np.asarray(basis.shells, dtype=np.int64) - 1
    nsph = int(np.sum(2 * shells_l + 1))
    Q = np.empty((P.shape[0], nsph))
    isph = 0
    for k in range(basis.nshells):
        l = int(shells_l[k])
        o = int(basis.shell_bfs_offset[k])
        n = int(basis.bfs_nbfshell[k])
        if l <= 1:
            Q[:, isph:isph + n] = P[:, o:o + n]     # s and p: identical in both representations
        else:
            Q[:, isph:isph + 2 * l + 1] = P[:, o:o + n] @ Basis.cart2sph(l).T
        isph += 2 * l + 1
    return Q


# ---------------------------------------------------------------------------------------
# SANO guess
# ---------------------------------------------------------------------------------------
def load_minimal_basis(mol, basis_name=SANO_MINIMAL_BASIS):
    """Minimal atomic-natural-orbital basis of ``mol`` (one block per basis species)."""
    species = []
    for i in range(mol.natoms):
        if mol.basisSpecies[i] not in species:
            species.append(mol.basisSpecies[i])
    text = ''.join(Basis.load(atom=sp, basis_name=basis_name, quiet=True) for sp in species)
    return Basis(mol, {'all': text})


def sano_dmat(mol, basis, S=None, configuration=None, renormalize=True):
    """
    Superposition of atomic natural-orbital densities (SANO) guess density matrix.

    Parameters
    ----------
    mol : Mol
        Molecule. Ghost atoms contribute no density; ECP atoms have their core shells empty.
    basis : Basis
        Basis of the calculation (Cartesian representation).
    S : ndarray, optional
        Overlap matrix of ``basis`` in the CAO representation; computed when not given.
    configuration : sequence, optional
        Atomic configurations, see :func:`shell_occupations`.
    renormalize : bool, optional
        Renormalize the projected atomic natural orbitals shell by shell in the calculation
        basis (default), which gives exactly the electron count of the molecule. With
        ``False`` the plain projected density of PySCF's ``minao`` guess is returned.

    Returns
    -------
    dmat : ndarray, shape (nbf, nbf)
        Guess density matrix in the CAO representation.
    info : dict
        ``nelectrons`` (electrons of the molecule), ``nelectrons_guess`` (trace of DS),
        ``nelectrons_projected`` (trace of DS before the renormalization, i.e. the part of
        the atomic orbitals the calculation basis can represent), ``species``,
        ``nuclear_charges``, ``minimal_basis``, ``renormalized``, ``time`` in seconds.

    Raises
    ------
    ValueError
        For elements beyond Cm or an ECP core size that is not tabulated.
    """
    t_start = time.perf_counter()
    natoms = mol.natoms
    Z_all = list(getattr(mol, 'Zcharges_all_electron', mol.Zcharges))
    ncore = list(getattr(mol, 'ecp_core_electrons', [0] * natoms))

    occ_by_atom = {}
    species, nuclear_charges = [], []
    for i in range(natoms):
        if mol.atomicSpecies[i] == 'Ghost' or Z_all[i] == 0:
            continue
        if Z_all[i] > SANO_MAX_Z:
            raise ValueError('SANO guess: element %s (Z = %d) is beyond Cm, no atomic natural orbitals available.'
                             % (mol.atomicSpecies[i], Z_all[i]))
        occ_by_atom[i] = shell_occupations(int(Z_all[i]), int(ncore[i]), configuration)
        nuclear_charges.append(int(Z_all[i]))
        if mol.atomicSpecies[i] not in species:
            species.append(mol.atomicSpecies[i])

    nbf = basis.bfs_nao
    if S is None:
        S = Integrals.overlap_mat_symm(basis)
    S = np.asarray(S, dtype=np.float64)

    mb = load_minimal_basis(mol)
    occ = _spherical_occupations(mb, _shell_occupation_vector(mb, occ_by_atom, _shell_atoms(mb)))
    occupied = occ > 0
    degen = 2 * (np.asarray(mb.shells, dtype=np.int64) - 1) + 1
    shell_of_function = np.repeat(np.arange(mb.nshells), degen)[occupied]

    dmat = np.zeros((nbf, nbf))
    nelec_projected = 0.0
    if np.any(occupied):
        # Atomic natural orbitals as orbitals of the calculation basis: c = S^{-1} S_cross a
        S_cross = Integrals.cross_overlap_mat_symm(basis, mb)
        P = _solve_spd(S, S_cross)
        Q = _cartesian_to_spherical_columns(P, mb)[:, occupied]
        occ_q = occ[occupied]
        norms2 = np.einsum('ij,ij->j', Q, S @ Q)
        nelec_projected = float(np.dot(occ_q, norms2))
        if renormalize:
            # one factor per shell: the norm summed over its 2l+1 components is rotationally
            # invariant, the norms of the individual components are not
            shell_norm2 = np.zeros(mb.nshells)
            shell_count = np.zeros(mb.nshells)
            np.add.at(shell_norm2, shell_of_function, norms2)
            np.add.at(shell_count, shell_of_function, 1.0)
            Q = Q * np.sqrt(shell_count[shell_of_function] / shell_norm2[shell_of_function])
        dmat = (Q * occ_q) @ Q.T
        dmat = 0.5 * (dmat + dmat.T)

    info = {
        'nelectrons': int(mol.nelectrons),
        'nelectrons_guess': float(np.einsum('ij,ji->', dmat, S)),
        'nelectrons_projected': nelec_projected,
        'species': species,
        'nuclear_charges': nuclear_charges,
        'minimal_basis': SANO_MINIMAL_BASIS.upper(),
        'renormalized': bool(renormalize),
        'time': time.perf_counter() - t_start,
    }
    return dmat, info


def project_dmat(dmat, basis_from, basis_to, S_to=None):
    """
    Project a density matrix from one basis onto another (both CAO):
    ``D_to = P D_from P^T`` with ``P = S_to^{-1} <to|from>``.

    Useful for starting a calculation from the converged density of a smaller basis or of a
    slightly different geometry. The projected density keeps its electron count only to the
    extent that the source functions lie in the span of the target basis.
    """
    if S_to is None:
        S_to = Integrals.overlap_mat_symm(basis_to)
    S_cross = Integrals.cross_overlap_mat_symm(basis_to, basis_from)
    P = _solve_spd(np.asarray(S_to, dtype=np.float64), S_cross)
    D = (P @ np.asarray(dmat, dtype=np.float64)) @ P.T
    return 0.5 * (D + D.T)
