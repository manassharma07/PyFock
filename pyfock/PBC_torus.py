"""
PBC_torus.py - 2D periodic boundary condition approximation via torus embedding.

For a 2D periodic system with lattice vectors a1, a2 (not necessarily orthogonal),
we map the flat 2D periodic structure onto the surface of a torus in 3D.

The key insight: every atom position can be decomposed as
    r = f1 * a1 + f2 * a2 + d_perp * n_hat
where f1, f2 are fractional coordinates along the lattice vectors,
d_perp is the out-of-plane displacement, and n_hat is the surface normal.

Torus mapping:
    theta = 2*pi * (i1 + f1) / N1    (major angle, from lattice vector a1)
    phi   = 2*pi * (i2 + f2) / N2    (minor angle, from lattice vector a2)

    x = (R1 + (R2 + d_perp) * cos(phi)) * cos(theta)
    y = (R1 + (R2 + d_perp) * cos(phi)) * sin(theta)
    z = (R2 + d_perp) * sin(phi)

where R1, R2 are computed from the actual lattice vector lengths:
    R1 = N1 * |a1| / (2*pi)
    R2 = N2 * |a2| / (2*pi)
"""

import math
import numpy as np
from typing import Optional, List, Tuple
from pyfock import Mol


def _decompose_positions(mol: Mol, a1: np.ndarray, a2: np.ndarray):
    """
    Decompose atom positions into fractional coordinates along a1, a2
    and out-of-plane displacement.

    Given lattice vectors a1, a2 (which define the periodic plane),
    every atom position r is decomposed as:
        r = f1 * a1 + f2 * a2 + d_perp * n_hat

    Args:
        mol: PyFock Mol object
        a1: First lattice vector (3D numpy array, Angstrom)
        a2: Second lattice vector (3D numpy array, Angstrom)

    Returns:
        symbols: List of element symbols
        frac1: Array of fractional coordinates along a1
        frac2: Array of fractional coordinates along a2
        d_perp: Array of out-of-plane displacements (Angstrom)
        n_hat: Unit normal to the a1-a2 plane
    """
    a1 = np.asarray(a1, dtype=float)
    a2 = np.asarray(a2, dtype=float)

    # Surface normal
    n = np.cross(a1, a2)
    n_norm = np.linalg.norm(n)
    if n_norm < 1e-10:
        raise ValueError("Lattice vectors a1 and a2 are parallel!")
    n_hat = n / n_norm

    # Build the 3x3 matrix [a1 | a2 | n_hat] to solve for components
    # r = f1*a1 + f2*a2 + d_perp*n_hat
    # => [a1, a2, n_hat] @ [f1, f2, d_perp]^T = r
    M = np.column_stack([a1, a2, n_hat])
    M_inv = np.linalg.inv(M)

    symbols = []
    frac1 = []
    frac2 = []
    d_perp = []

    for atom in mol.atoms:
        symbols.append(atom[0])
        r = np.array([atom[1], atom[2], atom[3]], dtype=float)
        components = M_inv @ r
        frac1.append(components[0])
        frac2.append(components[1])
        d_perp.append(components[2])

    return symbols, np.array(frac1), np.array(frac2), np.array(d_perp), n_hat


def torus(mol: Mol, N1: int, N2: int,
          a1: List[float], a2: List[float],
          output_xyz: bool = True,
          xyz_filename: Optional[str] = None) -> Mol:
    """
    Create a torus supercell from a 2D periodic unit cell.

    Works with arbitrary (non-orthogonal) lattice vectors.

    The first lattice vector a1 maps to the major ring of the torus,
    the second lattice vector a2 maps to the minor ring (tube).

    Args:
        mol: PyFock Mol object containing the unit cell atoms.
             Atom positions should be in Cartesian coordinates (Angstrom),
             consistent with the lattice vectors a1 and a2.
        N1: Number of unit cells along a1 (major ring). Should be large.
        N2: Number of unit cells along a2 (minor ring/tube). Should be large.
        a1: First lattice vector [x, y, z] in Angstrom.
        a2: Second lattice vector [x, y, z] in Angstrom.
        output_xyz: Whether to save XYZ file. Default: True
        xyz_filename: Custom filename for XYZ output (without .xyz extension).

    Returns:
        Mol: New PyFock Mol object containing the torus structure.

    Notes:
        - R1 > R2 is required to avoid torus self-intersection.
          This means N1*|a1| > N2*|a2|.
        - Both N1 and N2 should be large enough that the curvature
          is small compared to bond lengths.
        - Fractional coordinates of atoms within the unit cell must
          satisfy 0 <= f1 < 1 and 0 <= f2 < 1.

    Example:
        >>> unit_mol = Mol(coordfile='bn_unit.xyz')
        >>> # Hexagonal BN: a1 along x, a2 at 120 degrees
        >>> a = 2.51  # lattice constant
        >>> a1 = [a, 0.0, 0.0]
        >>> a2 = [-a/2, a*math.sqrt(3)/2, 0.0]
        >>> torus_mol = torus(unit_mol, N1=30, N2=15, a1=a1, a2=a2)
    """
    a1 = np.asarray(a1, dtype=float)
    a2 = np.asarray(a2, dtype=float)

    len_a1 = np.linalg.norm(a1)
    len_a2 = np.linalg.norm(a2)

    # Radii from circumference
    R1 = (N1 * len_a1) / (2.0 * math.pi)  # major radius
    R2 = (N2 * len_a2) / (2.0 * math.pi)  # minor radius

    if R2 >= R1:
        import warnings
        warnings.warn(
            f"Minor radius R2={R2:.3f} A >= Major radius R1={R1:.3f} A. "
            f"The torus will self-intersect! Need N1*|a1| > N2*|a2|. "
            f"Currently N1*|a1|={N1*len_a1:.2f}, N2*|a2|={N2*len_a2:.2f}. "
            f"Increase N1 or decrease N2.",
            stacklevel=2
        )

    # Decompose unit cell atom positions into fractional coords
    symbols, frac1, frac2, d_perp, n_hat = _decompose_positions(mol, a1, a2)
    n_atoms_unit = len(symbols)

    # Validate fractional coordinates
    for i in range(n_atoms_unit):
        if frac1[i] < -0.01 or frac1[i] > 1.01 or frac2[i] < -0.01 or frac2[i] > 1.01:
            import warnings
            warnings.warn(
                f"Atom {i} ({symbols[i]}) has fractional coords ({frac1[i]:.4f}, {frac2[i]:.4f}) "
                f"outside [0, 1). Ensure atom positions are within the unit cell "
                f"defined by a1 and a2.",
                stacklevel=2
            )

    # Build torus
    torus_atoms = []

    for i1 in range(N1):
        for i2 in range(N2):
            for ia in range(n_atoms_unit):
                # Total fractional position in the full supercell
                # mapped to angles on the torus
                theta = 2.0 * math.pi * (i1 + frac1[ia]) / N1
                phi = 2.0 * math.pi * (i2 + frac2[ia]) / N2

                # Effective minor radius (shifted by out-of-plane displacement)
                R2_eff = R2 + d_perp[ia]

                # Torus parameterization
                x = (R1 + R2_eff * math.cos(phi)) * math.cos(theta)
                y = (R1 + R2_eff * math.sin(phi)) * math.sin(theta)  # BUG - intentionally NOT this
                z = R2_eff * math.sin(phi)

                # CORRECT torus parameterization:
                x = (R1 + R2_eff * math.cos(phi)) * math.cos(theta)
                y = (R1 + R2_eff * math.cos(phi)) * math.sin(theta)
                z = R2_eff * math.sin(phi)

                torus_atoms.append([symbols[ia], x, y, z])

    # Sanity check: verify nearest-neighbor distances aren't crazy
    _check_distances(torus_atoms, mol, a1, a2, n_atoms_unit)

    # Create Mol object
    torus_mol = Mol(atoms=torus_atoms)
    torus_mol.label = f"Torus_{N1}x{N2}_R1_{R1:.1f}_R2_{R2:.1f}"

    if output_xyz:
        if xyz_filename is None:
            xyz_filename = f"torus_{N1}x{N2}"
        torus_mol.exportXYZ(
            xyz_filename,
            label=(f"Torus {N1}x{N2}, R1={R1:.3f} A, R2={R2:.3f} A, "
                   f"{len(torus_atoms)} atoms, |a1|={len_a1:.3f}, |a2|={len_a2:.3f}")
        )
        print(f"  Saved: {xyz_filename}.xyz")

    print(f"  Torus created: {N1}x{N2} = {N1*N2} cells, {len(torus_atoms)} atoms, "
          f"R1={R1:.3f} A, R2={R2:.3f} A")

    return torus_mol


def _check_distances(torus_atoms, mol, a1, a2, n_atoms_unit):
    """
    Sanity check: compare nearest-neighbor distances in the torus
    against the flat unit cell to detect problems.
    """
    # Find min distance in flat cell (including periodic images)
    a1 = np.asarray(a1)
    a2 = np.asarray(a2)
    min_flat = float('inf')
    for i in range(n_atoms_unit):
        ri = np.array([mol.atoms[i][1], mol.atoms[i][2], mol.atoms[i][3]])
        for j in range(n_atoms_unit):
            rj = np.array([mol.atoms[j][1], mol.atoms[j][2], mol.atoms[j][3]])
            for s1 in [-1, 0, 1]:
                for s2 in [-1, 0, 1]:
                    rj_img = rj + s1 * a1 + s2 * a2
                    d = np.linalg.norm(ri - rj_img)
                    if d > 1e-10 and d < min_flat:
                        min_flat = d

    # Find min distance in torus (sample first few cells)
    n_sample = min(len(torus_atoms), 500)
    min_torus = float('inf')
    coords = np.array([[a[1], a[2], a[3]] for a in torus_atoms[:n_sample]])
    for i in range(min(n_sample, 200)):
        diffs = coords - coords[i]
        dists = np.linalg.norm(diffs, axis=1)
        dists[i] = float('inf')  # skip self
        d_min_i = np.min(dists)
        if d_min_i < min_torus:
            min_torus = d_min_i

    ratio = min_torus / min_flat if min_flat > 0 else 0
    if ratio < 0.8:
        import warnings
        warnings.warn(
            f"Torus min distance ({min_torus:.4f} A) is much shorter than "
            f"flat cell min distance ({min_flat:.4f} A). Ratio={ratio:.3f}. "
            f"Possible overlapping atoms! Increase N1 and/or N2.",
            stacklevel=2
        )
    elif ratio < 0.95:
        print(f"  Warning: Torus bonds distorted ~{(1-ratio)*100:.1f}% "
              f"(flat min={min_flat:.4f}, torus min={min_torus:.4f})")
    else:
        print(f"  Distance check OK: flat min={min_flat:.4f} A, "
              f"torus min={min_torus:.4f} A, ratio={ratio:.4f}")


def estimate_torus_sizes(a1: List[float], a2: List[float],
                         min_radius_factor: float = 5.0,
                         target_sizes: int = 3) -> list:
    """
    Suggest good (N1, N2) pairs for convergence studies.

    Ensures R1 > R2 (no self-intersection) and radii are large
    enough that curvature effects are small.

    Args:
        a1: First lattice vector [x, y, z] in Angstrom
        a2: Second lattice vector [x, y, z] in Angstrom
        min_radius_factor: Minimum R / |a| ratio. Default: 5.0
        target_sizes: Number of size pairs to suggest. Default: 3

    Returns:
        List of (N1, N2) tuples in increasing size
    """
    len_a1 = np.linalg.norm(a1)
    len_a2 = np.linalg.norm(a2)

    # R = N*|a|/(2*pi) >= min_radius_factor * |a|
    # => N >= 2*pi*min_radius_factor
    N_min = int(math.ceil(2.0 * math.pi * min_radius_factor))

    suggestions = []
    scale_factors = np.linspace(1.0, 3.0, target_sizes)

    for sf in scale_factors:
        n2 = max(int(N_min * sf), N_min)
        # Need R1 > R2: N1*|a1| > N2*|a2|
        # With margin: N1*|a1| > 1.5 * N2*|a2|
        n1 = max(int(math.ceil(1.5 * n2 * len_a2 / len_a1)), int(N_min * sf))
        # Also ensure n1 >= N_min
        n1 = max(n1, N_min)
        suggestions.append((n1, n2))

    return suggestions