"""
Cyclic Boundary Condition (CBC) Ring Generator for PyFock
Emulates PBC by bending a 1D periodic system into a ring.

Handles multi-atom unit cells with arbitrary transverse positions,
including carbon nanotubes where atoms sit on a cylindrical surface.

Usage:
    import PBC_ring
    ring_mol = PBC_ring.ring(unit_mol, N=10, periodicity=2.4595, periodic_dir='z')
    
    basis = Basis(ring_mol, {'all': Basis.load(mol=ring_mol, basis_name='def2-SVP')})
    auxbasis = Basis(ring_mol, {'all': Basis.load(mol=ring_mol, basis_name='def2-universal-jfit')})
    dft_obj = DFT(ring_mol, basis, auxbasis, xc=[1, 7])
    energy = dft_obj.scf()
    
    energies = PBC_ring.convergence_study(unit_mol, ring_sizes=[8,10,12,15], periodicity=2.4595)
    tdl_energy = PBC_ring.extrapolate_tdl(energies)
"""

import numpy as np
import math
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional
from pyfock import Mol, Basis, DFT


def ring(mol: Mol, N: int, periodicity: float, periodic_dir: str = 'z',
         output_xyz: bool = True, xyz_filename: Optional[str] = None) -> Mol:
    """
    Create a ring supercell from a 1D periodic unit cell.
    
    The periodic direction is bent into a circle of circumference N * periodicity.
    The two transverse directions are treated as a 2D cross-section that is
    carried along the ring, properly accounting for radial offset from the
    bending axis. This correctly handles systems like carbon nanotubes where
    atoms have non-trivial positions in BOTH transverse directions.
    
    The bending places the ring in a plane. For each atom:
      - Its fractional position along the periodic direction sets its angle 
        on the ring.
      - Its 2D transverse offset (distance from the periodic axis and angle 
        around it) is preserved: the radial part shifts the atom inward/outward 
        from the ring radius, and the angular part rotates it around the local 
        ring tangent.
    
    Args:
        mol: PyFock Mol object containing the unit cell
        N: Number of unit cells to include in the ring
        periodicity: Length of unit cell along periodic direction (Angstrom)
        periodic_dir: Direction of periodicity ('x', 'y', or 'z'). Default: 'z'
        output_xyz: Whether to save XYZ file of the ring. Default: True
        xyz_filename: Custom filename for XYZ output. If None, auto-generates name
        
    Returns:
        Mol: New PyFock Mol object containing the ring structure
        
    Example:
        >>> # Carbon nanotube with z-periodicity
        >>> unit_mol = Mol(coordfile='cnt_unit.xyz')
        >>> ring_mol = ring(unit_mol, N=20, periodicity=2.4595, periodic_dir='z')
    """
    
    # Determine periodic direction index
    dir_map = {'x': 0, 'y': 1, 'z': 2}
    if periodic_dir.lower() not in dir_map:
        raise ValueError("periodic_dir must be 'x', 'y', or 'z'")
    periodic_idx = dir_map[periodic_dir.lower()]
    
    # The two transverse indices (in order)
    transverse_idx = [i for i in range(3) if i != periodic_idx]
    t1, t2 = transverse_idx  # e.g. for z-periodic: t1=0 (x), t2=1 (y)
    
    # Calculate ring radius from circumference = N * periodicity
    circumference = N * periodicity
    R = circumference / (2.0 * math.pi)
    
    # ----------------------------------------------------------------
    # Find the centroid of the unit cell in the transverse plane.
    # This is the "axis" of the tube / wire that we bend into the ring.
    # For a well-centered structure this will be near (0, 0), but we
    # compute it to be robust.
    # ----------------------------------------------------------------
    coords = np.array([[a[1], a[2], a[3]] for a in mol.atoms])
    center_t1 = np.mean(coords[:, t1])
    center_t2 = np.mean(coords[:, t2])
    
    # Build ring coordinates
    ring_atoms = []
    
    for unit_idx in range(N):
        # Base angle for this replica
        unit_angle = 2.0 * math.pi * unit_idx / N
        
        for atom in mol.atoms:
            symbol = atom[0]
            orig = np.array([atom[1], atom[2], atom[3]])
            
            # --- position along periodic direction within the unit cell ---
            s = orig[periodic_idx]
            
            # Total angle on the ring for this atom
            theta = unit_angle + 2.0 * math.pi * s / circumference
            
            # --- transverse displacement relative to tube axis ---
            dt1 = orig[t1] - center_t1
            dt2 = orig[t2] - center_t2
            
            # Decompose transverse offset into radial (outward from ring
            # centre) and "axial-of-tube" (perpendicular to ring plane)
            # components.  We define a local frame at angle theta:
            #
            #   e_radial  = (cos theta, sin theta, 0)   [in ring plane]
            #   e_axial   = (0, 0, 1)                   [out of ring plane]
            #
            # The transverse cross-section of the original cell lives in the
            # (t1, t2) plane.  We map:
            #   dt1 -> radial   (outward from ring centre)
            #   dt2 -> out-of-plane (perpendicular to ring plane)
            #
            # This is a convention choice: for a nanotube with z-periodic,
            # the tube cross-section is in the xy-plane.  Atom at (x,y)
            # relative to tube axis has radial distance r = sqrt(x²+y²) and
            # azimuthal angle φ = atan2(y, x).
            #
            # When we bend z into a ring in, say, the XZ-plane, we want to
            # preserve the full cylindrical structure.  So we keep the local
            # polar coordinates (r, φ) of each atom in the cross-section and
            # orient them relative to the outward radial direction of the ring.
            # ----------------------------------------------------------------
            
            r_local = math.sqrt(dt1**2 + dt2**2)       # distance from tube axis
            phi_local = math.atan2(dt2, dt1)            # angle in cross-section
            
            # In the ring frame the outward radial direction at angle theta
            # lies in the ring plane.  We define the ring to lie in the
            # (X, Z) plane with Y as the out-of-plane direction.
            #
            # Ring backbone point:
            #   P = (R cos θ,  0,  R sin θ)
            #
            # Local radial outward unit vector (in ring plane):
            #   e_r = (cos θ,  0,  sin θ)
            #
            # Out-of-plane unit vector:
            #   e_y = (0, 1, 0)
            #
            # Atom position:
            #   P + r_local * cos(φ) * e_r  +  r_local * sin(φ) * e_y
            
            # Effective radial distance from ring axis
            rho = R + r_local * math.cos(phi_local)
            out_of_plane = r_local * math.sin(phi_local)
            
            # Place ring in the (X, Z) plane, with Y out-of-plane
            X = rho * math.cos(theta)
            Y = out_of_plane + center_t2   # shift back if centroid wasn't at origin
            Z = rho * math.sin(theta)
            
            # Now map (X, Y, Z) back to the original axis labelling so that
            # the output coordinates are easy to interpret.
            # Ring plane axes came from (t1, periodic_idx) and out-of-plane
            # is t2.  But we must be careful: we defined X,Z as ring-plane
            # and Y as out-of-plane.  Let's build the final coords directly.
            #
            # Actually, let's use a cleaner mapping that doesn't depend on
            # which direction is periodic.  We'll construct final_coords as
            # a 3-vector.
            
            final = np.zeros(3)
            # Ring plane spans two of the three Cartesian directions:
            # we use the periodic direction and the first transverse direction
            # for the ring plane, and the second transverse direction for
            # out-of-plane.  This keeps the output intuitive.
            
            # ring plane axis 1 -> original t1 axis
            # ring plane axis 2 -> original periodic axis
            # out-of-plane     -> original t2 axis
            
            final[t1] = rho * math.cos(theta)
            final[periodic_idx] = rho * math.sin(theta)
            final[t2] = out_of_plane + center_t2
            
            ring_atoms.append([symbol, final[0], final[1], final[2]])
    
    # Create new Mol object for ring
    ring_mol = Mol(atoms=ring_atoms)
    ring_mol.label = f"Ring_{N}units_R{R:.2f}A"
    
    # Save XYZ file if requested
    if output_xyz:
        if xyz_filename is None:
            xyz_filename = f"ring_{N}_units"
        ring_mol.exportXYZ(xyz_filename,
                label=f"Ring with {N} units, R={R:.3f} A, "
                      f"circumference={circumference:.3f} A")
        print(f"Saved ring structure: {xyz_filename}")
    
    print(f"Created ring: {N} units, {len(ring_atoms)} atoms, "
          f"radius={R:.3f} A, circumference={circumference:.3f} A")
    return ring_mol


def ring_preserve_bonds(mol: Mol, N: int, periodicity: float, target_radius: float,
                       periodic_dir: str = 'x', output_xyz: bool = True, 
                       xyz_filename: Optional[str] = None) -> Mol:
    """
    Create a ring supercell that preserves interatomic spacing from the unit cell.
    
    This version prioritizes maintaining correct bond lengths over fitting exactly
    around a complete circle. The ring may span less than or more than 2π radians.
    
    Args:
        mol: PyFock Mol object containing the unit cell
        N: Number of unit cells to include in the ring
        periodicity: Length of unit cell along periodic direction (Angstrom)
        target_radius: Desired radius for the ring (Angstrom)
        periodic_dir: Direction of periodicity ('x', 'y', or 'z'). Default: 'x'
        output_xyz: Whether to save XYZ file of the ring. Default: True
        xyz_filename: Custom filename for XYZ output. If None, auto-generates name
        
    Returns:
        Mol: New PyFock Mol object containing the ring structure
        
    Example:
        >>> unit_mol = Mol(coordfile='lih_unit.xyz')  # LiH unit cell, 3.2 Å period
        >>> # Create ring with 5 Å radius, preserving bond lengths
        >>> ring_mol = ring_preserve_bonds(unit_mol, N=10, periodicity=3.2, target_radius=5.0)
        >>> # Bonds remain 3.2 Å apart, but ring spans ~6.4 radians (not full 2π circle)
    """
    
    # Determine periodic direction index
    dir_map = {'x': 0, 'y': 1, 'z': 2}
    if periodic_dir.lower() not in dir_map:
        raise ValueError("periodic_dir must be 'x', 'y', or 'z'")
    periodic_idx = dir_map[periodic_dir.lower()]
    
    # Calculate angular spacing to preserve bond lengths
    angular_spacing_per_unit = periodicity / target_radius  # radians per unit cell
    total_angle_spanned = N * angular_spacing_per_unit
    
    # Provide feedback about circle closure
    angle_deficit = 2.0 * math.pi - total_angle_spanned
    if abs(angle_deficit) > 0.1:  # More than ~6 degrees off
        if angle_deficit > 0:
            print(f"Note: Ring spans {total_angle_spanned:.3f} rad ({total_angle_spanned*180/math.pi:.1f}°)")
            print(f"      Gap of {angle_deficit:.3f} rad ({angle_deficit*180/math.pi:.1f}°) to complete circle")
        else:
            print(f"Note: Ring spans {total_angle_spanned:.3f} rad ({total_angle_spanned*180/math.pi:.1f}°)")
            print(f"      Overlaps by {-angle_deficit:.3f} rad ({-angle_deficit*180/math.pi:.1f}°) beyond full circle")
    else:
        print(f"Ring nearly closes: {total_angle_spanned:.3f} rad (~2π)")
    
    # Build ring coordinates
    ring_atoms = []
    
    for unit_idx in range(N):
        # Angular position for this unit (preserves spacing)
        unit_angle = angular_spacing_per_unit * unit_idx
        
        # Add all atoms from this unit cell
        for atom in mol.atoms:
            symbol = atom[0]
            orig_coords = np.array([atom[1], atom[2], atom[3]])
            
            # Position along periodic direction within unit cell
            periodic_pos = orig_coords[periodic_idx]
            
            # Total angle for this atom (includes intra-unit-cell position)
            atom_angle = unit_angle + (periodic_pos / target_radius)
            
            # Ring coordinates (place periodic direction in appropriate plane)
            if periodic_idx == 0:  # x-direction periodic
                x_ring = target_radius * math.cos(atom_angle)
                y_ring = target_radius * math.sin(atom_angle)
                z_ring = orig_coords[2]
            elif periodic_idx == 1:  # y-direction periodic  
                x_ring = orig_coords[0]
                y_ring = target_radius * math.cos(atom_angle)
                z_ring = target_radius * math.sin(atom_angle)
            else:  # z-direction periodic
                x_ring = target_radius * math.cos(atom_angle)
                y_ring = orig_coords[1]  
                z_ring = target_radius * math.sin(atom_angle)
            
            ring_atoms.append([symbol, x_ring, y_ring, z_ring])
    
    # Calculate actual circumference spanned
    actual_circumference = total_angle_spanned * target_radius
    
    # Create new Mol object for ring
    ring_mol = Mol(atoms=ring_atoms)
    ring_mol.label = f"Ring_{N}units_R{target_radius:.2f}A_bondpreserved"
    
    # Save XYZ file if requested
    if output_xyz:
        if xyz_filename is None:
            xyz_filename = f"ring_{N}units_R{target_radius:.1f}A_bonds_preserved.xyz"
        
        ring_mol.exportXYZ(xyz_filename, 
                label=f"Bond-preserving ring: {N} units, R={target_radius:.3f} A, "
                      f"span={total_angle_spanned:.3f} rad, arc_length={actual_circumference:.3f} A")
        print(f"Saved ring structure: {xyz_filename}")
    
    print(f"Created bond-preserving ring: {N} units, {len(ring_atoms)} atoms")
    print(f"  Radius: {target_radius:.3f} A")
    print(f"  Arc length: {actual_circumference:.3f} A (vs {N*periodicity:.3f} A linear)")
    print(f"  Bond spacing preserved: {periodicity:.3f} A")
    
    return ring_mol


def suggest_radius_for_closure(N: int, periodicity: float) -> float:
    """
    Suggest a radius that would give near-perfect ring closure.
    
    Args:
        N: Number of unit cells
        periodicity: Unit cell length
        
    Returns:
        float: Suggested radius for ~perfect circle closure
    """
    ideal_radius = (N * periodicity) / (2.0 * math.pi)
    return ideal_radius


def ring_with_closure_optimization(mol: Mol, N: int, periodicity: float, 
                                 target_radius: Optional[float] = None,
                                 optimize_closure: bool = True, 
                                 periodic_dir: str = 'x',
                                 output_xyz: bool = True,
                                 xyz_filename: Optional[str] = None) -> Mol:
    """
    Create a ring with option to optimize for perfect closure or preserve bonds.
    
    Args:
        mol: PyFock Mol object containing the unit cell
        N: Number of unit cells
        periodicity: Unit cell length (Angstrom)
        target_radius: Desired radius. If None and optimize_closure=True, calculates optimal
        optimize_closure: If True and target_radius=None, optimizes for perfect ring closure
        periodic_dir: Direction of periodicity ('x', 'y', or 'z')
        output_xyz: Whether to save XYZ file
        xyz_filename: Custom filename for XYZ output
        
    Returns:
        Mol: Ring structure
    """
    
    if target_radius is None:
        if optimize_closure:
            target_radius = suggest_radius_for_closure(N, periodicity)
            print(f"Using closure-optimized radius: {target_radius:.3f} A")
        else:
            raise ValueError("Must provide target_radius if optimize_closure=False")
    
    return ring_preserve_bonds(mol, N, periodicity, target_radius, 
                             periodic_dir, output_xyz, xyz_filename)
