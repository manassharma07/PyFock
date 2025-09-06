"""
Cyclic Boundary Condition (CBC) Ring Generator for PyFock
This is a way to emulate the PBC

Simple interface for creating ring supercells from 1D periodic systems and 
performing systematic convergence studies with extrapolation to thermodynamic limit.

Usage:
    # Create ring supercells
    import PBC_ring
    ring_mol = PBC_ring.ring(unit_mol, N=10, periodicity=2.5)
    
    # Run DFT as usual
    basis = Basis(ring_mol, {'all': Basis.load(mol=ring_mol, basis_name='def2-SVP')})
    auxbasis = Basis(ring_mol, {'all': Basis.load(mol=ring_mol, basis_name='def2-universal-jfit')})
    dft_obj = DFT(ring_mol, basis, auxbasis, xc=[1, 7])
    energy = dft_obj.scf()
    
    # Convergence study utilities
    energies = PBC_ring.convergence_study(unit_mol, ring_sizes=[8,10,12,15], periodicity=2.5)
    tdl_energy = PBC_ring.extrapolate_tdl(energies)
"""

import numpy as np
import math
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional
from pyfock import Mol, Basis, DFT

def ring(mol: Mol, N: int, periodicity: float, periodic_dir: str = 'x', 
         output_xyz: bool = True, xyz_filename: Optional[str] = None) -> Mol:
    """
    Create a ring supercell from a 1D periodic unit cell.
    
    Args:
        mol: PyFock Mol object containing the unit cell
        N: Number of unit cells to include in the ring
        periodicity: Length of unit cell along periodic direction (Angstrom)
        periodic_dir: Direction of periodicity ('x', 'y', or 'z'). Default: 'x'
        output_xyz: Whether to save XYZ file of the ring. Default: False
        xyz_filename: Custom filename for XYZ output. If None, auto-generates name
        
    Returns:
        Mol: New PyFock Mol object containing the ring structure
        
    Example:
        >>> unit_mol = Mol(coordfile='lih_unit.xyz')  # LiH unit cell
        >>> ring_mol = ring(unit_mol, N=10, periodicity=3.2)
        >>> # Now use ring_mol in normal PyFock DFT workflow
    """
    
    # Determine periodic direction index
    dir_map = {'x': 0, 'y': 1, 'z': 2}
    if periodic_dir.lower() not in dir_map:
        raise ValueError("periodic_dir must be 'x', 'y', or 'z'")
    periodic_idx = dir_map[periodic_dir.lower()]
    
    # Calculate ring radius from circumference = N * periodicity
    circumference = N * periodicity
    radius = circumference / (2.0 * math.pi)
    
    # Build ring coordinates
    ring_atoms = []
    
    for unit_idx in range(N):
        # Angular position for this unit
        unit_angle = 2.0 * math.pi * unit_idx / N
        
        # Add all atoms from this unit cell
        for atom in mol.atoms:
            symbol = atom[0]
            orig_coords = np.array([atom[1], atom[2], atom[3]])
            
            # Position along periodic direction within unit cell
            periodic_pos = orig_coords[periodic_idx]
            
            # Total angle for this atom
            atom_angle = unit_angle + (2.0 * math.pi * periodic_pos / periodicity) / N
            
            # Ring coordinates (place periodic direction in xy-plane)
            if periodic_idx == 0:  # x-direction periodic
                x_ring = radius * math.cos(atom_angle)
                y_ring = radius * math.sin(atom_angle)
                z_ring = orig_coords[2]
            elif periodic_idx == 1:  # y-direction periodic  
                x_ring = orig_coords[0]
                y_ring = radius * math.cos(atom_angle)
                z_ring = radius * math.sin(atom_angle)
            else:  # z-direction periodic
                x_ring = radius * math.cos(atom_angle)
                y_ring = orig_coords[1]  
                z_ring = radius * math.sin(atom_angle)
            
            ring_atoms.append([symbol, x_ring, y_ring, z_ring])
    
    # Create new Mol object for ring
    ring_mol = Mol(atoms=ring_atoms)
    ring_mol.label = f"Ring_{N}units_R{radius:.2f}A"
    
    # Save XYZ file if requested
    if output_xyz:
        if xyz_filename is None:
            xyz_filename = f"ring_{N}_units"
        ring_mol.exportXYZ(xyz_filename, 
                label=f"Ring with {N} units, R={radius:.3f} A, circumference={circumference:.3f} A")
        print(f"Saved ring structure: {xyz_filename}")
    
    print(f"Created ring: {N} units, {len(ring_atoms)} atoms, radius={radius:.3f} A")
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
