"""
CI-NEB Example: Internal rotation of ethane (C2H6)

    Staggered (60°) --> Eclipsed (0°) --> Staggered (60°)

The rotation barrier around the C-C bond is one of the most fundamental
concepts in organic chemistry. The experimental barrier is ~3.0 kcal/mol
(~0.13 eV). DFT/PBE typically gives ~2.7-3.0 kcal/mol depending on basis.

This is a simple but instructive NEB example because:
- The reaction coordinate (dihedral angle) is well-defined
- The barrier is small but well-reproduced by DFT
- The path is physically intuitive
"""

import os
import numpy as np
from ase import Atoms
from ase.build import molecule
from ase.optimize import BFGS
from ase.mep.neb import NEB
from ase.io import write, read

from pyfock import PyFockCalculator


def get_calculator(directory):
    """Create a RIPERCalculator for ethane."""
    return PyFockCalculator(
        functional="PBE",
        basis="def2-SVP",
        auxbasis="def2-universal-jfit",
        ncores=4,
        DF=True,
        save_ao_values=True,
        sao = True,
        conv_crit=1e-7,
        directory=directory,
    )


def make_ethane_staggered():
    """
    Build staggered ethane using ASE's built-in molecule database.
    ASE's C2H6 is already in the staggered conformation.
    """
    return molecule("C2H6")


def make_ethane_eclipsed():
    """
    Build eclipsed ethane by rotating one CH3 group by 60 degrees
    from the staggered conformation.
    """
    ethane = molecule("C2H6")

    # ASE's C2H6 has atoms ordered as: C0, C1, H2, H3, H4, H5, H6, H7
    # C0 is bonded to H2, H3, H4
    # C1 is bonded to H5, H6, H7
    # The C-C bond is along a specific axis

    positions = ethane.get_positions()

    # Get the C-C bond vector
    c0_pos = positions[0]
    c1_pos = positions[1]
    cc_axis = c1_pos - c0_pos
    cc_axis_norm = cc_axis / np.linalg.norm(cc_axis)

    # Rotate H5, H6, H7 (indices 5, 6, 7) by 60 degrees around C-C axis
    # centered at C1
    angle_rad = np.pi / 3.0  # 60 degrees

    # Rodrigues' rotation formula
    def rotate_around_axis(point, axis, center, angle):
        """Rotate a point around an axis passing through center."""
        p = point - center
        k = axis / np.linalg.norm(axis)
        p_rot = (
            p * np.cos(angle)
            + np.cross(k, p) * np.sin(angle)
            + k * np.dot(k, p) * (1 - np.cos(angle))
        )
        return p_rot + center

    # Rotate the second CH3 group
    for i in [5, 6, 7]:
        positions[i] = rotate_around_axis(
            positions[i], cc_axis_norm, c1_pos, angle_rad
        )

    ethane.set_positions(positions)
    return ethane


def get_dihedral_angles(atoms):
    """
    Calculate H-C-C-H dihedral angles for ethane.
    Returns representative dihedral H2-C0-C1-H5.
    """
    return atoms.get_dihedral(2, 0, 1, 5)


def main():
    # =========================================================================
    # 1. Build reactant (staggered) and product (staggered, rotated 120°)
    # =========================================================================
    # For a full rotation: staggered -> eclipsed -> next staggered
    # We go from one staggered minimum to the next (60° rotation passes
    # through the eclipsed TS at 60° dihedral = 0°/60° depending on convention)

    reactant = make_ethane_staggered()
    product = make_ethane_eclipsed()  # This is actually the other staggered minimum

    # Actually, for NEB we want to go between two minima.
    # Staggered -> Eclipsed -> Staggered (rotated by 120°)
    # But staggered -> eclipsed is minimum -> TS, which is trivial.
    # Better: rotate by 120° to get the next equivalent staggered minimum.

    # Let's redefine: rotate by 120° for the product to get next staggered minimum
    # and the eclipsed geometry (60° rotation) will be found as the TS.

    # Rebuild product as staggered rotated by 120°
    product_120 = molecule("C2H6")
    positions = product_120.get_positions()
    c0_pos = positions[0]
    c1_pos = positions[1]
    cc_axis = c1_pos - c0_pos
    cc_axis_norm = cc_axis / np.linalg.norm(cc_axis)

    def rotate_around_axis(point, axis, center, angle):
        p = point - center
        k = axis / np.linalg.norm(axis)
        p_rot = (
            p * np.cos(angle)
            + np.cross(k, p) * np.sin(angle)
            + k * np.dot(k, p) * (1 - np.cos(angle))
        )
        return p_rot + center

    # Rotate second CH3 group by 120° (2π/3) to reach next staggered minimum
    for i in [5, 6, 7]:
        positions[i] = rotate_around_axis(
            positions[i], cc_axis_norm, c1_pos, 2 * np.pi / 3
        )

    product_120.set_positions(positions)
    product = product_120

    print("Reactant dihedral (H2-C0-C1-H5):", get_dihedral_angles(reactant))
    print("Product dihedral  (H2-C0-C1-H5):", get_dihedral_angles(product))

    # =========================================================================
    # 2. Optimize endpoints
    # =========================================================================
    print()
    print("=" * 60)
    print("Optimizing reactant (staggered ethane)...")
    print("=" * 60)

    reactant_dir = "calculations/ethane_reactant"
    os.makedirs(reactant_dir, exist_ok=True)
    reactant.calc = get_calculator(directory=reactant_dir)
    opt_r = BFGS(reactant, trajectory=os.path.join(reactant_dir, "opt.traj"))
    opt_r.run(fmax=0.05)
    e_reactant = reactant.get_potential_energy()
    print(f"Reactant energy: {e_reactant:.6f} eV")
    print(f"Reactant dihedral: {get_dihedral_angles(reactant):.1f}°")
    print()

    print("=" * 60)
    print("Optimizing product (rotated staggered ethane)...")
    print("=" * 60)

    product_dir = "calculations/ethane_product"
    os.makedirs(product_dir, exist_ok=True)
    product.calc = get_calculator(directory=product_dir)
    opt_p = BFGS(product, trajectory=os.path.join(product_dir, "opt.traj"))
    opt_p.run(fmax=0.05)
    e_product = product.get_potential_energy()
    print(f"Product energy: {e_product:.6f} eV")
    print(f"Product dihedral: {get_dihedral_angles(product):.1f}°")
    print()

    print(f"Energy difference (should be ~0): {abs(e_product - e_reactant):.6f} eV")

    # =========================================================================
    # 3. Set up CI-NEB
    # =========================================================================
    n_images = 7

    print()
    print("=" * 60)
    print(f"Setting up CI-NEB with {n_images} intermediate images...")
    print("=" * 60)

    images = [reactant.copy()]
    for i in range(n_images):
        images.append(reactant.copy())
    images.append(product.copy())

    # Use linear interpolation (for rotation, IDPP might be better)
    neb = NEB(images, climb=True, k=0.5)
    neb.interpolate("idpp")  # IDPP interpolation handles rotations better

    # Assign calculators to ALL images (including endpoints)
    for i, image in enumerate(images):
        img_dir = f"calculations/ethane_neb_image_{i:02d}"
        os.makedirs(img_dir, exist_ok=True)
        image.calc = get_calculator(directory=img_dir)
    # =========================================================================
    # 4. Run CI-NEB
    # =========================================================================
    print("Running CI-NEB optimization...")
    print()

    neb_opt = BFGS(neb, trajectory="calculations/ethane_neb_band.traj")
    neb_opt.run(fmax=0.05)

    # =========================================================================
    # 5. Analyze results
    # =========================================================================
    print()
    print("=" * 60)
    print("CI-NEB Results: Ethane Rotation")
    print("=" * 60)

    energies = [image.get_potential_energy() for image in images]
    e_ref = energies[0]

    print(f"\n{'Image':>6s} {'Energy (eV)':>14s} {'Rel. (eV)':>12s} "
          f"{'Rel. (kcal/mol)':>16s} {'Dihedral (°)':>14s}")
    print("-" * 70)

    for i, (image, e) in enumerate(zip(images, energies)):
        de = e - e_ref
        de_kcal = de * 23.0609
        dih = get_dihedral_angles(image)
        label = ""
        if i == 0:
            label = " <- reactant"
        elif i == len(energies) - 1:
            label = " <- product"
        elif e == max(energies):
            label = " <- TS (eclipsed)"
        print(f"{i:>6d} {e:>14.6f} {de:>12.6f} {de_kcal:>16.4f} {dih:>14.1f}{label}")

    barrier = max(energies) - e_ref
    print(f"\nRotation barrier: {barrier:.4f} eV  ({barrier * 23.0609:.2f} kcal/mol)")
    print(f"Expected:         ~0.13 eV  (~3.0 kcal/mol)")
    print(f"Reaction energy:  {energies[-1] - energies[0]:.6f} eV (should be ~0)")

    # =========================================================================
    # 6. TS geometry analysis
    # =========================================================================
    ts_index = energies.index(max(energies))
    ts_image = images[ts_index]

    print(f"\nTransition state geometry (image {ts_index}):")
    print(f"  H-C-C-H dihedral: {get_dihedral_angles(ts_image):.1f}°")
    print(f"  (Eclipsed conformation expected: ~0° or ~120°)")

    d_cc = ts_image.get_distance(0, 1)
    print(f"  C-C bond length: {d_cc:.4f} Å")

    # Save the path
    write("calculations/ethane_neb_path.extxyz", images)
    print("\nNEB path saved to 'calculations/ethane_neb_path.extxyz'")


if __name__ == "__main__":
    main()
