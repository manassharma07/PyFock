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
from ase.io import read, write
from ase.optimize import BFGS
from ase.mep.neb import NEB

from pyfock import PyFockCalculator


def get_calculator(directory):
    return PyFockCalculator(
        functional="PBE",
        basis="def2-SVP",
        auxbasis="def2-universal-jfit",
        ncores=4,
        DF=True,
        save_ao_values=True,
        sao=True,
        conv_crit=1e-7,
        directory=directory,
    )

def get_dihedral_angles(atoms):
    """
    Calculate H-C-C-H dihedral angles for ethane.
    Returns representative dihedral H2-C0-C1-H5.
    """
    return atoms.get_dihedral(2, 0, 1, 5)


def main():
    # =========================================================================
    # 1. Read reactant and product from XYZ
    # =========================================================================
    reactant = read("ethane_staggered_reactant.xyz")
    product = read("ethane_staggered_product.xyz")

    print("Reactant dihedral (H2-C0-C1-H5):", get_dihedral_angles(reactant))
    print("Product dihedral  (H2-C0-C1-H5):", get_dihedral_angles(product))

    # =========================================================================
    # 2. (Optional but recommended) Optimize endpoints
    # =========================================================================
    print("Optimizing endpoints...")

    reactant_dir = "calculations/reactant"
    product_dir = "calculations/product"
    os.makedirs(reactant_dir, exist_ok=True)
    os.makedirs(product_dir, exist_ok=True)

    
    reactant.calc = get_calculator(reactant_dir)
    product.calc = get_calculator(product_dir)

    print()
    print("=" * 60)
    print("Optimizing reactant (staggered ethane)...")
    print("=" * 60)
    BFGS(reactant, trajectory=f"{reactant_dir}/opt.traj").run(fmax=0.05)
    e_reactant = reactant.get_potential_energy()
    print(f"Reactant energy: {e_reactant:.6f} eV")
    print(f"Reactant dihedral: {get_dihedral_angles(reactant):.1f}°")
    print()

    print("=" * 60)
    print("Optimizing product (rotated staggered ethane)...")
    print("=" * 60)
    BFGS(product, trajectory=f"{product_dir}/opt.traj").run(fmax=0.05)
    e_product = product.get_potential_energy()
    print(f"Product energy: {e_product:.6f} eV")
    print(f"Product dihedral: {get_dihedral_angles(product):.1f}°")
    print()

    print(f"Energy difference (should be ~0): {abs(e_product - e_reactant):.6f} eV")

    # =========================================================================
    # 3. Set up NEB
    # =========================================================================
    n_images = 7  # number of intermediate images
    print()
    print("=" * 60)
    print(f"Setting up CI-NEB with {n_images} intermediate images...")
    print("=" * 60)

    images = [reactant.copy()]
    images += [reactant.copy() for _ in range(n_images)]
    images += [product.copy()]

    neb = NEB(images, climb=True, k=0.5)

    # IDPP interpolation (important for rotations / large rearrangements)
    neb.interpolate(method="idpp")

    # =========================================================================
    # 4. Attach calculators
    # =========================================================================
    for i, image in enumerate(images):
        img_dir = f"calculations/neb_image_{i:02d}"
        os.makedirs(img_dir, exist_ok=True)
        image.calc = get_calculator(img_dir)

    # =========================================================================
    # 5. Run NEB
    # =========================================================================
    print("Running CI-NEB...")

    opt = BFGS(neb, trajectory="calculations/neb.traj")
    opt.run(fmax=0.05)

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