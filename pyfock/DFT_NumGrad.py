import copy
import numpy as np

from . import Data
from .Basis import Basis
from .DFT import DFT
from .Mol import Mol

# This code is a simple finite difference numerical derivative code for getting DFT forces uising the DFT object
# Right now it is there for testing purposes, as I'm tryubg to implement analytical gradients. 
# TODO:the 3c2e analytical gradients are not working so this can be used in the meanwhile

class DFT_NumGrad:
    """
    Finite-difference nuclear gradients for PyFock DFT calculations.

    This is a simple implementation meant for small molecules and
    only for benchmarking.
    """
    #  The way it works is that It rebuilds displaced single-point DFT jobs from a
    # given converved  DFTobj and uses either central finite differences or may forwatd differences of the total
    # energy with respect to nuclear coordinates.

    def __init__(
        self,
        dft_obj,
        step_size=1.0e-3, # This works out well mostly so should not be changed
        step_unit="bohr",
        method="central", # This requires 6*N calculations and forward requires 3*N calculations (but less accurate)
        use_fixed_grids=True, # Using fixed gris is faster and gives me better performance and accuracy even
        verbose=True,
    ):
        if dft_obj is None:
            print("ERROR: A PyFock DFT object is required.")
            return
        if not getattr(dft_obj, "converged", False):
            print("ERROR: The supplied DFT object must already be converged before ")
            return

        self.dft_obj = dft_obj
        self.step_size = float(step_size)
        self.step_unit = step_unit.lower()
        self.method = method.lower()
        self.use_fixed_grids = use_fixed_grids
        self.verbose = verbose

        if self.step_size <= 0.0:
            raise ValueError("ERROR: step_size must be positive.")
        if self.step_unit not in ("bohr", "angs", "angstrom", "angstroms"):
            raise ValueError("step_unit must be 'bohr' or 'angs'.")
        if self.method not in ("central", "forward"):
            raise ValueError("method must be 'central' or 'forward'.")

    def _step_size_in_angstrom(self):
        if self.step_unit == "bohr":
            return self.step_size / Data.Angs2BohrFactor
        return self.step_size

    def _step_size_in_bohr(self):
        if self.step_unit == "bohr":
            return self.step_size
        return self.step_size * Data.Angs2BohrFactor

    def _atoms_from_coords(self, coords_angstrom):
        atoms = []
        for iatom, symbol in enumerate(self.dft_obj.mol.atomicSpecies):
            x, y, z = coords_angstrom[iatom]
            atoms.append([symbol, float(x), float(y), float(z)])
        return atoms

    def _build_mol_basis(self, coords_angstrom):
        mol = Mol(
            atoms=self._atoms_from_coords(coords_angstrom),
            charge=self.dft_obj.mol.charge,
        )
        basis = Basis(mol, copy.deepcopy(self.dft_obj.basis.basis))

        auxbasis = None
        if self.dft_obj.isDF:
            if self.dft_obj.auxbasis is not None:
                auxbasis = Basis(mol, copy.deepcopy(self.dft_obj.auxbasis.basis))
            else:
                auxbasis = Basis(
                    mol,
                    {"all": Basis.load(mol=mol, basis_name="def2-universal-jfit")},
                )

        return mol, basis, auxbasis

    def _build_displaced_dft(self, coords_angstrom, dmat_guess=None):
        mol, basis, auxbasis = self._build_mol_basis(coords_angstrom)
        displaced_dft = copy.deepcopy(self.dft_obj)
        displaced_dft.mol = mol
        displaced_dft.basis = basis
        displaced_dft.auxbasis = auxbasis
        if not self.use_fixed_grids:
            displaced_dft.grids = None
        displaced_dft.KSmats = []
        displaced_dft.errVecs = []
        displaced_dft.dmat = dmat_guess
        displaced_dft.converged = False
        displaced_dft.scf_energies = []
        displaced_dft.niter = 0
        return displaced_dft

    def _single_point(self, coords_angstrom, dmat_guess=None):
        displaced_dft = self._build_displaced_dft(coords_angstrom, dmat_guess=dmat_guess)
        energy, dmat = displaced_dft.scf()
        return energy, dmat, displaced_dft

    def calculate(self, atom_indices=None):
        """
        Calculate finite-difference gradients and forces.

        Parameters
        ----------
        atom_indices : iterable of int, optional
            Subset of atoms for which the gradient should be evaluated.

        Returns
        -------
        dict
            Dictionary with `energy`, `gradient`, `forces`, `step_size_bohr`,
            and the per-displacement `energies`.
        """
        coords0 = np.array(self.dft_obj.mol.coords, dtype=np.float64, copy=True)
        step_ang = self._step_size_in_angstrom()
        step_bohr = self._step_size_in_bohr()

        if atom_indices is None:
            atom_indices = range(self.dft_obj.mol.natoms)
        atom_indices = list(atom_indices)

        energy0 = self.dft_obj.Total_energy
        dmat_ref = self.dft_obj.dmat
        if energy0 is None or dmat_ref is None:
            raise ValueError(
                "The converged DFT object must contain both Total_energy and dmat."
            )

        gradient = np.zeros_like(coords0)
        displacement_energies = {}

        for iatom in atom_indices:
            for icart, axis_label in enumerate(("x", "y", "z")):
                if self.verbose:
                    print(
                        f"Evaluating finite difference for atom {iatom} axis {axis_label} ..."
                    )

                coords_plus = coords0.copy()
                coords_plus[iatom, icart] += step_ang
                e_plus, dmat_plus, _ = self._single_point(
                    coords_plus, dmat_guess=dmat_ref
                )
                displacement_energies[(iatom, axis_label, "+")] = e_plus

                if self.method == "central":
                    coords_minus = coords0.copy()
                    coords_minus[iatom, icart] -= step_ang
                    e_minus, _, _ = self._single_point(
                        coords_minus, dmat_guess=dmat_plus
                    )
                    displacement_energies[(iatom, axis_label, "-")] = e_minus
                    gradient[iatom, icart] = (e_plus - e_minus) / (2.0 * step_bohr)
                else:
                    gradient[iatom, icart] = (e_plus - energy0) / step_bohr

        forces = -gradient
        return {
            "energy": energy0,
            "gradient": gradient,
            "forces": forces,
            "step_size_bohr": step_bohr,
            "method": self.method,
            "energies": displacement_energies,
        }
