# HF_atoms.py
# Hartree-Fock for isolated atoms (RHF and UHF)
# Uses 4c2e integrals with OS scheme and coul_algo=1

import numpy as np
import scipy
from timeit import default_timer as timer
from opt_einsum import contract
import numba
import os

import pyfock.Mol as Mol
import pyfock.Basis as Basis
import pyfock.Integrals as Integrals
from pyfock import Data

GROUND_STATE_MULTIPLICITY = Data.GROUND_STATE_MULTIPLICITY

class HF_atoms:
    """
    Hartree-Fock calculation for isolated atoms.
    Automatically selects RHF or UHF based on the ground-state multiplicity.
    """

    def __init__(self, mol, basis, conv_crit=1e-7, max_itr=100, ncores=1, multiplicity=None):
        self.mol = mol
        self.basis = basis
        self.conv_crit = conv_crit
        self.max_itr = max_itr
        self.ncores = ncores

        # Determine atom symbol
        self.symbol = mol.atoms[0] if hasattr(mol, 'atoms') else mol.atomNames[0]

        # Determine multiplicity
        if multiplicity is not None:
            self.multiplicity = multiplicity
        else:
            self.multiplicity = GROUND_STATE_MULTIPLICITY.get(self.symbol[0], 1)

        self.nelectrons = mol.nelectrons
        self.nalpha = (self.nelectrons + self.multiplicity - 1) // 2
        self.nbeta = self.nelectrons - self.nalpha
        self.is_uhf = (self.nalpha != self.nbeta)

        # Results
        self.Total_energy = None
        self.converged = False
        self.niter = 0
        self.mo_energies_alpha = None
        self.mo_energies_beta = None
        self.mo_coefficients_alpha = None
        self.mo_coefficients_beta = None
        self.dmat_alpha = None
        self.dmat_beta = None
        self.scf_energies = []

        # DIIS storage
        self._fock_list_a = []
        self._err_list_a = []
        self._fock_list_b = []
        self._err_list_b = []
        self._diis_space = 6

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------
    def _nuclear_rep_energy(self):
        mol = self.mol
        e = 0.0
        for i in range(mol.natoms):
            for j in range(i + 1, mol.natoms):
                dist = mol.coordsBohrs[i] - mol.coordsBohrs[j]
                e += mol.Zcharges[i] * mol.Zcharges[j] / np.sqrt(np.sum(dist ** 2))
        return e

    @staticmethod
    def _gen_dm(C, nocc):
        """Build density matrix from MO coefficients for `nocc` occupied orbitals."""
        Cocc = C[:, :nocc]
        return Cocc @ Cocc.T

    @staticmethod
    def solve(H, S, x=None):
        """Solve generalised eigenvalue problem via orthogonalisation."""
        if x is None:
            eig_val_s, eig_vec_s = scipy.linalg.eigh(S)
            x = eig_vec_s[:, eig_val_s > 1e-7] / np.sqrt(eig_val_s[eig_val_s > 1e-7])
        xHx = x.T @ H @ x
        eigvals, eigvecs = scipy.linalg.eigh(xHx)
        eigvecs = x @ eigvecs
        # Fix phase
        idx = np.argmax(np.abs(eigvecs.real), axis=0)
        eigvecs[:, eigvecs[idx, np.arange(len(eigvals))].real < 0] *= -1
        return eigvals, eigvecs

    def diis(self, S, D, F, fock_list, err_list):
        """DIIS extrapolation for a single Fock matrix."""
        FDS = F @ D @ S
        err = FDS - FDS.conj().T
        fock_list.append(F.copy())
        err_list.append(err)
        n = len(fock_list)
        if n > self._diis_space:
            fock_list.pop(0)
            err_list.pop(0)
            n -= 1
        B = np.zeros((n + 1, n + 1))
        B[-1, :] = B[:, -1] = -1.0
        B[-1, -1] = 0.0
        for i in range(n):
            for j in range(i + 1):
                val = np.real(np.trace(err_list[i].conj().T @ err_list[j]))
                B[i, j] = B[j, i] = val
        rhs = np.zeros(n + 1)
        rhs[-1] = -1.0
        try:
            w = scipy.linalg.solve(B, rhs)
        except scipy.linalg.LinAlgError:
            return F
        F_new = np.zeros_like(F)
        for i in range(n):
            F_new += w[i] * fock_list[i]
        return F_new

    # ------------------------------------------------------------------
    # SCF driver
    # ------------------------------------------------------------------
    def scf(self):
        mol = self.mol
        basis = self.basis
        ncores = self.ncores
        numba.set_num_threads(ncores)
        os.environ['RAYON_NUM_THREADS'] = str(ncores)

        nalpha = self.nalpha
        nbeta = self.nbeta
        is_uhf = self.is_uhf
        nbf = basis.bfs_nao

        print("=" * 60)
        print(f"  HF Calculation for Atom: {self.symbol}")
        print(f"  Electrons: {self.nelectrons}   "
              f"Multiplicity: {self.multiplicity}   "
              f"Nalpha: {nalpha}   Nbeta: {nbeta}")
        print(f"  Method: {'UHF' if is_uhf else 'RHF'}")
        print(f"  Basis functions: {nbf}")
        print(f"  Using 4c2e integrals (OS, coul_algo=1)")
        print("=" * 60)

        start_total = timer()

        # ----------------------------------------------------------
        # 1-electron integrals
        # ----------------------------------------------------------
        print("\nComputing 1-electron integrals...", flush=True)
        t0 = timer()
        S = Integrals.overlap_mat_symm(basis)
        T = Integrals.kin_mat_symm(basis)
        V = Integrals.nuc_mat_symm(basis, mol)
        Hcore = T + V
        print(f"  done  ({timer() - t0:.2f} s)")

        # Orthogonalisation matrix
        eig_val_s, eig_vec_s = scipy.linalg.eigh(S)
        x = eig_vec_s[:, eig_val_s > 1e-7] / np.sqrt(eig_val_s[eig_val_s > 1e-7])

        # ----------------------------------------------------------
        # 2-electron integrals  (full tensor, coul_algo=1, OS)
        # ----------------------------------------------------------
        print("\nComputing 4c2e ERI tensor (OS)...", flush=True)
        t0 = timer()
        # ints4c2e = Integrals.os_4c2e_symm(basis)
        ints4c2e = Integrals.conv_4c2e_symm(basis)
        print(f"  done  ({timer() - t0:.2f} s)")
        print(f"  ERI tensor size: {ints4c2e.nbytes / 1e9:.4f} GB")

        # ----------------------------------------------------------
        # Nuclear repulsion
        # ----------------------------------------------------------
        Enn = self._nuclear_rep_energy()

        # ----------------------------------------------------------
        # Initial guess (core Hamiltonian)
        # ----------------------------------------------------------
        eigvals0, eigvecs0 = self.solve(Hcore, S, x)
        Da = self._gen_dm(eigvecs0, nalpha)
        if is_uhf:
            Db = self._gen_dm(eigvecs0, nbeta)
        else:
            Db = Da.copy()

        # ----------------------------------------------------------
        # SCF iterations
        # ----------------------------------------------------------
        Etot = 0.0
        scf_converged = False
        itr = 0

        print("\nStarting SCF iterations...\n", flush=True)

        while itr < self.max_itr:
            itr += 1
            t_iter = timer()

            Dtotal = Da + Db

            # Coulomb matrix from total density
            J = contract('ijkl,kl->ij', ints4c2e, Dtotal)

            # Exchange matrices
            Ka = contract('ijkl,ik->jl', ints4c2e, Da)  # K_jl = sum_ik (ij|kl) Da_ik  ??? 
            # Actually exchange: K_ij = sum_kl (ik|jl) D_kl
            Ka = contract('ikjl,kl->ij', ints4c2e, Da)
            Kb = contract('ikjl,kl->ij', ints4c2e, Db)

            # Fock matrices
            Fa = Hcore + J - Ka
            if is_uhf:
                Fb = Hcore + J - Kb
            else:
                Fb = Fa.copy()

            # Energies
            Eone_a = 0.5 * np.trace((Hcore + Fa) @ Da)
            Eone_b = 0.5 * np.trace((Hcore + Fb) @ Db)
            Etot_new = Eone_a + Eone_b + Enn

            self.scf_energies.append(Etot_new)

            ediff = abs(Etot_new - Etot)
            print(f"  Iter {itr:3d}   E = {Etot_new:18.12f}   dE = {ediff:.2e}", flush=True)

            if ediff < self.conv_crit and itr > 1:
                scf_converged = True
                Etot = Etot_new
                break

            Etot = Etot_new

            # DIIS
            Fa = self.diis(S, Da, Fa, self._fock_list_a, self._err_list_a)
            if is_uhf:
                Fb = self.diis(S, Db, Fb, self._fock_list_b, self._err_list_b)

            # Diagonalise
            eigvals_a, eigvecs_a = self.solve(Fa, S, x)
            Da = self._gen_dm(eigvecs_a, nalpha)
            if is_uhf:
                eigvals_b, eigvecs_b = self.solve(Fb, S, x)
                Db = self._gen_dm(eigvecs_b, nbeta)
            else:
                eigvals_b, eigvecs_b = eigvals_a, eigvecs_a
                Db = self._gen_dm(eigvecs_b, nbeta)

        # ----------------------------------------------------------
        # Post-SCF
        # ----------------------------------------------------------
        self.converged = scf_converged
        self.niter = itr
        self.Total_energy = Etot
        self.dmat_alpha = Da
        self.dmat_beta = Db
        self.mo_energies_alpha = eigvals_a
        self.mo_energies_beta = eigvals_b
        self.mo_coefficients_alpha = eigvecs_a
        self.mo_coefficients_beta = eigvecs_b

        duration = timer() - start_total

        print("\n" + "=" * 60)
        if scf_converged:
            print(f"  SCF CONVERGED in {itr} iterations")
        else:
            print(f"  SCF NOT CONVERGED after {itr} iterations")
        print(f"  Total Energy = {Etot:18.12f} Hartree")
        print(f"  Nuclear Repulsion = {Enn:18.12f} Hartree")
        print(f"  Wall time: {duration:.2f} s")
        print("=" * 60 + "\n")

        return Etot, Da, Db