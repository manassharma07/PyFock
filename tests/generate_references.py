from __future__ import annotations

from pathlib import Path

import numpy as np

from pyfock import Integrals

from conftest import (
    AUX_BASIS_NAME,
    BASIS_NAMES,
    DATA_DIR,
    REFS_PATH,
    RYS_4C2E_SLICE,
    build_basis,
    build_h2o_mol,
)


def make_os_sample_indices(nao: int) -> np.ndarray:
    candidates = [
        (0, 0, 0, 0),
        (0, 0, 0, min(1, nao - 1)),
        (0, min(1, nao - 1), 0, min(1, nao - 1)),
        (min(1, nao - 1), min(1, nao - 1), min(1, nao - 1), min(1, nao - 1)),
        (0, min(2, nao - 1), min(1, nao - 1), min(2, nao - 1)),
    ]
    return np.asarray(candidates, dtype=np.int64)


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    mol = build_h2o_mol()
    auxbasis = build_basis(mol, AUX_BASIS_NAME)
    refs: dict[str, np.ndarray] = {}

    refs["rys_2c2e_aux"] = Integrals.rys_2c2e_symm(auxbasis)
    refs["rys_2c2e_diag_aux"] = Integrals.rys_2c2e_diag(auxbasis)

    for basis_name in BASIS_NAMES:
        tag = basis_name.lower()
        basis = build_basis(mol, basis_name)

        refs[f"overlap_{tag}"] = Integrals.overlap_mat_symm(basis)
        refs[f"nuclear_{tag}"] = Integrals.nuc_mat_symm(basis, mol)
        refs[f"kinetic_{tag}"] = Integrals.kin_mat_symm(basis)
        refs[f"dipole_{tag}"] = Integrals.dipole_moment_mat_symm(basis)

        refs[f"overlap_grad_{tag}"] = Integrals.overlap_mat_grad_symm(basis)
        refs[f"kinetic_grad_{tag}"] = Integrals.kin_mat_grad_symm(basis)
        refs[f"nuclear_grad_{tag}"] = Integrals.nuc_mat_grad_symm(basis, mol)

        refs[f"rys_3c2e_{tag}"] = Integrals.rys_3c2e_symm(basis, auxbasis)
        refs[f"rys_3c2e_tri_{tag}"] = Integrals.rys_3c2e_tri(basis, auxbasis)
        refs[f"rys_4c2e_slice_{tag}"] = Integrals.rys_4c2e_symm(basis, slice=list(RYS_4C2E_SLICE))

        os_4c2e = Integrals.os_4c2e_symm(basis)
        sample_indices = make_os_sample_indices(basis.bfs_nao)
        refs[f"os_4c2e_shape_{tag}"] = np.asarray(os_4c2e.shape, dtype=np.int64)
        refs[f"os_4c2e_sum_{tag}"] = np.asarray([os_4c2e.sum()], dtype=np.float64)
        refs[f"os_4c2e_norm_{tag}"] = np.asarray([np.linalg.norm(os_4c2e.ravel())], dtype=np.float64)
        refs[f"os_4c2e_sample_indices_{tag}"] = sample_indices
        refs[f"os_4c2e_sample_values_{tag}"] = np.asarray(
            [os_4c2e[tuple(index)] for index in sample_indices], dtype=np.float64
        )

    np.savez_compressed(REFS_PATH, **refs)
    print(f"Wrote references to {REFS_PATH}")


if __name__ == "__main__":
    main()
