from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from pyscf import gto, scf


EXAMPLE = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "ex35_VQE_using_PyFock_and_Pennylane.py"
)


def load_example():
    spec = importlib.util.spec_from_file_location("ex35_PennyLane_VQE_integrals", EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def pyscf_hf_quantities(symbols, coordinates, charge=0, basis_name="sto-3g"):
    mol = gto.M(
        atom=[(symbol, tuple(xyz)) for symbol, xyz in zip(symbols, coordinates)],
        basis=basis_name,
        charge=charge,
        spin=0,
        unit="Angstrom",
        cart=True,
        verbose=0,
    )
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-10
    hf_energy = mf.kernel()

    mo_coeff = mf.mo_coeff
    h_core_ao = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
    eri_ao = mol.intor("int2e")
    one_mo = np.einsum("qr,rs,st->qt", mo_coeff.T, h_core_ao, mo_coeff)
    two_mo = np.swapaxes(
        np.einsum(
            "ab,cd,bdeg,ef,gh->acfh", mo_coeff.T, mo_coeff.T, eri_ao, mo_coeff, mo_coeff
        ),
        1,
        3,
    )

    return {
        "hf_energy": hf_energy,
        "core_constant": np.array([mol.energy_nuc()]),
        "one_mo": one_mo,
        "two_mo": two_mo,
    }


def test_pyfock_hf_quantities_match_pyscf_for_h2_sto3g():
    example = load_example()
    symbols = ["H", "H"]
    coordinates = np.array([[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]])

    pyfock_data = example.pyfock_hf_quantities(symbols, coordinates)
    pyscf_data = pyscf_hf_quantities(symbols, coordinates)

    assert pyfock_data["qubits"] == 4
    assert pyfock_data["electrons"] == 2
    np.testing.assert_allclose(pyfock_data["hf_energy"], pyscf_data["hf_energy"], atol=5e-9)
    np.testing.assert_allclose(pyfock_data["core_constant"], pyscf_data["core_constant"], atol=5e-9)
    np.testing.assert_allclose(pyfock_data["one_mo"], pyscf_data["one_mo"], atol=5e-8)
    np.testing.assert_allclose(pyfock_data["two_mo"], pyscf_data["two_mo"], atol=5e-8)


def test_pennylane_hamiltonian_from_pyfock_integrals_matches_pyscf():
    try:
        import pennylane as qml
    except Exception as exc:
        pytest.skip(f"PennyLane is not importable in this environment: {exc}")
    example = load_example()
    symbols = ["H", "H"]
    coordinates = np.array([[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]])

    pyfock_data = example.pyfock_hf_quantities(symbols, coordinates)
    pyscf_data = pyscf_hf_quantities(symbols, coordinates)

    h_pyfock = qml.qchem.qubit_observable(
        qml.qchem.fermionic_observable(
            pyfock_data["core_constant"], pyfock_data["one_mo"], pyfock_data["two_mo"]
        ),
        mapping="jordan_wigner",
    )
    h_pyscf = qml.qchem.qubit_observable(
        qml.qchem.fermionic_observable(
            pyscf_data["core_constant"], pyscf_data["one_mo"], pyscf_data["two_mo"]
        ),
        mapping="jordan_wigner",
    )

    wires = range(pyfock_data["qubits"])
    np.testing.assert_allclose(
        qml.matrix(h_pyfock, wire_order=wires), qml.matrix(h_pyscf, wire_order=wires), atol=5e-8
    )
