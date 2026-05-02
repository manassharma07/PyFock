from contextlib import nullcontext, redirect_stderr, redirect_stdout
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/pyfock_matplotlib")

import matplotlib.pyplot as plt
import numpy as np

from pyfock import Basis, DFT, Integrals, Mol


def pyfock_hf_quantities(symbols, coordinates, charge=0, basis_name="sto-3g", verbose=False):
    """Return HF orbitals and MO integrals in PennyLane's qchem convention."""
    atoms = [[symbol, *xyz] for symbol, xyz in zip(symbols, coordinates)]
    mol = Mol(atoms=atoms, charge=charge)
    basis = Basis(mol, {"all": Basis.load(mol=mol, basis_name=basis_name)})

    hf = DFT(mol, basis, xc="HF")
    hf.isDF = False
    hf.rys = True
    hf.direct_scf = False
    hf.coul_algo = 1
    hf.conv_crit = 1e-8
    hf.max_itr = 50
    hf.ncores = 1
    hf.threshold_schwarz = 1e-12

    output = nullcontext() if verbose else open(os.devnull, "w")
    with output as stream:
        out_context = nullcontext() if verbose else redirect_stdout(stream)
        err_context = nullcontext() if verbose else redirect_stderr(stream)
        with out_context, err_context:
            hf_energy, _ = hf.scf()

    mo_coeff = hf.mo_coefficients
    h_core_ao = Integrals.kin_mat_symm(basis) + Integrals.nuc_mat_symm(basis, mol)
    eri_ao = Integrals.rys_4c2e_symm(basis)

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
        "core_constant": np.array([hf.nuclear_rep_energy(mol)]),
        "one_mo": one_mo,
        "two_mo": two_mo,
        "electrons": mol.nelectrons,
        "qubits": 2 * one_mo.shape[0],
    }


def vqe_energy(data, params=None, max_iter=80, conv_tol=1e-7, verbose=False):
    import pennylane as qml
    from pennylane import numpy as pnp

    fermionic_h = qml.qchem.fermionic_observable(
        data["core_constant"], data["one_mo"], data["two_mo"]
    )
    hamiltonian = qml.qchem.qubit_observable(fermionic_h, mapping="jordan_wigner")

    electrons = data["electrons"]
    qubits = data["qubits"]
    hf_state = qml.qchem.hf_state(electrons, qubits)
    singles, doubles = qml.qchem.excitations(electrons, qubits)
    s_wires, d_wires = qml.qchem.excitations_to_wires(singles, doubles)

    dev = qml.device("lightning.qubit", wires=qubits)

    @qml.qnode(dev)
    def circuit(theta):
        qml.UCCSD(
            theta, wires=range(qubits), s_wires=s_wires, d_wires=d_wires, init_state=hf_state
        )
        return qml.expval(hamiltonian)

    if params is None:
        params = pnp.zeros(len(singles) + len(doubles))
    else:
        params = pnp.array(params, requires_grad=True)

    opt = qml.AdagradOptimizer(stepsize=0.1)
    energy = circuit(params)
    for step in range(max_iter):
        params, energy = opt.step_and_cost(circuit, params)
        if verbose:
            print(f"Step {step+1:2d}: Energy = {energy:.12f} Ha")
        if step and abs(energy - prev_energy) < conv_tol:
            break
        prev_energy = energy

    return float(energy), np.array(params, dtype=float)


if __name__ == "__main__":
    bond_lengths = np.linspace(1.2, 2.44, 10)
    energies = []
    params = None

    for r in bond_lengths:
        symbols = ["Li", "H"]
        coordinates = np.array([[0.0, 0.0, 0.0], [r, 0.0, 0.0]])
        data = pyfock_hf_quantities(symbols, coordinates, basis_name="sto-3g")
        energy, params = vqe_energy(data, params=params, max_iter=80, conv_tol=1e-7, verbose=True)
        energies.append(energy)
        print(f"R = {r:5.2f} A   E_VQE = {energy: .10f} Ha")

    energies = np.array(energies)
    i_min = int(np.argmin(energies))
    r_eq = bond_lengths[i_min]
    e_eq = energies[i_min]
    binding_energy = energies[-1] - e_eq

    print(f"\nEquilibrium bond length: {r_eq:.3f} A")
    print(f"Minimum VQE energy:      {e_eq:.10f} Ha")
    print(f"Binding energy:          {binding_energy:.10f} Ha  (relative to largest R)")
    print(f"Binding energy:          {binding_energy * 627.509474:.4f} kcal/mol")

    plt.plot(bond_lengths, energies, "o-", label="VQE")
    plt.axvline(r_eq, color="black", linestyle="--", linewidth=1)
    plt.xlabel("Li-H bond length (A)")
    plt.ylabel("Energy (Hartree)")
    plt.title("LiH dissociation curve from PyFock integrals + PennyLane VQE")
    plt.legend()
    plt.tight_layout()
    # plt.savefig("ex37_LiH_VQE_dissociation_curve.png", dpi=200)
    plt.show()
