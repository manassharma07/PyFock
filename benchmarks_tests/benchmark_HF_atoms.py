import numpy as np
from pyfock import Mol, Basis, HF_atoms

# Ground state multiplicities for Z=1 to Z=89
# Format: {atomic_symbol: (Z, multiplicity)}
atom_data = {
    'H':  (1, 2),   'He': (2, 1),   'Li': (3, 2),   'Be': (4, 1),
    'B':  (5, 2),   'C':  (6, 3),   'N':  (7, 4),   'O':  (8, 3),
    'F':  (9, 2),   'Ne': (10, 1),  'Na': (11, 2),  'Mg': (12, 1),
    'Al': (13, 2),  'Si': (14, 3),  'P':  (15, 4),  'S':  (16, 3),
    'Cl': (17, 2),  'Ar': (18, 1),  'K':  (19, 2),  'Ca': (20, 1),
    'Sc': (21, 2),  'Ti': (22, 3),  'V':  (23, 4),  'Cr': (24, 7),
    'Mn': (25, 6),  'Fe': (26, 5),  'Co': (27, 4),  'Ni': (28, 3),
    'Cu': (29, 2),  'Zn': (30, 1),  'Ga': (31, 2),  'Ge': (32, 3),
    'As': (33, 4),  'Se': (34, 3),  'Br': (35, 2),  'Kr': (36, 1),
    # 'Rb': (37, 2),  'Sr': (38, 1),  'Y':  (39, 2),  'Zr': (40, 3),
    # 'Nb': (41, 6),  'Mo': (42, 7),  'Tc': (43, 6),  'Ru': (44, 5),
    # 'Rh': (45, 4),  'Pd': (46, 1),  'Ag': (47, 2),  'Cd': (48, 1),
    # 'In': (49, 2),  'Sn': (50, 3),  'Sb': (51, 4),  'Te': (52, 3),
    # 'I':  (53, 2),  'Xe': (54, 1),  'Cs': (55, 2),  'Ba': (56, 1),
    # 'La': (57, 2),  'Ce': (58, 3),  'Pr': (59, 4),  'Nd': (60, 5),
    # 'Pm': (61, 6),  'Sm': (62, 7),  'Eu': (63, 8),  'Gd': (64, 9),
    # 'Tb': (65, 6),  'Dy': (66, 5),  'Ho': (67, 4),  'Er': (68, 3),
    # 'Tm': (69, 2),  'Yb': (70, 1),  'Lu': (71, 2),  'Hf': (72, 3),
    # 'Ta': (73, 4),  'W':  (74, 5),  'Re': (75, 6),  'Os': (76, 5),
    # 'Ir': (77, 4),  'Pt': (78, 3),  'Au': (79, 2),  'Hg': (80, 1),
    # 'Tl': (81, 2),  'Pb': (82, 3),  'Bi': (83, 4),  'Po': (84, 3),
    # 'At': (85, 2),  'Rn': (86, 1),  'Fr': (87, 2),  'Ra': (88, 1),
    # 'Ac': (89, 2),
}

# Try importing PySCF
try:
    import pyscf
    from pyscf import gto, scf
    HAS_PYSCF = True
    print("PySCF found. Will compare results.\n")
except ImportError:
    HAS_PYSCF = False
    print("PySCF not found. Only PyFock results will be shown.\n")


def run_pyfock(atom_symbol):
    """Run HF calculation using PyFock for a single atom."""
    try:
        mol = Mol(atoms=[[atom_symbol, 0.0, 0.0, 0.0]])
        basis = Basis(mol, {'all': Basis.load(mol=mol, basis_name='def2-SVP')})
        hf = HF_atoms(mol, basis)
        Etot, Da, Db = hf.scf()
        converged = hf.converged
        method = 'UHF' if hf.is_uhf else 'RHF'
        return Etot, converged, method
    except Exception as e:
        return None, False, str(e)


def run_pyscf(atom_symbol, Z, mult):
    """Run HF calculation using PySCF for a single atom."""
    try:
        charge = 0
        spin = mult - 1  # PySCF uses 2S, not 2S+1

        mol = gto.M(
            atom=f'{atom_symbol} 0.0 0.0 0.0',
            basis='def2-SVP',
            charge=charge,
            spin=spin,
            symmetry=False,
            verbose=0,
            cart=True
        )

        if mult == 1:
            # Restricted HF for singlets
            mf = scf.RHF(mol)
        else:
            # Unrestricted HF for open-shell
            mf = scf.UHF(mol)

        mf.init_guess = '1e'
        mf.max_cycle = 500
        mf.conv_tol = 1e-10
        energy = mf.kernel()
        converged = mf.converged

        return energy, converged
    except Exception as e:
        return None, False


# ============================================================
# Main calculation loop
# ============================================================
print("=" * 100)
print(f"{'Atom':>4s} {'Z':>3s} {'Mult':>4s} | {'PyFock E (Ha)':>18s} {'Conv':>5s} {'Method':>6s} | "
      f"{'PySCF E (Ha)':>18s} {'Conv':>5s} | {'ΔE (mHa)':>12s} {'ΔE (kcal/mol)':>14s}")
print("=" * 100)

results = []
failed_pyfock = []
failed_pyscf = []
large_deviations = []

for symbol, (Z, mult) in atom_data.items():
    # --- PyFock ---
    pyfock_E, pyfock_conv, pyfock_method = run_pyfock(symbol)

    # --- PySCF ---
    if HAS_PYSCF:
        pyscf_E, pyscf_conv = run_pyscf(symbol, Z, mult)
    else:
        pyscf_E, pyscf_conv = None, False

    # --- Compute differences ---
    if pyfock_E is not None and pyscf_E is not None:
        delta_mHa = (pyfock_E - pyscf_E) * 1000.0  # millihartree
        delta_kcal = (pyfock_E - pyscf_E) * 627.5094740631  # kcal/mol
        delta_mHa_str = f"{delta_mHa:12.4f}"
        delta_kcal_str = f"{delta_kcal:14.6f}"
    else:
        delta_mHa_str = "N/A"
        delta_kcal_str = "N/A"
        delta_mHa = None
        delta_kcal = None

    # Format energies
    pyfock_E_str = f"{pyfock_E:18.10f}" if pyfock_E is not None else "FAILED"
    pyfock_conv_str = "Yes" if pyfock_conv else "No"
    pyscf_E_str = f"{pyscf_E:18.10f}" if pyscf_E is not None else "FAILED"
    pyscf_conv_str = "Yes" if pyscf_conv else "No"

    print(f"{symbol:>4s} {Z:3d} {mult:4d} | {pyfock_E_str:>18s} {pyfock_conv_str:>5s} {pyfock_method:>6s} | "
          f"{pyscf_E_str:>18s} {pyscf_conv_str:>5s} | {delta_mHa_str:>12s} {delta_kcal_str:>14s}")

    # Store results
    results.append({
        'symbol': symbol,
        'Z': Z,
        'mult': mult,
        'pyfock_E': pyfock_E,
        'pyfock_conv': pyfock_conv,
        'pyfock_method': pyfock_method,
        'pyscf_E': pyscf_E,
        'pyscf_conv': pyscf_conv,
        'delta_mHa': delta_mHa,
        'delta_kcal': delta_kcal,
    })

    if pyfock_E is None:
        failed_pyfock.append(symbol)
    if pyscf_E is None and HAS_PYSCF:
        failed_pyscf.append(symbol)
    if delta_mHa is not None and abs(delta_mHa) > 1.0:
        large_deviations.append((symbol, delta_mHa))

# ============================================================
# Summary statistics
# ============================================================
print("\n" + "=" * 100)
print("SUMMARY")
print("=" * 100)

# Filter valid comparisons
valid = [r for r in results if r['delta_mHa'] is not None]

if valid:
    deltas = np.array([r['delta_mHa'] for r in valid])
    abs_deltas = np.abs(deltas)

    print(f"\nTotal atoms attempted:          {len(results)}")
    print(f"Successful comparisons:         {len(valid)}")
    print(f"PyFock failures:                {len(failed_pyfock)}  {failed_pyfock if failed_pyfock else ''}")
    if HAS_PYSCF:
        print(f"PySCF failures:                 {len(failed_pyscf)}  {failed_pyscf if failed_pyscf else ''}")

    print(f"\n--- Energy Difference Statistics (PyFock - PySCF) ---")
    print(f"  Mean ΔE:                      {np.mean(deltas):12.6f} mHa")
    print(f"  Std Dev ΔE:                   {np.std(deltas):12.6f} mHa")
    print(f"  Mean |ΔE|:                    {np.mean(abs_deltas):12.6f} mHa")
    print(f"  Max |ΔE|:                     {np.max(abs_deltas):12.6f} mHa  ({valid[np.argmax(abs_deltas)]['symbol']})")
    print(f"  Min |ΔE|:                     {np.min(abs_deltas):12.6f} mHa  ({valid[np.argmin(abs_deltas)]['symbol']})")

    # Count atoms matching to various thresholds
    for thresh in [0.0000001, 0.00001, 0.001, 0.01, 0.1, 1.0, 10.0]:
        count = np.sum(abs_deltas < thresh)
        print(f"  |ΔE| < {thresh:6.3f} mHa:            {count:3d} / {len(valid)}")

    if large_deviations:
        print(f"\n--- Atoms with |ΔE| > 1 mHa ---")
        for sym, dE in sorted(large_deviations, key=lambda x: abs(x[1]), reverse=True):
            print(f"  {sym:>4s}:  ΔE = {dE:12.4f} mHa")

    # PyFock convergence summary
    pyfock_conv_count = sum(1 for r in results if r['pyfock_conv'])
    print(f"\nPyFock converged:               {pyfock_conv_count} / {len(results)}")
    not_converged = [r['symbol'] for r in results if not r['pyfock_conv'] and r['pyfock_E'] is not None]
    if not_converged:
        print(f"  Not converged: {not_converged}")

else:
    print("No valid comparisons could be made.")

print("\nDone.")