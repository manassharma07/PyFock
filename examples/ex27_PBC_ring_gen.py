from pyfock import Basis
from pyfock import Mol
from pyfock import Integrals
from pyfock import DFT
from pyfock import Utils
from pyfock import PBC_ring

from timeit import default_timer as timer
import numpy as np
import scipy
import matplotlib.pyplot as plt

# PySCF imports
from pyscf.pbc import gto, scf
from pyscf import gto as mol_gto, dft as molecular_dft

ncores = 4

#LDA
# funcx = 1
# funcc = 7
#PBE
funcx = 101
funcc = 130
funcidcrysx = [funcx, funcc]
funcidpyscf = str(funcx)+','+str(funcc)

basis_set_name = 'def2-SVP'
auxbasis_name = 'def2-universal-jfit'
xyzFilename = 'LiH.xyz'



#Initialize a Mol object with unit cell
unit_mol = Mol(coordfile=xyzFilename)

# Generate rings and save XYZ files
ring_sizes = [10, 15]
pyfock_results = []
pyscf_results = []

for N in ring_sizes:
    print(f"\nPyFock Ring N={N}")
    ring_mol = PBC_ring.ring(unit_mol, N=N, periodicity=3.2, periodic_dir='x', 
                            output_xyz=True, xyz_filename=f'pbc_ring_{N}')

    print("\n=== PySCF Ring Calculations (using PyFock XYZ files) ===")

    print(f"\nPySCF Ring N={N} (from pbc_ring_{N}.xyz)")
    
    # Read PyFock-generated XYZ file
    mol = mol_gto.Mole()
    mol.atom = f'pbc_ring_{N}.xyz'
    mol.basis = basis_set_name
    mol.spin = 0
    mol.charge = 0
    mol.build()
    
    # DFT calculation
    mf = molecular_dft.RKS(mol).density_fit()
    dmat_init = mf.init_guess_by_minao(mol)
    mf.xc = funcidpyscf
    mf.conv_tol = 1e-7
    e_total = mf.kernel(dmat_init)
    e_per_unit = e_total / N
    
    pyscf_results.append((N, e_total, e_per_unit))
    print(f"  PySCF E_total = {e_total:.10f} Ha, E/unit = {e_per_unit:.10f} Ha")
    
    print("=== PyFock Ring Calculation ===")
    #Initialize basis
    basis = Basis(ring_mol, {'all':Basis.load(mol=ring_mol, basis_name=basis_set_name)})
    auxbasis = Basis(ring_mol, {'all':Basis.load(mol=ring_mol, basis_name=auxbasis_name)})
    
    dftObj = DFT(ring_mol, basis, auxbasis, xc=funcidcrysx, grids=mf.grids)
    dftObj.conv_crit = 1e-7
    dftObj.max_itr = 50
    dftObj.ncores = ncores
    # dftObj.strict_schwarz = False
    dftObj.save_ao_values = True
    # dftObj.dmat = dmat_init
    
    energyCrysX, dmat = dftObj.scf()
    n_units = len(ring_mol.atoms) // 2
    energy_per_unit = energyCrysX / n_units
    
    pyfock_results.append((N, energyCrysX, energy_per_unit))
    print(f"  PyFock E_total = {energyCrysX:.10f} Ha, E/unit = {energy_per_unit:.10f} Ha")

print("\n=== PBC Reference (PySCF) ===")

# PBC calculation
def build_lih_pbc_cell():
    cell = gto.Cell()
    cell.unit = "Angstrom"
    cell.atom = [["Li", (0.0, 0.0, 0.0)], ["H", (1.6, 0.0, 0.0)]]  # 1.6 Å LiH bond
    cell.basis = basis_set_name
    cell.a = np.array([[3.2, 0.0, 0.0], [0.0, 25.0, 0.0], [0.0, 0.0, 25.0]])  # 3.2 Å periodicity
    cell.build()
    return cell

cell = build_lih_pbc_cell()
kpts = cell.make_kpts([20, 1, 1])
mf_pbc = scf.KRKS(cell, kpts=kpts).density_fit()
mf_pbc.xc = funcidpyscf
mf_pbc.conv_tol = 1e-7
e_pbc_total = mf_pbc.kernel()
e_pbc_per_unit = e_pbc_total  # Only 1 LiH unit in cell

print(f"PBC E_total = {e_pbc_total:.10f} Ha, E/unit = {e_pbc_per_unit:.10f} Ha")

print("\n=== PySCF Ring Calculations (using PyFock XYZ files) ===")



    

print("\n=== Extrapolation Analysis ===")

# Use PySCF results for extrapolation (more standard)
Ns = np.array([r[0] for r in pyscf_results])
E_per_unit = np.array([r[2] for r in pyscf_results])
x = 1.0 / (Ns**2)

# Linear fit
coeffs = np.polyfit(x, E_per_unit, 1)
a, b = coeffs[0], coeffs[1]
extrapolated_limit = b

print(f"Linear fit: E/unit = {a:.6e} * (1/N²) + {b:.10f}")
print(f"Extrapolated limit (N→∞): {extrapolated_limit:.10f} Ha")
print(f"PBC reference: {e_pbc_per_unit:.10f} Ha")
print(f"Difference: {(extrapolated_limit - e_pbc_per_unit)*1000:.3f} mHa")

# Plot
plt.figure(figsize=(10, 6))
plt.scatter(x, E_per_unit, s=100, color='red', label='PySCF Rings', zorder=5)

xfit = np.linspace(0, x.max()*1.1, 100)
yfit = a * xfit + b
plt.plot(xfit, yfit, '--', color='blue', label='Linear fit')
plt.axhline(y=e_pbc_per_unit, color='green', linestyle='-', linewidth=2, 
            label=f'PBC: {e_pbc_per_unit:.6f} Ha')

plt.xlabel("1 / N²")
plt.ylabel("Energy per LiH unit (Ha)")
plt.title("Ring Energy Extrapolation to Thermodynamic Limit")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("ring_extrapolation.png", dpi=150)
print(f"\nSaved plot: ring_extrapolation.png")

print("\n=== Summary Comparison ===")
print(f"{'Method':<20} {'N':<5} {'E/unit (Ha)':<15} {'Error vs PBC (mHa)':<15}")
print("-" * 60)
print(f"{'PBC Reference':<20} {'∞':<5} {e_pbc_per_unit:<15.10f} {'0.000':<15}")

for i, (pf_result, ps_result) in enumerate(zip(pyfock_results, pyscf_results)):
    N, pf_etot, pf_epu = pf_result
    _, ps_etot, ps_epu = ps_result
    
    pf_error = (pf_epu - e_pbc_per_unit) * 1000
    ps_error = (ps_epu - e_pbc_per_unit) * 1000
    
    print(f"{'PyFock Ring':<20} {N:<5} {pf_epu:<15.10f} {pf_error:<15.3f}")
    print(f"{'PySCF Ring':<20} {N:<5} {ps_epu:<15.10f} {ps_error:<15.3f}")

print(f"{'Extrapolated':<20} {'∞':<5} {extrapolated_limit:<15.10f} {(extrapolated_limit-e_pbc_per_unit)*1000:<15.3f}")