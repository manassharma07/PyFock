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
ring_sizes = [10, 15, 25]
pyfock_results = []
pyscf_results = []
pyfock_timings = []
pyscf_mol_timings = []

for N in ring_sizes:
    print(f"\nPyFock Ring N={N}")
    ring_mol = PBC_ring.ring(unit_mol, N=N, periodicity=3.2, periodic_dir='x', 
                            output_xyz=True, xyz_filename=f'pbc_ring_{N}')

    print(f"\n=== PySCF Molecular Ring N={N} (from pbc_ring_{N}.xyz) ===")
    
    # Read PyFock-generated XYZ file
    mol = mol_gto.Mole()
    mol.atom = f'pbc_ring_{N}.xyz'
    mol.basis = basis_set_name
    mol.spin = 0
    mol.charge = 0
    mol.verbose = 4
    mol.build()
    
    # DFT calculation with timing
    mf = molecular_dft.RKS(mol).density_fit(auxbasis=auxbasis_name)
    dmat_init = mf.init_guess_by_minao(mol)
    mf.xc = funcidpyscf
    mf.conv_tol = 1e-7
    
    t_pyscf_start = timer()
    e_total = mf.kernel(dmat_init)
    t_pyscf_end = timer()
    pyscf_mol_time = t_pyscf_end - t_pyscf_start
    
    e_per_unit = e_total / N
    
    pyscf_results.append((N, e_total, e_per_unit))
    pyscf_mol_timings.append((N, pyscf_mol_time))
    print(f"  PySCF E_total = {e_total:.10f} Ha, E/unit = {e_per_unit:.10f} Ha")
    print(f"  PySCF Molecular Wall time = {pyscf_mol_time:.2f} s")
    
    print(f"\n=== PyFock Ring Calculation N={N} ===")
    #Initialize basis
    basis = Basis(ring_mol, {'all':Basis.load(mol=ring_mol, basis_name=basis_set_name)})
    auxbasis = Basis(ring_mol, {'all':Basis.load(mol=ring_mol, basis_name=auxbasis_name)})
    
    dftObj = DFT(ring_mol, basis, auxbasis, xc=funcidcrysx, grids=mf.grids)
    dftObj.conv_crit = 1e-7
    dftObj.max_itr = 50
    dftObj.ncores = ncores
    dftObj.strict_schwarz = True
    dftObj.save_ao_values = True
    dftObj.dmat = dmat_init
    dftObj.sao = True
    
    t_pyfock_start = timer()
    energyCrysX, dmat = dftObj.scf()
    t_pyfock_end = timer()
    pyfock_time = t_pyfock_end - t_pyfock_start
    
    n_units = len(ring_mol.atoms) // 2
    energy_per_unit = energyCrysX / n_units
    
    pyfock_results.append((N, energyCrysX, energy_per_unit))
    pyfock_timings.append((N, pyfock_time))
    print(f"  PyFock E_total = {energyCrysX:.10f} Ha, E/unit = {energy_per_unit:.10f} Ha")
    print(f"  PyFock Wall time = {pyfock_time:.2f} s")

print("\n=== PBC Reference (PySCF) ===")

# PBC calculation with timing
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
mf_pbc = scf.KRKS(cell, kpts=kpts).density_fit(auxbasis=auxbasis_name)
mf_pbc.xc = funcidpyscf
mf_pbc.conv_tol = 1e-7

t_pbc_start = timer()
e_pbc_total = mf_pbc.kernel()
t_pbc_end = timer()
pyscf_pbc_time = t_pbc_end - t_pbc_start

e_pbc_per_unit = e_pbc_total  # Only 1 LiH unit in cell

print(f"PBC E_total = {e_pbc_total:.10f} Ha, E/unit = {e_pbc_per_unit:.10f} Ha")
print(f"PySCF PBC Wall time = {pyscf_pbc_time:.2f} s")

print("\n=== Extrapolation Analysis ===")

# PySCF extrapolation
Ns_pyscf = np.array([r[0] for r in pyscf_results])
E_per_unit_pyscf = np.array([r[2] for r in pyscf_results])
x_pyscf = 1.0 / (Ns_pyscf**2)

coeffs_pyscf = np.polyfit(x_pyscf, E_per_unit_pyscf, 1)
a_pyscf, b_pyscf = coeffs_pyscf[0], coeffs_pyscf[1]
extrapolated_pyscf = b_pyscf

print(f"PySCF Linear fit: E/unit = {a_pyscf:.6e} * (1/N²) + {b_pyscf:.10f}")
print(f"PySCF Extrapolated limit (N→∞): {extrapolated_pyscf:.10f} Ha")
print(f"PBC reference: {e_pbc_per_unit:.10f} Ha")
print(f"PySCF Difference: {(extrapolated_pyscf - e_pbc_per_unit)*1000:.3f} mHa")

# PyFock extrapolation
Ns_pyfock = np.array([r[0] for r in pyfock_results])
E_per_unit_pyfock = np.array([r[2] for r in pyfock_results])
x_pyfock = 1.0 / (Ns_pyfock**2)

coeffs_pyfock = np.polyfit(x_pyfock, E_per_unit_pyfock, 1)
a_pyfock, b_pyfock = coeffs_pyfock[0], coeffs_pyfock[1]
extrapolated_pyfock = b_pyfock

print(f"\nPyFock Linear fit: E/unit = {a_pyfock:.6e} * (1/N²) + {b_pyfock:.10f}")
print(f"PyFock Extrapolated limit (N→∞): {extrapolated_pyfock:.10f} Ha")
print(f"PyFock Difference vs PBC: {(extrapolated_pyfock - e_pbc_per_unit)*1000:.3f} mHa")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# --- Left panel: Energy extrapolation ---
ax1 = axes[0]
ax1.scatter(x_pyscf, E_per_unit_pyscf, s=100, color='red', label='PySCF Rings', zorder=5)
ax1.scatter(x_pyfock, E_per_unit_pyfock, s=100, color='blue', marker='^', label='PyFock Rings', zorder=5)

xfit = np.linspace(0, max(x_pyscf.max(), x_pyfock.max())*1.1, 100)
yfit_pyscf = a_pyscf * xfit + b_pyscf
yfit_pyfock = a_pyfock * xfit + b_pyfock
ax1.plot(xfit, yfit_pyscf, '--', color='red', alpha=0.7, label=f'PySCF fit (extrap={extrapolated_pyscf:.6f})')
ax1.plot(xfit, yfit_pyfock, '--', color='blue', alpha=0.7, label=f'PyFock fit (extrap={extrapolated_pyfock:.6f})')
ax1.axhline(y=e_pbc_per_unit, color='green', linestyle='-', linewidth=2, 
            label=f'PBC: {e_pbc_per_unit:.6f} Ha')

ax1.set_xlabel("1 / N²")
ax1.set_ylabel("Energy per LiH unit (Ha)")
ax1.set_title("Ring Energy Extrapolation to Thermodynamic Limit")
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)

# --- Right panel: Timings ---
ax2 = axes[1]
Ns_timing = np.array([t[0] for t in pyscf_mol_timings])
pyscf_times = np.array([t[1] for t in pyscf_mol_timings])
pyfock_times_arr = np.array([t[1] for t in pyfock_timings])

ax2.plot(Ns_timing, pyscf_times, 'o-', color='red', label='PySCF Molecular', markersize=8)
ax2.plot(Ns_timing, pyfock_times_arr, '^-', color='blue', label='PyFock', markersize=8)
ax2.axhline(y=pyscf_pbc_time, color='green', linestyle='-', linewidth=2, 
            label=f'PySCF PBC: {pyscf_pbc_time:.1f} s')

ax2.set_xlabel("Ring Size N")
ax2.set_ylabel("Wall Time (s)")
ax2.set_title("SCF Timing Comparison")
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("ring_extrapolation.png", dpi=150)
print(f"\nSaved plot: ring_extrapolation.png")

print("\n=== Summary Comparison ===")
print(f"{'Method':<20} {'N':<5} {'E/unit (Ha)':<18} {'Error vs PBC (mHa)':<20} {'Time (s)':<10}")
print("-" * 75)
print(f"{'PBC Reference':<20} {'∞':<5} {e_pbc_per_unit:<18.10f} {'0.000':<20} {pyscf_pbc_time:<10.2f}")

for i, (pf_result, ps_result) in enumerate(zip(pyfock_results, pyscf_results)):
    N, pf_etot, pf_epu = pf_result
    _, ps_etot, ps_epu = ps_result
    
    pf_error = (pf_epu - e_pbc_per_unit) * 1000
    ps_error = (ps_epu - e_pbc_per_unit) * 1000
    
    pf_time = pyfock_timings[i][1]
    ps_time = pyscf_mol_timings[i][1]
    
    print(f"{'PyFock Ring':<20} {N:<5} {pf_epu:<18.10f} {pf_error:<20.3f} {pf_time:<10.2f}")
    print(f"{'PySCF Ring':<20} {N:<5} {ps_epu:<18.10f} {ps_error:<20.3f} {ps_time:<10.2f}")

print(f"{'PySCF Extrapolated':<20} {'∞':<5} {extrapolated_pyscf:<18.10f} {(extrapolated_pyscf-e_pbc_per_unit)*1000:<20.3f}")
print(f"{'PyFock Extrapolated':<20} {'∞':<5} {extrapolated_pyfock:<18.10f} {(extrapolated_pyfock-e_pbc_per_unit)*1000:<20.3f}")

print("\n=== Timing Summary ===")
for N, t in pyscf_mol_timings:
    print(f"  PySCF Molecular N={N}: {t:.2f} s")
for N, t in pyfock_timings:
    print(f"  PyFock          N={N}: {t:.2f} s")
print(f"  PySCF PBC (k=[20,1,1]): {pyscf_pbc_time:.2f} s")