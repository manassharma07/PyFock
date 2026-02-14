import pylibxc
import numpy as np

# ============================================================
# Manual Thomas-Fermi KEDF Implementation
# ============================================================

def thomas_fermi_kedf(rho):
    """
    Evaluate the Thomas-Fermi Kinetic Energy Density Functional.
    
    Parameters
    ----------
    rho : array_like
        Electron density at grid points.
    
    Returns
    -------
    zk : np.ndarray
        Energy density per particle (epsilon = e/rho) at each grid point.
        This is what LibXC returns as 'zk'.
    vrho : np.ndarray
        Functional derivative dE/drho (potential) at each grid point.
        This is what LibXC returns as 'vrho'.
    e_density : np.ndarray
        Energy density per volume at each grid point.
    """
    rho = np.array(rho, dtype=np.float64)
    
    # Thomas-Fermi constant: C_F = (3/10) * (3*pi^2)^(2/3)
    C_F = (3.0 / 10.0) * (3.0 * np.pi**2) ** (2.0 / 3.0)
    
    # Energy density per volume: e = C_F * rho^(5/3)
    e_density = C_F * rho ** (5.0 / 3.0)
    
    # Energy density per particle (what LibXC returns as 'zk'): 
    # zk = e / rho = C_F * rho^(2/3)
    # Handle rho=0 to avoid division by zero
    zk = np.where(rho > 0, C_F * rho ** (2.0 / 3.0), 0.0)
    
    # Functional potential: v = d(e)/d(rho) = (5/3) * C_F * rho^(2/3)
    vrho = np.where(rho > 0, (5.0 / 3.0) * C_F * rho ** (2.0 / 3.0), 0.0)
    
    return zk, vrho, e_density


# ============================================================
# Test Configuration
# ============================================================

# LibXC functional ID for Thomas-Fermi KEDF
funcid = 50  # LDA_K_TF in LibXC
print("=" * 60)
print("Thomas-Fermi Kinetic Energy Density Functional Evaluation")
print("=" * 60)
print(f'\nFunctional ID in LibXC: {funcid}')

# Test density values at grid points
rho = [0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0]
print(f'Density at grid points: {rho}')

# Print the Thomas-Fermi constant
C_F = (3.0 / 10.0) * (3.0 * np.pi**2) ** (2.0 / 3.0)
print(f'\nThomas-Fermi constant C_F = (3/10)*(3*pi^2)^(2/3) = {C_F:.10f}')

# ============================================================
# LibXC Evaluation
# ============================================================
print("\n" + "-" * 60)
print("LibXC Results")
print("-" * 60)

func = pylibxc.LibXCFunctional(funcid, "unpolarized")
print(f'Family: {func.get_family()}')
print(func.describe())

inp = {'rho': rho}
ret_libxc = func.compute(inp)

zk_libxc = ret_libxc['zk'].flatten()
vrho_libxc = ret_libxc['vrho'].flatten()

print('Energy density per particle (zk) at grid points:')
print(zk_libxc)
print('Functional potential (vrho) at grid points:')
print(vrho_libxc)

# ============================================================
# Manual Implementation Evaluation
# ============================================================
print("\n" + "-" * 60)
print("Manual Thomas-Fermi Implementation Results")
print("-" * 60)

zk_manual, vrho_manual, e_density_manual = thomas_fermi_kedf(rho)

print('Energy density per particle (zk) at grid points:')
print(zk_manual)
print('Functional potential (vrho) at grid points:')
print(vrho_manual)
print('Energy density per volume (e) at grid points:')
print(e_density_manual)

# ============================================================
# Comparison
# ============================================================
print("\n" + "=" * 60)
print("Comparison: Manual vs LibXC")
print("=" * 60)

print(f"\n{'rho':>8s} | {'zk (manual)':>14s} | {'zk (LibXC)':>14s} | {'zk diff':>12s} | {'vrho (manual)':>14s} | {'vrho (LibXC)':>14s} | {'vrho diff':>12s}")
print("-" * 105)

for i, r in enumerate(rho):
    zk_diff = abs(zk_manual[i] - zk_libxc[i])
    vrho_diff = abs(vrho_manual[i] - vrho_libxc[i])
    print(f"{r:8.4f} | {zk_manual[i]:14.10f} | {zk_libxc[i]:14.10f} | {zk_diff:12.2e} | {vrho_manual[i]:14.10f} | {vrho_libxc[i]:14.10f} | {vrho_diff:12.2e}")

# Check numerical agreement
zk_max_error = np.max(np.abs(zk_manual - zk_libxc))
vrho_max_error = np.max(np.abs(vrho_manual - vrho_libxc))

print(f"\nMax absolute error in zk:   {zk_max_error:.2e}")
print(f"Max absolute error in vrho: {vrho_max_error:.2e}")

tol = 1e-12
if zk_max_error < tol and vrho_max_error < tol:
    print(f"\n✅ SUCCESS: Manual implementation matches LibXC within tolerance {tol:.0e}")
else:
    print(f"\n❌ FAILURE: Discrepancy exceeds tolerance {tol:.0e}")

# ============================================================
# Additional: Verify the relationship between zk and vrho
# For Thomas-Fermi: vrho = (5/3) * zk
# ============================================================
print("\n" + "=" * 60)
print("Verification: vrho = (5/3) * zk for Thomas-Fermi")
print("=" * 60)

ratio = vrho_manual / zk_manual
print(f"vrho/zk ratios: {ratio}")
print(f"Expected ratio: {5.0/3.0:.10f}")
print(f"Max deviation from 5/3: {np.max(np.abs(ratio - 5.0/3.0)):.2e}")

# ============================================================
# Integration with PyFock API (if available)
# ============================================================
print("\n" + "=" * 60)
print("PyFock API Comparison (if available)")
print("=" * 60)

try:
    from pyfock import XC
    ret_pyfock = XC.func_compute(funcid, rho, use_gpu=False)
    zk_pyfock = np.array(ret_pyfock[0]).flatten()
    vrho_pyfock = np.array(ret_pyfock[1]).flatten()
    
    print('Energy density (zk) using PyFock:')
    print(zk_pyfock)
    print('Potential (vrho) using PyFock:')
    print(vrho_pyfock)
    
    print(f"\nMax |zk_pyfock - zk_libxc|:   {np.max(np.abs(zk_pyfock - zk_libxc)):.2e}")
    print(f"Max |vrho_pyfock - vrho_libxc|: {np.max(np.abs(vrho_pyfock - vrho_libxc)):.2e}")
    
except ImportError:
    print("PyFock not available. Skipping PyFock comparison.")