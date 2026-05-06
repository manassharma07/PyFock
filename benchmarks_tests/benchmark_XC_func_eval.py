import pylibxc
from pyfock import XC
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

# (funcid, needs_sigma)
FUNCTIONALS = [
    (1,   False),   # LDA_X
    (7,   False),   # LDA_C_VWN
    (12,  False),   # LDA_C_PW
    (13,  False),   # LDA_C_PW_MOD
    (101, True),    # GGA_X_PBE
    (106, True),    # GGA_X_B88
    (109, True),    # GGA_X_PW91
    (130, True),    # GGA_C_PBE
    (131, True),    # GGA_C_LYP
    (132, True),    # GGA_C_P86
    (134, True),    # GGA_C_PW91
]

RHO_CASES = {
    "uniform_low"     : np.full(1000, 0.1),
    "uniform_mid"     : np.full(1000, 1.0),
    "uniform_high"    : np.full(1000, 10.0),
    "linspace"        : np.linspace(1e-6, 10.0, 5000),
    "logspace"        : np.logspace(-4, 1, 5000),
    "random_uniform"  : np.random.default_rng(0).uniform(1e-6, 5.0, 10_000),
    "random_lognormal": np.abs(np.random.default_rng(1).lognormal(0, 1, 10_000)) + 1e-9,
    "exponential"     : np.exp(-np.linspace(0, 10, 5000)) + 1e-9,
    "very_low"        : np.linspace(1e-6, 1e-3, 2000),
    "very_high"       : np.linspace(50.0, 200.0, 2000),
}

TOLERANCE = 1e-7

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def make_sigma(rho):
    """Simple proxy for sigma (∇ρ·∇ρ) using finite differences."""
    return np.maximum(np.gradient(rho) ** 2, 0.0)


def compare(label, libxc_vals, pyfock_vals):
    """Compare two arrays, print pass/fail, return result dict."""
    max_err = np.abs(np.asarray(libxc_vals).ravel() - np.asarray(pyfock_vals).ravel()).max()
    passed  = max_err < TOLERANCE
    print(f"    {'PASS ✓' if passed else 'FAIL ✗'}  {label:30s}  max_err={max_err:.3e}")
    return {"label": label, "max_err": max_err, "passed": passed}


# ──────────────────────────────────────────────────────────────────────────────
# Benchmark
# ──────────────────────────────────────────────────────────────────────────────

def benchmark(funcid, needs_sigma):
    func_name = pylibxc.LibXCFunctional(funcid, "unpolarized").get_name()
    print(f"\n{'='*60}\n  {func_name}  (ID={funcid})\n{'='*60}")

    results = []
    for case_name, rho in RHO_CASES.items():
        sigma = make_sigma(rho) if needs_sigma else None
        print(f"\n  [{case_name}]  n={len(rho)}  rho in [{rho.min():.2e}, {rho.max():.2e}]")

        # Run both implementations
        libxc_inp = {"rho": rho, **({"sigma": sigma} if sigma is not None else {})}
        libxc_out = pylibxc.LibXCFunctional(funcid, "unpolarized").compute(libxc_inp)
        pyfock_out = XC.func_compute(funcid, rho, sigma=sigma, use_gpu=False)

        # Compare zk and vrho (common to LDA and GGA)
        results.append(compare(f"{case_name}/zk",   libxc_out["zk"],   pyfock_out[0]))
        results.append(compare(f"{case_name}/vrho",  libxc_out["vrho"], pyfock_out[1]))

        # Compare vsigma for GGA functionals
        if needs_sigma and "vsigma" in libxc_out and len(pyfock_out) > 2:
            results.append(compare(f"{case_name}/vsigma", libxc_out["vsigma"], pyfock_out[2]))

    return results


def print_summary(all_results):
    passed = sum(r["passed"] for r in all_results)
    failed = len(all_results) - passed

    print(f"\n{'='*60}")
    print(f"  SUMMARY:  {passed} passed,  {failed} failed  (total {len(all_results)})")

    if failed:
        print("\n  Failed cases:")
        for r in all_results:
            if not r["passed"]:
                print(f"    {r['label']:40s}  max_err={r['max_err']:.3e}")

    print(f"\n  {'ALL PASSED ✓' if not failed else f'{failed} FAILED ✗'}")
    print(f"{'='*60}\n")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    all_results = []
    for funcid, needs_sigma in FUNCTIONALS:
        all_results.extend(benchmark(funcid, needs_sigma))
    print_summary(all_results)
