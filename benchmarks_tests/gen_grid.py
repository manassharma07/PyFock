import numpy as np

# ── Constants ────────────────────────────────────────────────────────
BRAGG_RADII = {
    1: 0.661,  2: 0.567,  3: 2.834,  4: 1.890,  5: 1.606,
    6: 1.417,  7: 1.228,  8: 1.134,  9: 0.945, 10: 0.882,
   11: 3.401, 12: 2.834, 13: 2.268, 14: 2.079, 15: 1.890,
   16: 1.890, 17: 1.701, 18: 1.701,
}

# ── Lebedev angular grids ───────────────────────────────────────────
def lebedev_6():
    """6-point Lebedev grid (exact up to l=3)."""
    pts = np.array([
        [ 1, 0, 0], [-1, 0, 0],
        [ 0, 1, 0], [ 0,-1, 0],
        [ 0, 0, 1], [ 0, 0,-1],
    ], dtype=np.float64)
    w = np.full(6, 4.0 * np.pi / 6.0)
    return pts, w

# ── Radial grid ──────────────────────────────────────────────────────
def treutler_ahlrichs_radial(n_rad, bragg_radius=1.0):
    k = np.arange(1, n_rad + 1)
    x = np.cos(k * np.pi / (n_rad + 1))

    inv_ln2 = 1.0 / np.log(2.0)
    r = inv_ln2 * (1 + x)**0.6 * np.log(2.0 / (1 - x))

    drdx = (inv_ln2 * 0.6 * (1 + x)**(-0.4) * np.log(2.0 / (1 - x))
          + inv_ln2 * (1 + x)**0.6 / (1 - x))

    cheb_w = np.pi / (n_rad + 1) * np.sin(k * np.pi / (n_rad + 1))

    w = cheb_w * r**2 * np.abs(drdx)

    r *= bragg_radius
    w *= bragg_radius**3

    idx = np.argsort(r)
    return r[idx], w[idx]

# ── Atomic grid ──────────────────────────────────────────────────────
def gen_atomic_grid(Z, center, n_rad=75, lebedev_fn=lebedev_6):
    bragg_r = BRAGG_RADII.get(Z, 1.0)
    r, w_rad = treutler_ahlrichs_radial(n_rad, bragg_r)
    ang_pts, w_ang = lebedev_fn()
    n_ang = len(w_ang)

    coords = np.empty((n_rad * n_ang, 3))
    weights = np.empty(n_rad * n_ang)

    for i in range(n_rad):
        s = i * n_ang
        e = s + n_ang
        coords[s:e] = np.asarray(center) + r[i] * ang_pts
        weights[s:e] = w_rad[i] * w_ang

    return coords, weights

# ── Becke partition ──────────────────────────────────────────────────
def becke_partition(grid_coords, atom_coords, atom_numbers, atom_indices):
    n_grid = len(grid_coords)
    n_atoms = len(atom_coords)

    dist_ga = np.linalg.norm(
        grid_coords[:, None, :] - atom_coords[None, :, :], axis=2)

    dist_aa = np.linalg.norm(
        atom_coords[:, None, :] - atom_coords[None, :, :], axis=2)

    bragg = np.array([BRAGG_RADII.get(Z, 1.0) for Z in atom_numbers])

    raw_P = np.ones((n_grid, n_atoms))

    for a in range(n_atoms):
        for b in range(n_atoms):
            if a == b:
                continue
            R_ab = dist_aa[a, b]
            if R_ab < 1e-14:
                continue

            mu = (dist_ga[:, a] - dist_ga[:, b]) / R_ab

            chi = bragg[a] / bragg[b]
            u = (chi - 1.0) / (chi + 1.0)
            denom = u * u - 1.0
            if abs(denom) > 1e-14:
                a_ab = u / denom
            else:
                a_ab = 0.0
            a_ab = np.clip(a_ab, -0.5, 0.5)

            nu = mu + a_ab * (1.0 - mu * mu)

            p = nu.copy()
            for _ in range(3):
                p = 1.5 * p - 0.5 * p**3

            s = 0.5 * (1.0 - p)
            raw_P[:, a] *= s

    total = raw_P.sum(axis=1)
    total = np.maximum(total, 1e-30)

    becke_w = raw_P[np.arange(n_grid), atom_indices] / total
    return becke_w

# ── Molecular grid ───────────────────────────────────────────────────
def gen_molecular_grid(atom_numbers, atom_coords, n_rad=75,
                       lebedev_fn=lebedev_6):
    atom_coords = np.asarray(atom_coords, dtype=np.float64)
    n_atoms = len(atom_numbers)

    all_coords = []
    all_weights = []
    all_atom_idx = []

    for a in range(n_atoms):
        c, w = gen_atomic_grid(atom_numbers[a], atom_coords[a],
                               n_rad, lebedev_fn)
        all_coords.append(c)
        all_weights.append(w)
        all_atom_idx.append(np.full(len(w), a, dtype=int))

    all_coords = np.vstack(all_coords)
    all_weights = np.concatenate(all_weights)
    all_atom_idx = np.concatenate(all_atom_idx)

    becke_w = becke_partition(all_coords, atom_coords,
                              atom_numbers, all_atom_idx)
    all_weights *= becke_w

    mask = np.abs(all_weights) > 1e-15
    return all_coords[mask], all_weights[mask]

# ── Tests ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    exact = np.pi**1.5

    # Test 1: single atom
    coords, weights = gen_molecular_grid([8], [[0.0, 0.0, 0.0]], n_rad=100)
    r2 = np.sum(coords**2, axis=1)
    result = np.sum(weights * np.exp(-r2))
    print(f"Single atom:      {result:.6f}  (exact: {exact:.6f})")

    # Test 2: two atoms far apart
    coords, weights = gen_molecular_grid(
        [8, 8],
        [[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]],
        n_rad=100)
    r2 = np.sum(coords**2, axis=1)
    result = np.sum(weights * np.exp(-r2))
    print(f"Two atoms (far):  {result:.6f}  (exact: {exact:.6f})")

    # Test 3: H2
    d = 1.4
    coords, weights = gen_molecular_grid(
        [1, 1],
        [[0, 0, 0], [0, 0, d]],
        n_rad=100)
    r2_A = np.sum(coords**2, axis=1)
    r2_B = np.sum((coords - [0, 0, d])**2, axis=1)
    f = np.exp(-r2_A) + np.exp(-r2_B)
    result = np.sum(weights * f)
    print(f"H2 two Gaussians: {result:.6f}  (approx: {2*exact:.6f})")
    print(f"Total grid points: {len(weights)}")