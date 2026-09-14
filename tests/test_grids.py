"""Tests of the XC integration grids (pyfock.Grids).

* the native ``'treutler'`` scheme (default) gives the same set of points and weights as PySCF's grid of the
  same level (Treutler-Ahlrichs radial grids, Lebedev angular grids, region-wise angular pruning, Becke/Treutler
  partitioning), including its variants (no pruning, Becke's or no size adjustment, per-element overrides);
* the numgrid ``'dense'`` preset reproduces the previous PyFock grids exactly and the ``'compact'`` preset gives
  grids of a size similar to PySCF's;
* the grids integrate the SANO guess density to the electron count;
* box sorting is a permutation matching PySCF's grouping; pruning keeps ``atom_idx`` consistent;
* an SCF on the default grid reproduces the SCF on PySCF's grid.
"""
import contextlib
import io
import sys

import numpy as np
import pytest

from pyfock import Basis, DFT, Integrals, Mol, Grids

GridsMod = sys.modules["pyfock.Grids"]

from tests.conftest import H2O_XYZ, ROOT  # noqa: E402

SNCL4_XYZ = ROOT / "benchmarks_tests" / "SnCl4.xyz"


def _pyscf():
    return pytest.importorskip("pyscf")


def _water():
    return Mol(coordfile=str(H2O_XYZ))


def _pyscf_mol(mol):
    from pyscf import gto
    molP = gto.Mole()
    molP.atom = [[sym, tuple(c)] for sym, c in zip(mol.atomicSpecies, np.asarray(mol.coordsBohrs))]
    molP.unit = "Bohr"
    molP.basis = "sto-3g"
    molP.verbose = 0
    molP.build()
    return molP


def _pyscf_grid(mol, level, sort=False, pruning="regions", size_adjustment="treutler"):
    _pyscf()
    from pyscf.dft import gen_grid, radi
    g = gen_grid.Grids(_pyscf_mol(mol))
    g.level = level
    g.alignment = 0
    g.prune = gen_grid.nwchem_prune if pruning == "regions" else None
    g.radii_adjust = {"treutler": radi.treutler_atomic_radii_adjust, "becke": radi.becke_atomic_radii_adjust, None: None}[size_adjustment]
    g.build(sort_grids=sort)
    return g


def _assert_same_point_set(coords, weights, ref_coords, ref_weights):
    from scipy.spatial import cKDTree
    assert coords.shape == ref_coords.shape
    dist, idx = cKDTree(ref_coords).query(coords)
    assert len(set(idx.tolist())) == coords.shape[0]  # one-to-one
    assert dist.max() < 1e-10
    assert np.abs(weights - ref_weights[idx]).max() < 1e-11 * np.abs(ref_weights).max()


@pytest.mark.parametrize("xyz, levels", [(H2O_XYZ, (1, 3, 5)), (SNCL4_XYZ, (2, 3))])
def test_treutler_scheme_reproduces_pyscf_grids(xyz, levels):
    mol = Mol(coordfile=str(xyz))
    for level in levels:
        g = Grids(mol, level=level, ncores=2, sort=False, verbose=False)
        assert g.scheme == "treutler" and g.preset is None
        ref = _pyscf_grid(mol, level)
        _assert_same_point_set(g.coords, g.weights, ref.coords, ref.weights)
        assert np.bincount(g.atom_idx, minlength=mol.natoms).tolist() == [
            GridsMod.single_atom_grid(int(z), level)[1].shape[0] for z in mol.Zcharges]


def test_treutler_scheme_variants_match_pyscf_options():
    mol = _water()
    for pruning, adjust in ((None, "treutler"), ("regions", "becke"), ("regions", None)):
        g = Grids(mol, level=2, ncores=2, sort=False, verbose=False, pruning=pruning, size_adjustment=adjust)
        ref = _pyscf_grid(mol, 2, pruning=pruning, size_adjustment=adjust)
        _assert_same_point_set(g.coords, g.weights, ref.coords, ref.weights)
    # per-element override and the level tables
    g = Grids(mol, level=3, ncores=2, verbose=False, points_per_element={"O": (60, 194), 1: (40, 110)})
    assert g.element_points == {"O": (60, 194), "H": (40, 110)}
    assert g.size == 60 * 194 * 0 + sum(GridsMod.single_atom_grid(z, 3, n_rad=n, n_ang=a)[1].shape[0]
                                        for z, (n, a) in ((8, (60, 194)), (1, (40, 110)), (1, (40, 110))))
    assert GridsMod.radial_points_for_level(6, 3) == 75 and GridsMod.angular_points_for_level(6, 3) == 302
    assert GridsMod.radial_points_for_level(17, 3) == 80 and GridsMod.angular_points_for_level(17, 3) == 434
    assert GridsMod.radial_points_for_level(0, 3) == 50  # ghost atoms count as period 0


def test_treutler_radial_grid_and_lebedev_tables():
    _pyscf()
    from pyscf.dft import gen_grid, radi
    for n, charge in ((50, 1), (75, 6), (95, 50)):
        r, dr = GridsMod.treutler_m4_radial_grid(n, charge)
        r_ref, dr_ref = radi.treutler_ahlrichs(n, charge)
        assert np.allclose(r, r_ref, rtol=1e-9, atol=1e-16) and np.allclose(dr, dr_ref, rtol=1e-9, atol=1e-18)
        assert np.all(np.diff(r) > 0)
    # exp(-r^2) integrates to pi^(3/2) on the radial grid
    r, dr = GridsMod.treutler_m4_radial_grid(75, 6)
    assert abs(np.sum(4 * np.pi * r ** 2 * dr * np.exp(-r ** 2)) - np.pi ** 1.5) < 1e-10
    for n in (50, 302, 590):
        xyz, w = GridsMod.lebedev_grid(n)
        ref = gen_grid.MakeAngularGrid(n)
        _assert_same_point_set(xyz, w, ref[:, :3], ref[:, 3])
    assert np.array_equal(GridsMod.region_angular_sizes(6, np.array([0.05, 0.5, 1.0, 2.0, 10.0]), 302),
                          gen_grid.nwchem_prune(6, np.array([0.05, 0.5, 1.0, 2.0, 10.0]), 302))


def test_dense_preset_reproduces_the_previous_pyfock_grid():
    mol = _water()
    grid_basis = Basis(mol, {"all": Basis.load(mol=mol, basis_name="def2-QZVP")})
    g = Grids(mol, basis=grid_basis, level=3, ncores=2, preset="dense", sort=False, verbose=False)
    assert g.scheme == "numgrid"
    # values of the previous implementation (radial precision 1e-13, angular 110-590 for O, 86-590 for H)
    assert g.size == 107530
    assert abs(g.weights.sum() - 34202.0389775492) < 1e-6
    assert g.radial_points == {"O": 145, "H": 113}
    assert g.angular_points == {"O": (110, 590), "H": (86, 590)}
    assert g.radial_precision == 1e-13
    # the basis is optional: def2-QZVP is built internally
    g2 = Grids(mol, level=3, ncores=2, scheme="numgrid", preset="dense", sort=False, verbose=False)
    assert np.array_equal(g.coords, g2.coords) and np.array_equal(g.weights, g2.weights)


def test_numgrid_compact_preset_angular_table_and_sizes():
    mol = _water()
    g = Grids(mol, level=3, ncores=2, scheme="numgrid", verbose=False)
    assert g.preset == "compact" and g.is_sorted
    assert g.angular_points == {"O": (50, 302), "H": (50, 302)}
    assert g.radial_precision == GridsMod.RADIAL_PRECISION[3] == 1e-8
    # numgrid options select the numgrid scheme without an explicit `scheme`
    g = Grids(mol, level=3, ncores=2, radial_precision=1e-10, angular_points=(86, 434), verbose=False)
    assert g.scheme == "numgrid" and g.radial_precision == 1e-10 and g.angular_points == {"O": (86, 434), "H": (86, 434)}
    _pyscf()
    for level in (1, 2, 3, 4, 5):
        g = Grids(mol, level=level, ncores=2, scheme="numgrid", verbose=False)
        ratio = g.size / _pyscf_grid(mol, level).weights.shape[0]
        assert 0.6 < ratio < 1.6, (level, g.size)
        for symbol, charge in (("O", 8), ("H", 1)):
            assert g.angular_points[symbol][1] == GridsMod.angular_points_for_level(charge, level)


def test_grids_integrate_the_guess_density_to_the_electron_count():
    mol = _water()
    basis = Basis(mol, {"all": Basis.load(mol=mol, basis_name="def2-SVP")})
    dft = DFT(mol, basis, xc=[1, 7], ncores=2)
    with contextlib.redirect_stdout(io.StringIO()):
        dmat = dft.guess_dmat(mol, basis)
    for kwargs, tol in (({}, 1e-6), ({"scheme": "numgrid"}, 2e-4), ({"scheme": "numgrid", "preset": "dense"}, 1e-6)):
        g = Grids(mol, level=3, ncores=2, verbose=False, **kwargs)
        ao = Integrals.bf_val_helpers.eval_bfs(basis, g.coords)
        nelec = np.einsum("ij,mi,mj,m->", dmat, ao, ao, g.weights)
        assert abs(nelec - mol.nelectrons) < tol, (kwargs, nelec)


def test_sorting_is_a_permutation_and_matches_pyscf_grouping():
    mol = _water()
    unsorted = Grids(mol, level=3, ncores=2, sort=False, verbose=False)
    grouped = Grids(mol, level=3, ncores=2, sort=True, verbose=False)
    assert not unsorted.is_sorted and grouped.is_sorted
    perm = GridsMod.box_grouping_order(np.asarray(mol.coordsBohrs), unsorted.coords)
    assert np.array_equal(unsorted.coords[perm], grouped.coords)
    assert np.array_equal(unsorted.weights[perm], grouped.weights)
    assert np.array_equal(unsorted.atom_idx[perm], grouped.atom_idx)
    unsorted.sort()
    assert np.array_equal(unsorted.coords, grouped.coords)
    _pyscf()
    from pyscf.dft import gen_grid
    assert np.array_equal(perm, gen_grid.arg_group_grids(_pyscf_mol(mol), grouped.coords[np.argsort(perm)]))


def test_prune_by_mask_keeps_atom_idx_consistent_and_inputs_are_validated():
    mol = _water()
    g = Grids(mol, level=2, ncores=2, verbose=False)
    keep = g.weights > np.median(g.weights)
    kept_atoms = g.atom_idx[keep]
    g.prune_by_mask(keep)
    assert g.size == keep.sum() and np.array_equal(g.atom_idx, kept_atoms)
    with pytest.raises(ValueError):
        Grids(None, level=3)
    with pytest.raises(ValueError):
        Grids(mol, level=10, verbose=False)
    with pytest.raises(ValueError):
        Grids(mol, level=2, preset="dense", verbose=False)
    with pytest.raises(ValueError):
        Grids(mol, level=3, scheme="lmg", verbose=False)
    with pytest.raises(ValueError):
        Grids(mol, level=3, pruning="sg1", verbose=False)


def _scf(mol, basis, aux, **kwargs):
    dft = DFT(mol, basis, aux, xc=[1, 7], conv_crit=1e-9, ncores=2, **kwargs)
    dft.sao = True
    dft.DF_algo = 11
    with contextlib.redirect_stdout(io.StringIO()):
        energy, _ = dft.scf()
    return energy, dft


def test_scf_on_the_default_grid_reproduces_the_pyscf_grid_energy():
    _pyscf()
    from pyscf import dft as pyscf_dft, gto
    mol = _water()
    basis = Basis(mol, {"all": Basis.load(mol=mol, basis_name="def2-SVP")})
    aux = Basis(mol, {"all": Basis.load(mol=mol, basis_name="def2-universal-jfit")})
    molP = gto.M(atom=str(H2O_XYZ), basis="def2-SVP", verbose=0)
    grids = pyscf_dft.gen_grid.Grids(molP)
    grids.level = 3
    grids.build()
    e_pyscf_grid, _ = _scf(mol, basis, aux, grids=grids)
    e_default, dft_default = _scf(mol, basis, aux)
    # same grid up to PyFock's density pruning of the generated grid
    assert abs(e_default - e_pyscf_grid) < 1e-8
    assert dft_default.grids.scheme == "treutler"
    assert 0.8 < dft_default.grids.coords.shape[0] / grids.weights.shape[0] <= 1.0
    dft = DFT(mol, basis, aux, xc=[1, 7], conv_crit=1e-9, ncores=2)
    dft.sao = True
    dft.DF_algo = 11
    dft.grids_scheme = "numgrid"
    with contextlib.redirect_stdout(io.StringIO()):
        e_numgrid, _ = dft.scf()
    assert dft.grids.scheme == "numgrid" and abs(e_numgrid - e_pyscf_grid) < 2e-5

def test_becke_weight_gradient_matches_finite_differences():
    """d(Becke factor)/dR from the Numba kernel against finite differences of the forward partitioning.

    Both ways the nuclei enter are exercised: the explicit dependence of the partitioning on every
    nuclear position, and the grid points translating rigidly with their own atom -- which is why the
    displaced geometry below shifts the points too.
    """
    import numpy as np
    from pyfock import Grids, Mol
    from pyfock.Grids import becke_partition_weights, becke_weight_gradient, size_adjustment_table

    mol = Mol(atoms=[['O', 0.0, 0.0, 0.117], ['H', 0.0, 0.757, -0.469], ['H', 0.0, -0.757, -0.469],
                     ['O', 2.9, 0.1, 0.05], ['H', 3.4, 0.8, 0.3], ['H', 3.3, -0.6, -0.2]])
    grids = Grids(mol, level=0, verbose=False)
    coords, atom_idx = grids.coords, grids.atom_idx
    centres = np.asarray(mol.coordsBohrs, dtype=np.float64).reshape(-1, 3)
    table = size_adjustment_table(np.asarray(mol.Zcharges), 'treutler')

    rng = np.random.default_rng(0)
    cotangent = rng.standard_normal(coords.shape[0])
    analytic = becke_weight_gradient(coords, atom_idx, centres, table, cotangent)

    def value(displaced):
        points = coords + (displaced - centres)[atom_idx]
        return float(cotangent @ becke_partition_weights(points, atom_idx, displaced, table))

    step = 1e-5
    numeric = np.zeros_like(analytic)
    for atom in range(centres.shape[0]):
        for direction in range(3):
            plus, minus = centres.copy(), centres.copy()
            plus[atom, direction] += step
            minus[atom, direction] -= step
            numeric[atom, direction] = (value(plus) - value(minus)) / (2 * step)

    scale = max(float(np.abs(numeric).max()), 1.0)
    assert np.allclose(analytic, numeric, atol=1e-6 * scale)
