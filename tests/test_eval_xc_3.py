"""
Unit tests for the CPU exchange-correlation algorithm 3 (:mod:`pyfock.Integrals.eval_xc_3`,
:mod:`pyfock.Integrals.eval_xc_grad_3`, :mod:`pyfock.XC.pointwise`).

* The leaf AO kernel (values, gradients, second derivatives) against the per-function kernels of
  :mod:`pyfock.Integrals.bf_val_helpers`.
* The point-wise Numba functionals against the array implementations they were transcribed from.
* The grid partition: every point in exactly one leaf, leaf shell lists equal to the per-point
  screening criterion, macro lists the union of their leaves.
* Energies and potential matrices against :func:`eval_xc_2` for LDA, GGA, a global hybrid and a
  meta-GGA, with and without the AO cache, with one and two threads, through the one-pass and the
  two-pass path.
* The nuclear-gradient contribution against :func:`eval_xc_grad_2`, with and without the grid
  response, and the gradient through ``DFT_Grad`` with the default ``XC_algo``.
"""
from __future__ import annotations

import numpy as np
import pytest

from pyfock import Basis, Grids, Mol, XC, Integrals
from pyfock.Integrals import bf_val_helpers as bfh
from pyfock.Integrals import eval_xc_3_module as m3
from pyfock.Integrals import eval_xc_3_helpers as h3
from pyfock.XC import pointwise as pw


WATER = [["O", 0.0, 0.0, 0.117], ["H", 0.0, 0.757, -0.467], ["H", 0.0, -0.757, -0.467]]
TWO_WATERS = list(WATER) + [[s, x + 2.9, y + 0.3, z - 0.2] for s, x, y, z in WATER]


def _system(atoms=TWO_WATERS, basis_name="def2-SVP"):
    mol = Mol(atoms=atoms)
    return mol, Basis(mol, {"all": Basis.load(mol=mol, basis_name=basis_name)})


def _dmat(basis, seed=3):
    rng = np.random.default_rng(seed)
    n = basis.bfs_nao
    c = rng.normal(size=(n, max(2, n // 6)))
    return 0.2 * c @ c.T


@pytest.fixture(scope="module")
def grid_system():
    mol, basis = _system()
    grids = Grids(mol, level=2, ncores=1)
    return mol, basis, grids


# ---------------------------------------------------------------------------------------
# AO kernel
# ---------------------------------------------------------------------------------------
def test_leaf_ao_kernel_matches_per_function_kernels():
    mol, basis = _system(basis_name="def2-TZVP")   # d and f functions
    sd = m3.shell_data(basis)
    rng = np.random.default_rng(0)
    coords = rng.uniform(-3.0, 4.0, (40, 3))
    shells = np.arange(sd['nsh'], dtype=np.int64)
    out = np.empty((10, coords.shape[0], basis.bfs_nao))
    # no per-point cutoff, so that the comparison covers every point
    rcut2 = np.full_like(sd['sh_rcut2'], np.inf)
    h3._eval_ao_block(coords, 0, coords.shape[0], shells, 0, sd['nsh'], sd['sh_coords'], rcut2, sd['sh_nprim'],
                      sd['sh_expnts'], sd['sh_bf0'], sd['sh_nbf'], sd['sh_L'], sd['bf_lmn'], sd['bf_coef'], 10, out)
    bd = bfh.pack_bfs_data(basis)
    ref_v, ref_g, ref_h = bfh.eval_bfs_grad_and_hess_sparse_internal_serial(
        bd[0], bd[1], bd[2], bd[3], bd[4], bd[5], bd[6], coords, np.arange(basis.bfs_nao))
    scale = np.abs(ref_v).max()
    assert np.abs(out[0] - ref_v).max() < 1e-12 * scale
    assert np.abs(out[1:4] - ref_g).max() < 1e-11 * np.abs(ref_g).max()
    assert np.abs(out[4:10] - ref_h).max() < 1e-10 * np.abs(ref_h).max()
    # values only / values and gradients give the same numbers
    out1 = np.empty((1, coords.shape[0], basis.bfs_nao))
    out4 = np.empty((4, coords.shape[0], basis.bfs_nao))
    for nd, o in ((1, out1), (4, out4)):
        h3._eval_ao_block(coords, 0, coords.shape[0], shells, 0, sd['nsh'], sd['sh_coords'], rcut2, sd['sh_nprim'],
                          sd['sh_expnts'], sd['sh_bf0'], sd['sh_nbf'], sd['sh_L'], sd['bf_lmn'], sd['bf_coef'], nd, o)
    assert np.array_equal(out1[0], out[0]) and np.array_equal(out4, out[:4])


# ---------------------------------------------------------------------------------------
# Point-wise functionals
# ---------------------------------------------------------------------------------------
def _rho_sigma_tau(n=400, seed=1):
    rng = np.random.default_rng(seed)
    rho = 10.0 ** rng.uniform(-7, 1.5, n)
    sigma = (10.0 ** rng.uniform(-8, 2, n)) * rho ** (8.0 / 3.0)
    tau = (10.0 ** rng.uniform(-2, 1, n)) * sigma / (8 * rho) + 1e-16
    return rho, sigma, tau


@pytest.mark.parametrize("fid", sorted(pw.POINTWISE_IDS - {134}))
def test_pointwise_mgga_matches_array_version(fid):
    rho, sigma, tau = _rho_sigma_tau()
    ref = XC.func_compute(fid, rho, sigma=sigma, tau=tau, use_gpu=False)
    e, vr, vs, vt = (np.zeros_like(rho) for _ in range(4))
    pw.eval_pointwise_mgga(fid, 1.0, rho, sigma, tau, e, vr, vs, vt)
    assert np.abs(e - ref[0]).max() <= 1e-13 * np.abs(ref[0]).max()
    # derivatives: finite differences in both, so the comparison is at the level of their rounding noise
    for mine, theirs, tol in ((vr, ref[1], 1e-9), (vs, ref[2], 1e-6), (vt, ref[3], 1e-4)):
        assert np.abs(mine - theirs).max() <= tol * np.abs(theirs).max()


def test_pointwise_pw91_matches_array_version():
    rho, sigma, _ = _rho_sigma_tau()
    ref = XC.func_compute(134, rho, sigma=sigma, use_gpu=False)
    e, vr, vs = (np.zeros_like(rho) for _ in range(3))
    pw.eval_pointwise_gga(134, 1.0, rho, sigma, e, vr, vs)
    for mine, theirs in ((e, ref[0]), (vr, ref[1]), (vs, ref[2])):
        assert np.abs(mine - theirs).max() <= 1e-12 * np.abs(theirs).max()


def test_inkernel_functional_set():
    assert pw.POINTWISE_IDS <= m3.INKERNEL_FUNCTIONALS
    assert set(XC.get_implemented_ids()) - {402, 406, 475} <= m3.INKERNEL_FUNCTIONALS


# ---------------------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------------------
def test_plan_partition_and_screening(grid_system):
    mol, basis, grids = grid_system
    coords, weights = np.asarray(grids.coords), np.asarray(grids.weights)
    plan = m3.build_xc_plan(basis, coords, weights, ncores=1, leaf_size=64, leaves_per_macro=4)
    n = coords.shape[0]
    assert plan.ngrids == n and sorted(plan.perm.tolist()) == list(range(n))
    assert plan.leaf_lo[0] == 0 and plan.leaf_hi[-1] == n
    assert np.all(plan.leaf_lo[1:] == plan.leaf_hi[:-1]) and np.all(plan.leaf_hi - plan.leaf_lo <= 64)
    assert np.allclose(plan.coords, coords[plan.perm]) and np.allclose(plan.weights, weights[plan.perm])
    sd = plan.shell
    for il in range(plan.nleaf):
        pts = plan.coords[plan.leaf_lo[il]:plan.leaf_hi[il]]
        dist = np.linalg.norm(pts[:, None, :] - sd['sh_coords'][None, :, :], axis=2)
        expected = np.flatnonzero((dist < sd['sh_rcut'][None, :]).any(axis=0))
        got = plan.leaf_sh[plan.leaf_sh_ptr[il]:plan.leaf_sh_ptr[il + 1]]
        assert np.array_equal(got, expected)
    for ib in range(plan.nmacro):
        leaves = range(plan.macro_leaf_lo[ib], plan.macro_leaf_hi[ib])
        union = np.unique(np.concatenate([plan.leaf_sh[plan.leaf_sh_ptr[il]:plan.leaf_sh_ptr[il + 1]] for il in leaves]))
        assert np.array_equal(plan.macro_sh[plan.macro_sh_ptr[ib]:plan.macro_sh_ptr[ib + 1]], union)
        bfs = np.concatenate([np.arange(sd['sh_bf0'][s], sd['sh_bf0'][s] + sd['sh_nbf'][s]) for s in union])
        assert np.array_equal(plan.macro_bf[plan.macro_bf_ptr[ib]:plan.macro_bf_ptr[ib + 1]], bfs)
    assert plan.matches(coords, weights) and not plan.matches(coords[:-1], weights[:-1])
    bin_ptr, bin_blocks = plan.bins(3, False)
    assert sorted(bin_blocks.tolist()) == list(range(plan.nmacro)) and bin_ptr[-1] == plan.nmacro


# ---------------------------------------------------------------------------------------
# Energy and potential
# ---------------------------------------------------------------------------------------
def _reference_xc2(basis, dmat, weights, coords, funcid, use_libxc=False):
    blocksize = 500
    ngrids = coords.shape[0]
    nblocks = ngrids // blocksize
    lni, cni = bfh.nonzero_ao_indices(basis, coords, blocksize, nblocks, ngrids)
    return Integrals.eval_xc_2(basis, dmat, weights, coords, funcid, use_libxc, ncores=1, blocksize=blocksize,
                               list_nonzero_indices=lni, count_nonzero_indices=cni, print_nelec=False)


@pytest.mark.parametrize("funcname", ["LDA", "PBE", "BLYP", "B3LYP", "R2SCAN", "TPSS"])
def test_energy_and_potential_match_algorithm_2(grid_system, funcname):
    mol, basis, grids = grid_system
    coords, weights = np.asarray(grids.coords), np.asarray(grids.weights)
    dmat = _dmat(basis)
    funcid = XC.resolve_functional(funcname)
    e2, v2 = _reference_xc2(basis, dmat, weights, coords, funcid)
    plan = m3.build_xc_plan(basis, coords, weights, ncores=1, leaf_size=64, leaves_per_macro=4)
    e3, v3 = Integrals.eval_xc_3(basis, dmat, weights, coords, funcid, ncores=1, plan=plan, print_nelec=False)
    assert abs(e3 - e2) < 2e-8 * max(1.0, abs(e2))
    assert np.abs(v3 - v2).max() < 2e-8 * max(1.0, np.abs(v2).max())
    assert np.array_equal(v3, v3.T)
    # two threads give exactly the same numbers (fixed reduction order)
    e3b, v3b = Integrals.eval_xc_3(basis, dmat, weights, coords, funcid, ncores=2, plan=plan, print_nelec=False)
    assert e3b == e3 and np.array_equal(v3b, v3)
    # the AO cache and the two-pass path give the same numbers as the one-pass evaluation
    nd = 1 if XC.xc_semilocal_family(funcid) == 1 else 4
    cache = plan.build_ao_cache(nd, 1)
    e3c, v3c = Integrals.eval_xc_3(basis, dmat, weights, coords, funcid, ncores=2, list_ao_values=cache, print_nelec=False)
    assert abs(e3c - e3) < 1e-12 and np.abs(v3c - v3).max() < 1e-12
    plan.cache = None
    e3t, v3t = Integrals.eval_xc_3(basis, dmat, weights, coords, funcid, ncores=2, plan=plan, print_nelec=False,
                                   force_two_pass=True)
    assert abs(e3t - e3) < 1e-10 and np.abs(v3t - v3).max() < 1e-10


def test_plan_is_memoized_and_rebuilt_for_new_arrays(grid_system):
    mol, basis, grids = grid_system
    coords, weights = np.asarray(grids.coords), np.asarray(grids.weights)
    dmat = _dmat(basis)
    plan_a = m3._get_plan(basis, coords, weights, 1, 256, 16, True)
    plan_b = m3._get_plan(basis, coords, weights, 1, 256, 16, True)
    assert plan_a is plan_b
    plan_c = m3._get_plan(basis, coords[:-5].copy(), weights[:-5].copy(), 1, 256, 16, True)
    assert plan_c is not plan_a and plan_c.ngrids == coords.shape[0] - 5
    e_a, _ = Integrals.eval_xc_3(basis, dmat, weights, coords, [1, 7], ncores=1, print_nelec=False)
    e_b, _ = Integrals.eval_xc_3(basis, dmat, weights, coords, [1, 7], ncores=1, plan=plan_a, print_nelec=False)
    assert e_a == e_b


def test_libxc_path_matches_native(grid_system):
    pylibxc = pytest.importorskip("pylibxc")
    if not hasattr(pylibxc, "LibXCFunctional"):
        pytest.skip("pylibxc without LibXCFunctional (placeholder package)")
    mol, basis, grids = grid_system
    coords, weights = np.asarray(grids.coords), np.asarray(grids.weights)
    dmat = _dmat(basis)
    e_n, v_n = Integrals.eval_xc_3(basis, dmat, weights, coords, [101, 130], ncores=2, print_nelec=False)
    e_l, v_l = Integrals.eval_xc_3(basis, dmat, weights, coords, [101, 130], use_libxc=True, ncores=2, print_nelec=False)
    assert abs(e_l - e_n) < 5e-6 * abs(e_n)     # the native PBE constants are those of the reference, LibXC's differ slightly
    assert np.abs(v_l - v_n).max() < 5e-6 * np.abs(v_n).max()


# ---------------------------------------------------------------------------------------
# Gradient
# ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("funcname,response", [("LDA", False), ("PBE", False), ("PBE", True), ("LDA", True),
                                               ("B3LYP", False), ("R2SCAN", False), ("R2SCAN", True)])
def test_gradient_matches_algorithm_2(grid_system, funcname, response):
    mol, basis, grids = grid_system
    coords, weights = np.asarray(grids.coords), np.asarray(grids.weights)
    dmat = _dmat(basis)
    funcid = XC.resolve_functional(funcname)
    blocksize = 500
    ngrids = coords.shape[0]
    lni, cni = bfh.nonzero_ao_indices(basis, coords, blocksize, ngrids // blocksize, ngrids)
    r2 = Integrals.eval_xc_grad_2(basis, dmat, weights, coords, funcid=funcid, ncores=1, blocksize=blocksize,
                                  list_nonzero_indices=lni, count_nonzero_indices=cni, grids=grids, grid_response=response)
    r3 = Integrals.eval_xc_grad_3(basis, dmat, weights, coords, funcid=funcid, ncores=2, grids=grids,
                                  grid_response=response, leaf_size=64, leaves_per_macro=4)
    if response:
        d2, a2 = r2
        d3, a3 = r3
        # the translation term of a random (unphysically rough) density is the most sensitive quantity
        # to the per-point cutoff of algorithm 3 (see the module documentation of eval_xc_3)
        assert np.abs(a3 - a2).max() < 5e-7 * max(1.0, np.abs(a2).max())
    else:
        d2, d3 = r2, r3
    assert d3.shape == (3, basis.bfs_nao)
    assert np.abs(d3 - d2).max() < 1e-7 * max(1.0, np.abs(d2).max())
    # one thread: identical numbers
    r3b = Integrals.eval_xc_grad_3(basis, dmat, weights, coords, funcid=funcid, ncores=1, grids=grids,
                                   grid_response=response, leaf_size=64, leaves_per_macro=4)
    d3b = r3b[0] if response else r3b
    assert np.array_equal(d3b, d3)
    if not response:
        r3t = Integrals.eval_xc_grad_3(basis, dmat, weights, coords, funcid=funcid, ncores=2, grids=grids,
                                       leaf_size=64, leaves_per_macro=4, force_two_pass=True)
        assert np.abs(r3t - d3).max() < 1e-9 * max(1.0, np.abs(d3).max())


def test_scf_and_forces_through_dft_objects():
    """The default CPU algorithm is 3; its SCF energy and forces agree with algorithm 2."""
    from pyfock import DFT, DFT_Grad
    mol, basis = _system(atoms=WATER)
    auxbasis = Basis(mol, {"all": Basis.load(mol=mol, basis_name="def2-universal-jfit")})
    results = {}
    for algo in (2, 3):
        dft = DFT(mol, basis, auxbasis, xc="PBE")
        dft.conv_crit = 1e-9
        dft.max_itr = 40
        dft.ncores = 2
        dft.gridsLevel = 2
        dft.XC_algo = algo
        energy, dmat = dft.scf()
        forces = np.asarray(DFT_Grad(dft).calculate()["forces"])
        results[algo] = (energy, forces, dft)
    assert results[3][2].XC_algo == 3 and results[3][2].xc_plan is not None
    assert abs(results[3][0] - results[2][0]) < 1e-7
    assert np.abs(results[3][1] - results[2][1]).max() < 1e-6
    # default selection and the AO cache (built for the SCF, released afterwards)
    import contextlib
    import io
    dft = DFT(mol, basis, auxbasis, xc="PBE")
    dft.conv_crit = 1e-9
    dft.max_itr = 40
    dft.ncores = 2
    dft.gridsLevel = 2
    dft.save_ao_values = True
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        energy, _ = dft.scf()
    assert dft.XC_algo == 3 and dft.xc_plan is not None and dft.xc_plan.cache is None
    assert 'on every leaf' in out.getvalue() and 'XC plan:' in out.getvalue()
    assert abs(energy - results[3][0]) < 1e-9
