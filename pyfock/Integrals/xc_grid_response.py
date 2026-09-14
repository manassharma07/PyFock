"""Grid-response terms of the XC nuclear gradient, shared by every functional.

PyFock's XC gradients are evaluated on a *fixed* grid by default: the quadrature grid is treated as if it
did not depend on the nuclear positions. It does, in two ways.

* The grid points of an atom translate rigidly with it, so the density features evaluated *at those
  points* change even with the density field held fixed.
* The Becke partitioning weights ``w_p = vol_p * P_A(r_p)`` depend on every nuclear position.

The two are large and of opposite sign -- they cancel to a large extent -- so neither may be included
without the other. Together they restore exact translational invariance: the net force drops from ~1e-4
Ha/Bohr (semilocal) or ~1e-2 Ha/Bohr (Skala, whose features are integrals over each atomic grid) to
~1e-13.

Both terms depend on the functional only through the per-grid-point cotangents

    c_rho = w dE/drho ,  c_grad = w dE/d(grad rho) ,  c_tau = w dE/dtau ,  eps = dE/dw

so the same code serves the semilocal path (:func:`pyfock.Integrals.eval_xc_grad_2`, where these are
``w*vrho``, ``2 w vsigma grad rho``, ``w*vtau`` and ``rho*e``) and the neural one
(:func:`pyfock.Integrals.eval_xc_grad_skala`, where they come out of a reverse pass through the model).
"""

import numpy as np

__all__ = ['grid_translation_term', 'weight_response_term']

# ao_hess components are stored as 0:xx 1:xy 2:xz 3:yy 4:yz 5:zz.
_HESS = ((0, 1, 2), (1, 3, 4), (2, 4, 5))


def grid_translation_term(c_rho, c_grad, c_tau, ao_grad, ao_hess, Fmj, Hgrad, rho_grad,
                          atom_idx_block, natm):
    """Gradient contribution from the grid points of an atom moving with it, for one block.

    Every point belongs to exactly one atom and translates rigidly with it, so shifting atom ``A`` by
    ``delta`` shifts its own points too and the density features there change by their spatial
    derivatives. The contribution to ``dE/dR_A`` along direction ``b`` is

        sum_{p in A} [ c_rho grad_b rho + sum_a c_grad[a] d_b (grad_a rho) + c_tau grad_b tau ]

    and everything needed is already available from the AO values, gradients and Hessians that the
    Pulay term evaluates.

    Parameters
    ----------
    c_rho : (n,) ndarray
        ``w * dE/drho`` at the block's points.
    c_grad : (3, n) ndarray or None
        ``w * dE/d(grad rho)``; None for an LDA.
    c_tau : (n,) ndarray or None
        ``w * dE/dtau``; None unless the functional is a meta-GGA.
    ao_grad : (3, n, nbf) ndarray
    ao_hess : (6, n, nbf) ndarray or None
        Required whenever ``c_grad`` or ``c_tau`` is given.
    Fmj : (n, nbf) ndarray
        ``(chi D)[g, mu]``.
    Hgrad : list of 3 (n, nbf) ndarray or None
        ``(d_k chi D)[g, mu]``; required whenever ``c_grad`` or ``c_tau`` is given.
    rho_grad : (3, n) ndarray
        Density gradient at the block's points.
    atom_idx_block : (n,) int ndarray
        Which atom each point belongs to.
    natm : int

    Returns
    -------
    (natm, 3) ndarray
    """
    translation = np.zeros((natm, 3))
    for b in range(3):
        term = c_rho * rho_grad[b]
        if c_grad is not None:
            for a in range(3):
                # d(grad_a rho)/d b = 2 [ (d_a d_b chi) . (chi D) + (d_a chi) . (d_b chi D) ]
                hessian_ab = 2.0 * (np.einsum('mj,mj->m', ao_hess[_HESS[a][b]], Fmj)
                                    + np.einsum('mj,mj->m', ao_grad[a], Hgrad[b]))
                term = term + c_grad[a] * hessian_ab
        if c_tau is not None:
            # d(tau)/d b = sum_k (d_b d_k chi) . (d_k chi D)
            grad_tau = sum(np.einsum('mj,mj->m', ao_hess[_HESS[b][k]], Hgrad[k]) for k in range(3))
            term = term + c_tau * grad_tau
        translation[:, b] = np.bincount(atom_idx_block, weights=term, minlength=natm)
    return translation


def weight_response_term(grids, eps):
    """Gradient contribution from the Becke weights depending on the nuclear positions.

    ``sum_p eps_p dw_p/dR_C`` with ``w_p = vol_p * P_A(r_p)``; only the partitioning factor depends on
    the nuclei, so the volume element rides along in the cotangent.

    Parameters
    ----------
    grids : Grids
        Must carry :attr:`~pyfock.Grids.Grids.atomic_weights`, i.e. be a native ('treutler') grid.
    eps : (G,) ndarray
        ``dE/dw_p``. For a semilocal functional this is the XC energy density per unit volume,
        ``rho_p * e_p``.

    Returns
    -------
    (natm, 3) ndarray
    """
    from pyfock.Grids import becke_weight_gradient, size_adjustment_table

    if getattr(grids, 'atomic_weights', None) is None:
        raise ValueError(
            "The grid response needs the unpartitioned single-atom quadrature weights, which the "
            "'numgrid' grid scheme does not expose. Build the grid with the native scheme instead, "
            "e.g. Grids(mol, level=3).")

    atom_coords = np.asarray(grids.mol.coordsBohrs, dtype=np.float64).reshape(-1, 3)
    a_table = size_adjustment_table(np.asarray(grids.mol.Zcharges, dtype=np.int64),
                                    getattr(grids, 'size_adjustment', 'treutler'))
    return becke_weight_gradient(grids.coords, grids.atom_idx, atom_coords, a_table,
                                 np.asarray(eps, dtype=np.float64) * grids.atomic_weights)
