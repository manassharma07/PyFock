"""CUDA kernels of the multipole-accelerated density-fitting Coulomb term (algorithm 12).

The near-field integrals reuse the generated Rys kernels of algorithm 11 with their far-field
mask switched on (see :mod:`~pyfock.Integrals.df_algo11_cuda_core`), and the near-field
``gamma`` contraction reuses ``gamma_items`` unchanged: the item triples ``(pair, auxiliary
shell, column)`` and the column map mean the same thing in both algorithms, only the set of
significant columns is smaller here.  This module adds what the far field needs per SCF
iteration:

``row_coefficients``
    ``w_ij D_ij`` for every function pair of every significant shell pair, once per ``gamma``
    pass, indexed by a global row offset.  Both far-field passes then read the density weight
    instead of repeating the strict-Schwarz test per entry.
``branch_moments`` / ``row_far_field``
    the two halves of the pre-translated branch-centred moments ``Mtil``: accumulate the
    density-weighted rows into the branch moments (``gamma``), and contract the branch local
    expansions back onto the rows (``J``).
``branches_to_atoms`` / ``atoms_to_branches``
    the branch <-> atom multipole coupling.  Unlike the CPU passes these never materialize the
    intermediate ``Y[b, lm, LM]`` / ``W[lm, LM]`` tensors: a warp (or block) contracts the
    translation table with the irregular harmonics of one branch-atom vector on the fly.  That
    trades a factor ``natoms`` in arithmetic for keeping everything in registers and shared
    memory - a good trade, because the real addition table has at most *two* non-zero ``LM`` per
    ``(lm, jk)`` while the dense intermediates are ``(l_aux_max+1)^2 (lmax+l_aux_max+1)^2``
    doubles per branch and would have to be streamed once per atom.  Both read the table in the
    fixed-width, slot-major form the driver builds (``tab_LM``/``tab_coef``, the missing second
    entry padded with a zero coefficient) rather than in CSR form: two unconditional coalesced
    loads per slot beat a dependent offset load plus a data-dependent trip count.
``shell_moments`` / ``gamma_far_field``
    the exact point multipoles of the fitted density per auxiliary shell, and the far-field part
    of ``gamma`` from the local expansions at the atoms.
``j_pairs_ff``
    algorithm 11's ``j_pairs`` plus the far-field row value, so every matrix element is written
    exactly once.

All accumulators are fp64; ``cuda.atomic.add`` on fp64 needs compute capability 6.0 or newer,
which the algorithm-11 driver already requires.  Dynamic shared memory is used so the per-warp
scratch follows ``lmax`` and the auxiliary angular momentum; the driver sizes it at launch, and
also picks the second grid dimension of the coupling and moment kernels, which splits the work
of one atom (or branch) over several blocks when there are too few of them to fill the device.
"""
import math

from numba import cuda, float64

from .df_algo10_helpers import STRICT_PAIR_CUTOFF
from .multipole_helpers_cuda import irregular_harmonics_warp, irregular_harmonics_block

__all__ = ['row_coefficients', 'branch_moments', 'row_far_field', 'branches_to_atoms',
           'atoms_to_branches', 'shell_moments', 'gamma_far_field', 'j_pairs_ff',
           'count_columns_ff']


@cuda.jit(device=True, cache=True)
def _row_to_functions(p, row, pairs, shells):
    """(i, j) of row ``row`` of shell pair ``p``; the row convention of algorithm 11."""
    I, J = pairs[0][p], pairs[1][p]
    nB = shells[1][J]
    if I == J:
        ia = int((math.sqrt(8.0 * row + 1.0) - 1.0) / 2.0)
        ib = row - ia * (ia + 1) // 2
    else:
        ia, ib = row // nB, row % nB
    return shells[0][I] + ia, shells[0][J] + ib


@cuda.jit(cache=True)
def row_coefficients(rows, pairs, shells, sqrt4, strict, dmat, row_off, rowcoef):
    """``rowcoef[row_off[p] + r] = D_ij`` (``i == j``) or ``D_ij + D_ji``, 0 if strict-screened."""
    idx = cuda.grid(1)
    if idx >= rows.shape[0]:
        return
    p, row = rows[idx, 0], rows[idx, 1]
    i, j = _row_to_functions(p, row, pairs, shells)
    value = 0.0
    if not (strict and sqrt4[i, j] * sqrt4[i, j] < STRICT_PAIR_CUTOFF):
        value = dmat[i, j] if i == j else dmat[i, j] + dmat[j, i]
    rowcoef[row_off[p] + row] = value


@cuda.jit(cache=True)
def branch_moments(branch_entry_off, branch_entries, entry_pair, entry_moff, pair_nrows,
                   row_off, Mtil, rowcoef, n_big, branch_mom):
    """
    ``branch_mom[b, LM] = sum over the entries of b of sum_r w_ij D_ij Mtil[entry][r, LM]``.

    One block per (branch, split); a branch's entries are shared out over its blocks, so the
    moment vector accumulates in shared memory and only one partial vector per block is added
    atomically.  Accumulating per entry instead costs one fp64 atomic per (entry, LM), which
    measured several times more than the loads it protects.  Consecutive threads own
    consecutive ``LM``, so the reads of ``Mtil`` are fully coalesced.
    """
    b = cuda.blockIdx.x
    tid = cuda.threadIdx.x
    nthreads = cuda.blockDim.x
    acc = cuda.shared.array(0, float64)
    for LM in range(tid, n_big, nthreads):
        acc[LM] = 0.0
    for idx in range(branch_entry_off[b] + cuda.blockIdx.y, branch_entry_off[b + 1], cuda.gridDim.y):
        e = branch_entries[idx]
        p = entry_pair[e]
        base = entry_moff[e]
        nr = pair_nrows[p]
        ro = row_off[p]
        for LM in range(tid, n_big, nthreads):
            s = 0.0
            for r in range(nr):
                c = rowcoef[ro + r]
                if c != 0.0:
                    s += c * Mtil[base + r * n_big + LM]
            acc[LM] += s
    for LM in range(tid, n_big, nthreads):
        value = acc[LM]
        if value != 0.0:
            cuda.atomic.add(branch_mom, (b, LM), value)


@cuda.jit(cache=True)
def row_far_field(entry_pair, entry_branch, entry_moff, pair_nrows, row_off, Mtil,
                  branch_local, n_big, rowff):
    """
    ``rowff[row_off[p] + r] += sum_LM Mtil[entry][r, LM] branch_local[b, LM]``, one block per
    entry and one warp per row: the lanes stride over ``LM``, so the reads of ``Mtil`` are
    coalesced, and the dot product finishes in a warp shuffle reduction.
    """
    e = cuda.blockIdx.x
    warp = cuda.threadIdx.x // 32
    lane = cuda.threadIdx.x % 32
    nwarps = cuda.blockDim.x // 32
    p = entry_pair[e]
    b = entry_branch[e]
    base = entry_moff[e]
    nr = pair_nrows[p]
    ro = row_off[p]
    local = cuda.shared.array(0, float64)
    for k in range(cuda.threadIdx.x, n_big, cuda.blockDim.x):
        local[k] = branch_local[b, k]
    cuda.syncthreads()
    for r in range(warp, nr, nwarps):
        acc = 0.0
        start = base + r * n_big
        for LM in range(lane, n_big, 32):
            acc += Mtil[start + LM] * local[LM]
        for shift in (16, 8, 4, 2, 1):
            acc += cuda.shfl_down_sync(0xffffffff, acc, shift)
        if lane == 0 and acc != 0.0:
            cuda.atomic.add(rowff, ro + r, acc)


@cuda.jit(cache=True)
def branches_to_atoms(branch_mom, branch_center, atom_coords, atom_shell_off, atom_shells,
                      atom_lmax, aux_l, ff, any_ff, lmax, n_big, n_small, n_irr,
                      npidx, tab_LM, tab_coef, sign_big, L_K):
    """
    ``gamma`` direction: the local expansion ``L_K[K, lm]`` (``l <= l_K``) at each auxiliary
    shell's atom from the moments of all its far-field branches.  One block per (atom, split),
    one warp per branch; ``L_K`` is accumulated atomically because several branches reach the
    same shell.  The warp evaluates its own irregular harmonics cooperatively, then walks the
    ``lm`` of the atom one at a time with the lanes striding over ``jk``.  Giving each lane its
    own ``lm`` instead would look more natural but makes every read of the translation table
    stride by ``n_big``, so each load costs 32 separate transactions; with the lanes on ``jk``
    the table, the branch moments and the signs are all read contiguously and only the small
    reduction at the end is extra work.
    """
    A = cuda.blockIdx.x
    warp = cuda.threadIdx.x // 32
    lane = cuda.threadIdx.x % 32
    nwarps = cuda.blockDim.x // 32
    shared = cuda.shared.array(0, float64)
    ir_base = warp * n_irr
    v_base = nwarps * n_irr + warp * n_small
    m_base = nwarps * (n_irr + n_small) + warp * n_big
    lA = atom_lmax[A]
    nsA = (lA + 1) * (lA + 1)
    ax = atom_coords[A, 0]
    ay = atom_coords[A, 1]
    az = atom_coords[A, 2]
    for b in range(cuda.blockIdx.y * nwarps + warp, branch_mom.shape[0], cuda.gridDim.y * nwarps):
        if not any_ff[b, A]:
            continue
        irregular_harmonics_warp(branch_center[b, 0] - ax, branch_center[b, 1] - ay,
                                 branch_center[b, 2] - az, lmax + lA, shared, ir_base, lane)
        # The signed branch moments are read once per lm otherwise; staging them costs one
        # shared vector per warp and turns nsA passes over global memory into one.
        for jk in range(lane, n_big, 32):
            shared[m_base + jk] = sign_big[jk] * branch_mom[b, jk]
        cuda.syncwarp()
        for lm in range(nsA):
            acc = 0.0
            for jk in range(lane, n_big, 32):
                pidx = lm * n_big + jk
                # l <= l_A and j <= lmax, so every L = l + j of this row is below nirrA and the
                # harmonics the warp just built cover every LM the table can name here.
                s = (tab_coef[pidx] * shared[ir_base + tab_LM[pidx]]
                     + tab_coef[npidx + pidx] * shared[ir_base + tab_LM[npidx + pidx]])
                acc += shared[m_base + jk] * s
            for shift in (16, 8, 4, 2, 1):
                acc += cuda.shfl_down_sync(0xffffffff, acc, shift)
            if lane == 0:
                shared[v_base + lm] = acc
        cuda.syncwarp()
        for s_ in range(atom_shell_off[A], atom_shell_off[A + 1]):
            K = atom_shells[s_]
            if ff[b, K]:
                nK = (aux_l[K] + 1) * (aux_l[K] + 1)
                for lm in range(lane, nK, 32):
                    value = shared[v_base + lm]
                    if value != 0.0:
                        cuda.atomic.add(L_K, (K, lm), value)
        cuda.syncwarp()


@cuda.jit(cache=True)
def atoms_to_branches(shell_mom, branch_center, atom_coords, atom_shell_off, atom_shells,
                      atom_lmax, aux_l, ff, any_ff, lmax, n_big, n_small, n_irr,
                      npidx, tab_LM, tab_coef, sign_big, branch_local):
    """
    ``J`` direction: the local expansion of the far-field fitted density at every branch centre.
    One block per (branch, split); the block accumulates in shared memory and adds one partial
    expansion atomically, so the branch's atoms can be shared out over several blocks when there
    are fewer branches than the device can keep busy.
    """
    b = cuda.blockIdx.x
    tid = cuda.threadIdx.x
    nthreads = cuda.blockDim.x
    shared = cuda.shared.array(0, float64)
    s_base = n_irr
    l_base = n_irr + n_small
    for k in range(tid, n_big, nthreads):
        shared[l_base + k] = 0.0
    bx = branch_center[b, 0]
    by = branch_center[b, 1]
    bz = branch_center[b, 2]
    for A in range(cuda.blockIdx.y, atom_coords.shape[0], cuda.gridDim.y):
        if not any_ff[b, A]:
            continue
        lA = atom_lmax[A]
        nsA = (lA + 1) * (lA + 1)
        for lm in range(tid, nsA, nthreads):
            shared[s_base + lm] = 0.0
        irregular_harmonics_block(bx - atom_coords[A, 0], by - atom_coords[A, 1],
                                  bz - atom_coords[A, 2], lmax + lA, shared, 0, tid, nthreads)
        for s_ in range(atom_shell_off[A], atom_shell_off[A + 1]):
            K = atom_shells[s_]
            if ff[b, K]:
                nK = (aux_l[K] + 1) * (aux_l[K] + 1)
                for lm in range(tid, nK, nthreads):
                    shared[s_base + lm] += shell_mom[K, lm]
        cuda.syncthreads()
        for jk in range(tid, n_big, nthreads):
            acc = 0.0
            for lm in range(nsA):
                pidx = lm * n_big + jk
                # l <= l_A and j <= lmax, so every L = l + j of this row is below nirrA.
                acc += shared[s_base + lm] * (tab_coef[pidx] * shared[tab_LM[pidx]]
                                              + tab_coef[npidx + pidx] * shared[tab_LM[npidx + pidx]])
            shared[l_base + jk] += sign_big[jk] * acc
        cuda.syncthreads()
    for k in range(tid, n_big, nthreads):
        value = shared[l_base + k]
        if value != 0.0:
            cuda.atomic.add(branch_local, (b, k), value)


@cuda.jit(cache=True)
def shell_moments(coeff, aux_mom, aux_off, aux_nbf, aux_l, out):
    """``out[K, lm] = sum_{P in K} c_P M^P_lm``, one block per auxiliary shell."""
    K = cuda.blockIdx.x
    lm = cuda.threadIdx.x
    if lm >= (aux_l[K] + 1) * (aux_l[K] + 1):
        return
    k0 = aux_off[K]
    acc = 0.0
    for c in range(aux_nbf[K]):
        acc += coeff[k0 + c] * aux_mom[k0 + c, lm]
    out[K, lm] = acc


@cuda.jit(cache=True)
def gamma_far_field(L_K, aux_mom, bf_shell, aux_l, gamma):
    """``gamma_P += sum_lm M^P_lm L_K[shell(P), lm]``."""
    P = cuda.grid(1)
    if P >= aux_mom.shape[0]:
        return
    K = bf_shell[P]
    acc = 0.0
    for lm in range((aux_l[K] + 1) * (aux_l[K] + 1)):
        acc += aux_mom[P, lm] * L_K[K, lm]
    gamma[P] += acc


@cuda.jit(cache=True)
def j_pairs_ff(work, pairs, shells, aux_shells, offsets, values, sqrt4, strict,
               column_offsets, column_map, coeff, row_off, rowff, out):
    """Algorithm 11's ``j_pairs`` with the far-field row value added; one warp per row."""
    idx = cuda.grid(1) // 32
    lane = cuda.threadIdx.x % 32
    if idx >= work.shape[0]:
        return
    p, row = work[idx, 0], work[idx, 1]
    i, j = _row_to_functions(p, row, pairs, shells)
    if strict and sqrt4[i, j] * sqrt4[i, j] < STRICT_PAIR_CUTOFF:
        return
    acc = 0.0
    start = offsets[p] + row * pairs[3][p]
    map_start = column_offsets[p]
    for col in range(lane, pairs[3][p], 32):
        acc += values[start + col] * coeff[column_map[map_start + col]]
    for shift in (16, 8, 4, 2, 1):
        acc += cuda.shfl_down_sync(0xffffffff, acc, shift)
    if lane == 0:
        acc += rowff[row_off[p] + row]
        out[i, j] = acc
        out[j, i] = acc


@cuda.jit(cache=True)
def count_columns_ff(items, pairs, aux_shells, qpair, qaux, threshold, pp_off, pp_branch,
                     ff, column_offsets, column_map, counts, errors):
    """
    Independent device replay of the near-field column decision, for debug validation.

    Each item re-derives, from the screening data alone, that its own auxiliary shell really is a
    near-field column of its pair, and that the columns it claims are exactly that shell's
    functions at that place in the host column map the ``J`` pass indexes with.  Together with the
    per-pair column total the caller checks against ``pair_ncols``, that pins the whole layout:
    a duplicated column would have to agree with the map at two different shells (impossible,
    the map holds each shell's own function indices), and a missing one shows up in the total.
    Deriving the offset instead by scanning every earlier shell would make the check quadratic
    in the auxiliary basis, which dominated the runtime of every ``debug=True`` test.
    """
    idx = cuda.grid(1)
    if idx >= items.shape[0]:
        return
    p, K, col = items[idx, 0], items[idx, 1], items[idx, 2]
    if qpair[p] * qaux[K] <= threshold or col + aux_shells[1][K] > pairs[3][p]:
        errors[idx] = 1
        return
    near = False
    for q in range(pp_off[p], pp_off[p + 1]):
        if not ff[pp_branch[q], K]:
            near = True
            break
    if not near:
        errors[idx] = 1
        return
    for c in range(aux_shells[1][K]):
        if column_map[column_offsets[p] + col + c] != aux_shells[0][K] + c:
            errors[idx] = 1
    cuda.atomic.add(counts, p, aux_shells[1][K])
