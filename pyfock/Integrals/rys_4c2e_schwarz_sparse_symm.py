import numpy as np
from numba import njit, prange

from .rys_helpers import coulomb_rys

def rys_4c2e_schwarz_sparse_symm(basis, sqrt_ints4c2e_diag=None, threshold=1e-9):
    """
    Compute 4c2e integrals in sparse format with memory estimates.
    Two-pass approach: first pass identifies significant integrals, second pass computes them.
    """
    
    # Calculate memory estimates
    nbf = basis.bfs_nao
    n_total_integrals = nbf ** 4
    dense_memory_mb = n_total_integrals * 8 / (1024 ** 2)
    
    print(f"Dense storage would require: {dense_memory_mb:.2f} MB ({dense_memory_mb/1024:.2f} GB)")
    
    # Convert basis properties to numpy arrays
    bfs_coords = np.array(basis.bfs_coords)
    bfs_contr_prim_norms = np.array(basis.bfs_contr_prim_norms)
    bfs_lmn = np.array(basis.bfs_lmn)
    bfs_nprim = np.array(basis.bfs_nprim)
    
    maxnprim = max(basis.bfs_nprim)
    bfs_coeffs = np.zeros([basis.bfs_nao, maxnprim])
    bfs_expnts = np.zeros([basis.bfs_nao, maxnprim])
    bfs_prim_norms = np.zeros([basis.bfs_nao, maxnprim])
    for i in range(basis.bfs_nao):
        for j in range(basis.bfs_nprim[i]):
            bfs_coeffs[i, j] = basis.bfs_coeffs[i][j]
            bfs_expnts[i, j] = basis.bfs_expnts[i][j]
            bfs_prim_norms[i, j] = basis.bfs_prim_norms[i][j]
    
    if sqrt_ints4c2e_diag is None:
        sqrt_ints4c2e_diag = np.zeros((1, 1), dtype=np.float64)
        isSchwarz = False
    else:
        isSchwarz = True
    
    ints4c2e_values, ints4c2e_indices = rys_4c2e_schwarz_symm_internal(
        bfs_coords, bfs_contr_prim_norms, bfs_lmn, bfs_nprim, 
        bfs_coeffs, bfs_prim_norms, bfs_expnts,
        sqrt_ints4c2e_diag, isSchwarz, threshold
    )
    
    # Calculate actual memory usage
    n_significant = sum(len(vals) for vals in ints4c2e_values)
    sparse_memory_mb = (n_significant * 8 + n_significant * 3 * 2) / (1024 ** 2)
    # sparse_memory_mb = ints4c2e_values.size / (1024 ** 2) + ints4c2e_indices.size / (1024 ** 2)
    
    print(f"Significant integrals: {n_significant:,}")
    print(f"Sparse storage used: {sparse_memory_mb:.2f} MB ({sparse_memory_mb/1024:.4f} GB)")
    print(f"Compression ratio: {dense_memory_mb/sparse_memory_mb:.2f}x")
    
    return ints4c2e_values, ints4c2e_indices


@njit(parallel=True, cache=True, fastmath=True, error_model="numpy")
def rys_4c2e_schwarz_symm_internal(bfs_coords, bfs_contr_prim_norms, bfs_lmn, 
                                    bfs_nprim, bfs_coeffs, bfs_prim_norms, 
                                    bfs_expnts, sqrt_ints4c2e_diag, 
                                    isSchwarz=False, threshold=1e-9):
    """
    Two-pass sparse computation:
    Pass 1: Schwarz screening to identify significant integrals
    Pass 2: Compute only significant integrals
    """
    
    nbf = len(bfs_coords)
    
    # Pass 1: Count significant integrals per basis function
    counts = np.zeros(nbf, dtype=np.uint32)
    
    for i in range(nbf):
        count = 0
        for j in range(i + 1):
            if i < j:
                triangle2ij = (j) * (j + 1) // 2 + i
            else:
                triangle2ij = (i) * (i + 1) // 2 + j
            
            if isSchwarz:
                sqrt_ij = sqrt_ints4c2e_diag[i, j]
            
            for k in range(nbf):
                for l in range(k + 1):
                    if k < l:
                        triangle2kl = (l) * (l + 1) // 2 + k
                    else:
                        triangle2kl = (k) * (k + 1) // 2 + l
                    
                    if triangle2ij < triangle2kl:
                        continue
                    
                    if isSchwarz:
                        sqrt_kl = sqrt_ints4c2e_diag[k, l]
                        if sqrt_ij * sqrt_kl >= threshold:
                            count += 1
                    else:
                        count += 1
        
        counts[i] = count
    
    # Allocate arrays based on counts
    ints4c2e_values = []
    ints4c2e_indices = []
    for i in range(nbf):
        ints4c2e_values.append(np.zeros(counts[i], dtype=np.float64))
        ints4c2e_indices.append(np.zeros((counts[i], 3), dtype=np.uint16))
    
    # Pass 2: Compute integrals
    pi = np.pi
    pisq = pi * pi
    twopisq = 2.0 * pi * pi
    
    for i in prange(nbf):
        idx_counter = 0
        
        I = bfs_coords[i]
        Ni = bfs_contr_prim_norms[i]
        lmni = bfs_lmn[i]
        la, ma, na = lmni
        nprimi = bfs_nprim[i]
        
        for j in range(i + 1):
            if i < j:
                triangle2ij = (j) * (j + 1) // 2 + i
            else:
                triangle2ij = (i) * (i + 1) // 2 + j
            
            if isSchwarz:
                sqrt_ij = sqrt_ints4c2e_diag[i, j]
                if sqrt_ij < threshold: # Loose check
                    continue
            
            J = bfs_coords[j]
            IJ = I - J
            IJsq = np.sum(IJ ** 2)
            Nj = bfs_contr_prim_norms[j]
            lmnj = bfs_lmn[j]
            lb, mb, nb = lmnj
            tempcoeff1 = Ni * Nj
            nprimj = bfs_nprim[j]
            
            for k in range(nbf):
                K = bfs_coords[k]
                Nk = bfs_contr_prim_norms[k]
                lmnk = bfs_lmn[k]
                lc, mc, nc = lmnk
                tempcoeff2 = tempcoeff1 * Nk
                nprimk = bfs_nprim[k]
                
                for l in range(k + 1):
                    if k < l:
                        triangle2kl = (l) * (l + 1) // 2 + k
                    else:
                        triangle2kl = (k) * (k + 1) // 2 + l
                    
                    if triangle2ij < triangle2kl:
                        continue
                    
                    skip = False
                    if isSchwarz:
                        sqrt_kl = sqrt_ints4c2e_diag[k, l]
                        if sqrt_ij * sqrt_kl < threshold:
                            skip = True
                    
                    if skip:
                        continue
                    
                    L = bfs_coords[l]
                    KL = K - L
                    KLsq = np.sum(KL ** 2)
                    Nl = bfs_contr_prim_norms[l]
                    lmnl = bfs_lmn[l]
                    ld, md, nd = lmnl
                    tempcoeff3 = tempcoeff2 * Nl
                    npriml = bfs_nprim[l]
                    
                    norder = int((la + ma + na + lb + mb + nb + lc + mc + nc + ld + md + nd) / 2 + 1)
                    n = int(max(la + lb, ma + mb, na + nb))
                    m = int(max(lc + ld, mc + md, nc + nd))
                    roots = np.zeros((norder))
                    weights = np.zeros((norder))
                    G = np.zeros((n + 1, m + 1))
                    
                    val = 0.0
                    
                    for ik in range(nprimi):
                        dik = bfs_coeffs[i][ik]
                        Nik = bfs_prim_norms[i][ik]
                        alphaik = bfs_expnts[i][ik]
                        tempcoeff4 = tempcoeff3 * dik * Nik
                        
                        for jk in range(nprimj):
                            alphajk = bfs_expnts[j][jk]
                            gammaP = alphaik + alphajk
                            screenfactorAB = np.exp(-alphaik * alphajk / gammaP * IJsq)
                            if abs(screenfactorAB) < 1.0e-8:
                                continue
                            djk = bfs_coeffs[j][jk]
                            Njk = bfs_prim_norms[j][jk]
                            P = (alphaik * I + alphajk * J) / gammaP
                            PI = P - I
                            PJ = P - J
                            fac1 = twopisq / gammaP * screenfactorAB
                            onefourthgammaPinv = 0.25 / gammaP
                            tempcoeff5 = tempcoeff4 * djk * Njk
                            
                            for kk in range(nprimk):
                                dkk = bfs_coeffs[k][kk]
                                Nkk = bfs_prim_norms[k][kk]
                                alphakk = bfs_expnts[k][kk]
                                tempcoeff6 = tempcoeff5 * dkk * Nkk
                                
                                for lk in range(npriml):
                                    alphalk = bfs_expnts[l][lk]
                                    gammaQ = alphakk + alphalk
                                    screenfactorKL = np.exp(-alphakk * alphalk / gammaQ * KLsq)
                                    if abs(screenfactorKL) < 1.0e-8:
                                        continue
                                    if abs(screenfactorAB * screenfactorKL) < 1.0e-10:
                                        continue
                                    dlk = bfs_coeffs[l][lk]
                                    Nlk = bfs_prim_norms[l][lk]
                                    Q = (alphakk * K + alphalk * L) / gammaQ
                                    PQ = P - Q
                                    
                                    PQsq = np.sum(PQ ** 2)
                                    rho = gammaP * gammaQ / (gammaP + gammaQ)
                                    
                                    tempcoeff7 = tempcoeff6 * dlk * Nlk
                                    
                                    if norder <= 10:
                                        val += tempcoeff7 * coulomb_rys(
                                            roots, weights, G, PQsq, rho, norder, n, m,
                                            la, lb, lc, ld, ma, mb, mc, md, na, nb, nc, nd,
                                            alphaik, alphajk, alphakk, alphalk, I, J, K, L
                                        )
                    
                    ints4c2e_values[i][idx_counter] = val
                    ints4c2e_indices[i][idx_counter, 0] = j
                    ints4c2e_indices[i][idx_counter, 1] = k
                    ints4c2e_indices[i][idx_counter, 2] = l
                    idx_counter += 1
    
    return ints4c2e_values, ints4c2e_indices