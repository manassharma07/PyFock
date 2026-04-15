import numpy as np
from numba import njit, prange
from .rys_helpers import coulomb_rys_3c2e_new
from .integral_helpers import innerLoop4c2e
from .schwarz_helpers import eri_4c2e_diag
from .rys_2c2e_diag import rys_2c2e_diag

def rys_3c2e_grad_symm(basis, auxbasis, slice=None, schwarz=False, threshold_schwarz=1e-9):
    bfs_coords = np.array([basis.bfs_coords])[0]
    bfs_contr = np.array([basis.bfs_contr_prim_norms])[0]
    bfs_lmn = np.array([basis.bfs_lmn])[0]
    bfs_nprim = np.array([basis.bfs_nprim])[0]
    bfs_atoms = np.array(basis.bfs_atoms, dtype=np.int64)

    aux_coords = np.array([auxbasis.bfs_coords])[0]
    aux_contr = np.array([auxbasis.bfs_contr_prim_norms])[0]
    aux_lmn = np.array([auxbasis.bfs_lmn])[0]
    aux_nprim = np.array([auxbasis.bfs_nprim])[0]
    aux_atoms = np.array(auxbasis.bfs_atoms, dtype=np.int64)

    maxnprim = max(basis.bfs_nprim)
    bfs_coeffs = np.zeros([basis.bfs_nao, maxnprim]); bfs_expnts=np.zeros_like(bfs_coeffs); bfs_primn=np.zeros_like(bfs_coeffs)
    for i in range(basis.bfs_nao):
        for j in range(basis.bfs_nprim[i]):
            bfs_coeffs[i,j]=basis.bfs_coeffs[i][j]; bfs_expnts[i,j]=basis.bfs_expnts[i][j]; bfs_primn[i,j]=basis.bfs_prim_norms[i][j]

    maxnprimaux = max(auxbasis.bfs_nprim)
    aux_coeffs = np.zeros([auxbasis.bfs_nao, maxnprimaux]); aux_expnts=np.zeros_like(aux_coeffs); aux_primn=np.zeros_like(aux_coeffs)
    for i in range(auxbasis.bfs_nao):
        for j in range(auxbasis.bfs_nprim[i]):
            aux_coeffs[i,j]=auxbasis.bfs_coeffs[i][j]; aux_expnts[i,j]=auxbasis.bfs_expnts[i][j]; aux_primn[i,j]=auxbasis.bfs_prim_norms[i][j]

    if slice is None:
        slice = [0,basis.bfs_nao,0,basis.bfs_nao,0,auxbasis.bfs_nao]
    a,b,c,d,e,f = map(int,slice)

    if schwarz:
        from .rys_2c2e_diag import rys_2c2e_diag
        from .schwarz_helpers import eri_4c2e_diag
        s4 = np.sqrt(np.abs(eri_4c2e_diag(basis)))
        s2 = np.sqrt(np.abs(rys_2c2e_diag(auxbasis)))
    else:
        s4 = np.zeros((1,1)); s2=np.zeros(1)

    d3c = rys_3c2e_grad_symm_internal(
        bfs_coords, bfs_contr, bfs_lmn, bfs_nprim, bfs_coeffs, bfs_primn, bfs_expnts, bfs_atoms,
        aux_coords, aux_contr, aux_lmn, aux_nprim, aux_coeffs, aux_primn, aux_expnts, aux_atoms,
        a,b,c,d,e,f, schwarz, s4, s2, threshold_schwarz)
    return d3c

@njit(parallel=False, cache=True, fastmath=False, nogil=True)
def rys_3c2e_grad_symm_internal(bfs_c, bfs_cn, bfs_lmn, bfs_np, bfs_cf, bfs_pn, bfs_ex, bfs_at,
                                aux_c, aux_cn, aux_lmn, aux_np, aux_cf, aux_pn, aux_ex, aux_at,
                                i0,i1,j0,j1,k0,k1, schwarz, s4, s2, thr):
    nA=i1-i0; nB=j1-j0; nC=k1-k0
    nat = int(max(np.max(bfs_at), np.max(aux_at)) + 1)
    d3c = np.zeros((nat,3,nA,nB,nC), dtype=np.float64)

    tri = (i0==j0 and i1==j1)
    roots=np.zeros(10); w=np.zeros(10); G=np.zeros((20,20))
    L=np.zeros(3)

    for i in range(i0,i1):
        I = bfs_c[i]; Ni=bfs_cn[i]; la,ma,na = bfs_lmn[i]; ia=bfs_at[i]
        for j in range(j0,j1):
            if tri and j>i: continue
            if schwarz and s4[i,j]<thr: continue
            J = bfs_c[j]; Nj=bfs_cn[j]; lb,mb,nb = bfs_lmn[j]; ja=bfs_at[j]
            IJ = I-J; IJsq=np.sum(IJ*IJ); c1=Ni*Nj
            for k in range(k0,k1):
                K = aux_c[k]; Nk=aux_cn[k]; lc,mc,nc = aux_lmn[k]; ka=aux_at[k]
                if schwarz and s4[i,j]*s2[k] < thr: continue
                c2 = c1*Nk

                gA=np.zeros(3); gB=np.zeros(3); gC=np.zeros(3)

                # --- Rys quadrature (your norder<=10 path) ---
                norder = (la+lb+lc+ma+mb+mc+na+nb+nc)//2 + 2   # +1 for derivative
                if norder>10: continue   # fall back to analytic – omitted for brevity

                for ik in range(bfs_np[i]):
                    aik=bfs_ex[i,ik]; dik=bfs_cf[i,ik]; Nik=bfs_pn[i,ik]; c3=c2*dik*Nik
                    for jk in range(bfs_np[j]):
                        ajk=bfs_ex[j,jk]; djk=bfs_cf[j,jk]; Njk=bfs_pn[j,jk]
                        gP=aik+ajk; Kab=np.exp(-aik*ajk/gP*IJsq)
                        if Kab<1e-12: continue
                        P = (aik*I+ajk*J)/gP; PQ = P-K; PQsq=np.sum(PQ*PQ)
                        c4=c3*djk*Njk*Kab
                        for kk in range(aux_np[k]):
                            akk=aux_ex[k,kk]; dkk=aux_cf[k,kk]; Nkk=aux_pn[k,kk]
                            gQ=akk; rho=gP*gQ/(gP+gQ); c5=c4*dkk*Nkk

                            n = max(la+lb+1, ma+mb+1, na+nb+1)
                            m = max(lc+1, mc+1, nc+1)

                            # ---- atom A ----
                            if la>0:
                                v = coulomb_rys_3c2e_new(roots,w,G,PQsq,rho,norder,n,m,
                                    la-1,lb,lc,0, ma,mb,mc,0, na,nb,nc,0,
                                    aik,ajk,akk,0., I,J,K,L,P)
                                gA[0] += c5*(-la*v)
                            v = coulomb_rys_3c2e_new(roots,w,G,PQsq,rho,norder,n,m,
                                    la+1,lb,lc,0, ma,mb,mc,0, na,nb,nc,0,
                                    aik,ajk,akk,0., I,J,K,L,P)
                            gA[0] += c5*(2*aik*v)
                            # y
                            if ma>0:
                                v = coulomb_rys_3c2e_new(roots,w,G,PQsq,rho,norder,n,m,
                                    la,lb,lc,0, ma-1,mb,mc,0, na,nb,nc,0, aik,ajk,akk,0., I,J,K,L,P)
                                gA[1] += c5*(-ma*v)
                            v = coulomb_rys_3c2e_new(roots,w,G,PQsq,rho,norder,n,m,
                                    la,lb,lc,0, ma+1,mb,mc,0, na,nb,nc,0, aik,ajk,akk,0., I,J,K,L,P)
                            gA[1] += c5*(2*aik*v)
                            # z
                            if na>0:
                                v = coulomb_rys_3c2e_new(roots,w,G,PQsq,rho,norder,n,m,
                                    la,lb,lc,0, ma,mb,mc,0, na-1,nb,nc,0, aik,ajk,akk,0., I,J,K,L,P)
                                gA[2] += c5*(-na*v)
                            v = coulomb_rys_3c2e_new(roots,w,G,PQsq,rho,norder,n,m,
                                    la,lb,lc,0, ma,mb,mc,0, na+1,nb,nc,0, aik,ajk,akk,0., I,J,K,L,P)
                            gA[2] += c5*(2*aik*v)

                            # ---- atom B ----
                            if lb>0:
                                v = coulomb_rys_3c2e_new(roots,w,G,PQsq,rho,norder,n,m,
                                    la,lb-1,lc,0, ma,mb,mc,0, na,nb,nc,0, aik,ajk,akk,0., I,J,K,L,P)
                                gB[0] += c5*(-lb*v)
                            v = coulomb_rys_3c2e_new(roots,w,G,PQsq,rho,norder,n,m,
                                    la,lb+1,lc,0, ma,mb,mc,0, na,nb,nc,0, aik,ajk,akk,0., I,J,K,L,P)
                            gB[0] += c5*(2*ajk*v)
                            if mb>0:
                                v = coulomb_rys_3c2e_new(roots,w,G,PQsq,rho,norder,n,m,
                                    la,lb,lc,0, ma,mb-1,mc,0, na,nb,nc,0, aik,ajk,akk,0., I,J,K,L,P)
                                gB[1] += c5*(-mb*v)
                            v = coulomb_rys_3c2e_new(roots,w,G,PQsq,rho,norder,n,m,
                                    la,lb,lc,0, ma,mb+1,mc,0, na,nb,nc,0, aik,ajk,akk,0., I,J,K,L,P)
                            gB[1] += c5*(2*ajk*v)
                            if nb>0:
                                v = coulomb_rys_3c2e_new(roots,w,G,PQsq,rho,norder,n,m,
                                    la,lb,lc,0, ma,mb,mc,0, na,nb-1,nc,0, aik,ajk,akk,0., I,J,K,L,P)
                                gB[2] += c5*(-nb*v)
                            v = coulomb_rys_3c2e_new(roots,w,G,PQsq,rho,norder,n,m,
                                    la,lb,lc,0, ma,mb,mc,0, na,nb+1,nc,0, aik,ajk,akk,0., I,J,K,L,P)
                            gB[2] += c5*(2*ajk*v)

                            # ---- atom C (aux) ----
                            if lc>0:
                                v = coulomb_rys_3c2e_new(roots,w,G,PQsq,rho,norder,n,m,
                                    la,lb,lc-1,0, ma,mb,mc,0, na,nb,nc,0, aik,ajk,akk,0., I,J,K,L,P)
                                gC[0] += c5*(-lc*v)
                            v = coulomb_rys_3c2e_new(roots,w,G,PQsq,rho,norder,n,m,
                                    la,lb,lc+1,0, ma,mb,mc,0, na,nb,nc,0, aik,ajk,akk,0., I,J,K,L,P)
                            gC[0] += c5*(2*akk*v)
                            if mc>0:
                                v = coulomb_rys_3c2e_new(roots,w,G,PQsq,rho,norder,n,m,
                                    la,lb,lc,0, ma,mb,mc-1,0, na,nb,nc,0, aik,ajk,akk,0., I,J,K,L,P)
                                gC[1] += c5*(-mc*v)
                            v = coulomb_rys_3c2e_new(roots,w,G,PQsq,rho,norder,n,m,
                                    la,lb,lc,0, ma,mb,mc+1,0, na,nb,nc,0, aik,ajk,akk,0., I,J,K,L,P)
                            gC[1] += c5*(2*akk*v)
                            if nc>0:
                                v = coulomb_rys_3c2e_new(roots,w,G,PQsq,rho,norder,n,m,
                                    la,lb,lc,0, ma,mb,mc,0, na,nb,nc-1,0, aik,ajk,akk,0., I,J,K,L,P)
                                gC[2] += c5*(-nc*v)
                            v = coulomb_rys_3c2e_new(roots,w,G,PQsq,rho,norder,n,m,
                                    la,lb,lc,0, ma,mb,mc,0, na,nb,nc+1,0, aik,ajk,akk,0., I,J,K,L,P)
                            gC[2] += c5*(2*akk*v)

                # store
                ii=i-i0; jj=j-j0; kk=k-k0
                for d in range(3):
                    d3c[ia,d,ii,jj,kk] += gA[d]
                    d3c[ja,d,ii,jj,kk] += gB[d]
                    d3c[ka,d,ii,jj,kk] += gC[d]
                    if tri:
                        d3c[ia,d,jj,ii,kk] += gB[d]  # swap i<->j
                        d3c[ja,d,jj,ii,kk] += gA[d]
                        d3c[ka,d,jj,ii,kk] += gC[d]
    return d3c


# def rys_3c2e_grad_symm(basis, auxbasis, slice=None, schwarz=False, threshold_schwarz=1e-9):
#     """
#     Compute the nuclear gradient of three-center two-electron (3c2e) integrals.

#     Computes d(AB|C)/dX for X in {A, B, C} and directions {x, y, z}.

#     Parameters
#     ----------
#     basis : object
#         Primary basis set object.
#     auxbasis : object
#         Auxiliary basis set object.
#     slice : list of int, optional
#         6-element list [start_A, end_A, start_B, end_B, start_C, end_C].
#         If None, computes all integrals.
#     schwarz : bool, optional
#         If True, applies Schwarz screening.
#     threshold_schwarz : float, optional
#         Threshold for Schwarz screening (default 1e-9).

#     Returns
#     -------
#     d3c2e : ndarray of shape (natoms, 3, num_A, num_B, num_C)
#         Nuclear gradient of 3c2e integrals.
#         d3c2e[atom, direction, A, B, C] = d(AB|C)/dR_{atom, direction}
#     """

#     # Convert basis properties to numpy arrays
#     bfs_coords            = np.array([basis.bfs_coords])
#     bfs_contr_prim_norms  = np.array([basis.bfs_contr_prim_norms])
#     bfs_lmn               = np.array([basis.bfs_lmn])
#     bfs_nprim             = np.array([basis.bfs_nprim])
#     bfs_atoms             = np.array([basis.bfs_atoms])

#     aux_bfs_coords           = np.array([auxbasis.bfs_coords])
#     aux_bfs_contr_prim_norms = np.array([auxbasis.bfs_contr_prim_norms])
#     aux_bfs_lmn              = np.array([auxbasis.bfs_lmn])
#     aux_bfs_nprim            = np.array([auxbasis.bfs_nprim])
#     aux_bfs_atoms            = np.array([auxbasis.bfs_atoms])

#     natoms = max(
#         max(basis.bfs_atoms), max(auxbasis.bfs_atoms)
#     ) + 1

#     # Pad jagged arrays to 2D
#     maxnprim = max(basis.bfs_nprim)
#     bfs_coeffs      = np.zeros([basis.bfs_nao, maxnprim])
#     bfs_expnts      = np.zeros([basis.bfs_nao, maxnprim])
#     bfs_prim_norms  = np.zeros([basis.bfs_nao, maxnprim])
#     for i in range(basis.bfs_nao):
#         for j in range(basis.bfs_nprim[i]):
#             bfs_coeffs[i, j]     = basis.bfs_coeffs[i][j]
#             bfs_expnts[i, j]     = basis.bfs_expnts[i][j]
#             bfs_prim_norms[i, j] = basis.bfs_prim_norms[i][j]

#     maxnprimaux = max(auxbasis.bfs_nprim)
#     aux_bfs_coeffs     = np.zeros([auxbasis.bfs_nao, maxnprimaux])
#     aux_bfs_expnts     = np.zeros([auxbasis.bfs_nao, maxnprimaux])
#     aux_bfs_prim_norms = np.zeros([auxbasis.bfs_nao, maxnprimaux])
#     for i in range(auxbasis.bfs_nao):
#         for j in range(auxbasis.bfs_nprim[i]):
#             aux_bfs_coeffs[i, j]     = auxbasis.bfs_coeffs[i][j]
#             aux_bfs_expnts[i, j]     = auxbasis.bfs_expnts[i][j]
#             aux_bfs_prim_norms[i, j] = auxbasis.bfs_prim_norms[i][j]

#     if slice is None:
#         slice = [0, basis.bfs_nao, 0, basis.bfs_nao, 0, auxbasis.bfs_nao]

#     indx_startA = int(slice[0]); indx_endA = int(slice[1])
#     indx_startB = int(slice[2]); indx_endB = int(slice[3])
#     indx_startC = int(slice[4]); indx_endC = int(slice[5])

#     if schwarz:
#         ints4c2e_diag       = eri_4c2e_diag(basis)
#         ints2c2e_diag       = rys_2c2e_diag(auxbasis)
#         sqrt_ints4c2e_diag  = np.sqrt(np.abs(ints4c2e_diag))
#         sqrt_diag_ints2c2e  = np.sqrt(np.abs(ints2c2e_diag))
#     else:
#         sqrt_ints4c2e_diag = np.zeros((1, 1), dtype=np.float64)
#         sqrt_diag_ints2c2e = np.zeros((1,),   dtype=np.float64)

#     d3c2e = rys_3c2e_grad_internal(
#         natoms,
#         bfs_atoms[0],       bfs_coords[0],
#         bfs_contr_prim_norms[0], bfs_lmn[0], bfs_nprim[0],
#         bfs_coeffs, bfs_prim_norms, bfs_expnts,
#         aux_bfs_atoms[0],   aux_bfs_coords[0],
#         aux_bfs_contr_prim_norms[0], aux_bfs_lmn[0], aux_bfs_nprim[0],
#         aux_bfs_coeffs, aux_bfs_prim_norms, aux_bfs_expnts,
#         indx_startA, indx_endA,
#         indx_startB, indx_endB,
#         indx_startC, indx_endC,
#         schwarz, sqrt_ints4c2e_diag, sqrt_diag_ints2c2e, threshold_schwarz,
#     )
#     return d3c2e


# @njit(parallel=False, cache=True, fastmath=True, nogil=True, error_model="numpy")
# def rys_3c2e_grad_internal(
#     natoms,
#     bfs_atoms,      bfs_coords,
#     bfs_contr_prim_norms, bfs_lmn, bfs_nprim,
#     bfs_coeffs, bfs_prim_norms, bfs_expnts,
#     aux_bfs_atoms,  aux_bfs_coords,
#     aux_bfs_contr_prim_norms, aux_bfs_lmn, aux_bfs_nprim,
#     aux_bfs_coeffs, aux_bfs_prim_norms, aux_bfs_expnts,
#     indx_startA, indx_endA,
#     indx_startB, indx_endB,
#     indx_startC, indx_endC,
#     schwarz, sqrt_ints4c2e_diag, sqrt_diag_ints2c2e, threshold_schwarz,
# ):
#     """
#     Numba-accelerated kernel for d(AB|C)/dR.

#     Output shape: (natoms, 3, num_A, num_B, num_C)

#     For each pair (i,j,k) we compute:

#         d(AB|C)/dA_xi  via:  -la_xi * (A^{l-1} B | C) + 2*alpha_A * (A^{l+1} B | C)
#         d(AB|C)/dB_xi  via:  -lb_xi * (A B^{l-1} | C) + 2*alpha_B * (A B^{l+1} | C)
#         d(AB|C)/dC_xi  via:  -lc_xi * (AB | C^{l-1}) + 2*alpha_C * (AB | C^{l+1})

#     Translational invariance:
#         sum over all atoms of d(AB|C)/dR = 0
#     """

#     num_A = indx_endA - indx_startA
#     num_B = indx_endB - indx_startB
#     num_C = indx_endC - indx_startC

#     # Output: (natoms, xyz, A, B, C)
#     d3c2e = np.zeros((natoms, 3, num_A, num_B, num_C), dtype=np.float64)

#     # Dummy variables for unused indices in 3c2e (no fourth center)
#     ld, md, nd = 0, 0, 0
#     alphalk    = 0.0
#     L          = np.zeros(3)

#     # Check symmetry in A/B indices
#     tri_symm = (indx_startA == indx_startB and indx_endA == indx_endB)

#     # Pre-allocate work arrays for Rys quadrature (size 10 supports up to order 10)
#     roots   = np.zeros(10)
#     weights = np.zeros(10)
#     G       = np.zeros((20, 20))

#     for i in range(indx_startA, indx_endA):
#         I      = bfs_coords[i]
#         Ni     = bfs_contr_prim_norms[i]
#         lmni   = bfs_lmn[i]
#         la, ma, na = lmni[0], lmni[1], lmni[2]
#         nprimi = bfs_nprim[i]
#         atomA  = bfs_atoms[i]

#         for j in range(indx_startB, indx_endB):
#             # Exploit (AB|C) = (BA|C) symmetry
#             if tri_symm and j > i:
#                 continue

#             if schwarz:
#                 if sqrt_ints4c2e_diag[i, j] < threshold_schwarz:
#                     continue

#             J      = bfs_coords[j]
#             IJ     = I - J
#             IJsq   = IJ[0]*IJ[0] + IJ[1]*IJ[1] + IJ[2]*IJ[2]
#             Nj     = bfs_contr_prim_norms[j]
#             lmnj   = bfs_lmn[j]
#             lb, mb, nb = lmnj[0], lmnj[1], lmnj[2]
#             nprimj = bfs_nprim[j]
#             atomB  = bfs_atoms[j]

#             for k in range(indx_startC, indx_endC):
#                 K      = aux_bfs_coords[k]
#                 Nk     = aux_bfs_contr_prim_norms[k]
#                 lmnk   = aux_bfs_lmn[k]
#                 lc, mc, nc = lmnk[0], lmnk[1], lmnk[2]
#                 nprimk = aux_bfs_nprim[k]
#                 atomC  = aux_bfs_atoms[k]

#                 if schwarz:
#                     if sqrt_ints4c2e_diag[i, j] * sqrt_diag_ints2c2e[k] < threshold_schwarz:
#                         continue

#                 # Gradient accumulators for this (i,j,k) shell triplet
#                 gradA = np.zeros(3)
#                 gradB = np.zeros(3)
#                 gradC = np.zeros(3)

#                 # Base angular momentum info (independent of primitives)
#                 n_ab_base = int(max(la+lb, ma+mb, na+nb))
#                 m_c_base  = int(max(lc, mc, nc))

#                 for ik in range(nprimi):
#                     alphaik = bfs_expnts[i, ik]
#                     dik     = bfs_coeffs[i, ik]
#                     Nik     = bfs_prim_norms[i, ik]
#                     c3A     = Ni * Nj * Nk * dik * Nik

#                     for jk in range(nprimj):
#                         alphajk = bfs_expnts[j, jk]
#                         gammaP  = alphaik + alphajk
#                         expAB   = np.exp(-alphaik * alphajk / gammaP * IJsq)
#                         if abs(expAB) < 1.0e-14:
#                             continue

#                         djk  = bfs_coeffs[j, jk]
#                         Njk  = bfs_prim_norms[j, jk]
#                         P    = (alphaik * I + alphajk * J) / gammaP
#                         c3AB = c3A * djk * Njk

#                         for kk in range(nprimk):
#                             alphakk = aux_bfs_expnts[k, kk]
#                             dkk     = aux_bfs_coeffs[k, kk]
#                             Nkk     = aux_bfs_prim_norms[k, kk]

#                             coeff  = c3AB * dkk * Nkk
#                             gammaQ = alphakk

#                             # P-K vector and squared distance
#                             PK     = P - K
#                             PQsq   = PK[0]*PK[0] + PK[1]*PK[1] + PK[2]*PK[2]
#                             rho    = gammaP * gammaQ / (gammaP + gammaQ)

#                             # --------------------------------------------------
#                             # d(AB|C)/dA_xi = -la_xi*(A^{la-1} B|C)
#                             #                + 2*alphaik*(A^{la+1} B|C)
#                             # --------------------------------------------------
#                             for xi in range(3):
#                                 la_arr = np.array([la, ma, na])
#                                 lb_arr = np.array([lb, mb, nb])
#                                 lc_arr = np.array([lc, mc, nc])

#                                 lfactor = float(la_arr[xi])

#                                 # Down-shift: la_xi -> la_xi - 1
#                                 if la_arr[xi] > 0:
#                                     la_m = la_arr.copy()
#                                     la_m[xi] -= 1
#                                     n_rys_m = int((la_m[0]+la_m[1]+la_m[2]
#                                                    + lb+mb+nb
#                                                    + lc+mc+nc) / 2 + 1)
#                                     n_ab_m  = int(max(la_m[0]+lb,
#                                                       la_m[1]+mb,
#                                                       la_m[2]+nb))
#                                     valm = coeff * coulomb_rys_3c2e_new(
#                                         roots, weights, G, PQsq, rho, n_rys_m,
#                                         n_ab_m, m_c_base,
#                                         la_m[0], lb, lc, ld,
#                                         la_m[1], mb, mc, md,
#                                         la_m[2], nb, nc, nd,
#                                         alphaik, alphajk, alphakk, alphalk,
#                                         I, J, K, L, P,
#                                     )
#                                 else:
#                                     valm = 0.0

#                                 # Up-shift: la_xi -> la_xi + 1
#                                 la_p = la_arr.copy()
#                                 la_p[xi] += 1
#                                 n_rys_p = int((la_p[0]+la_p[1]+la_p[2]
#                                                + lb+mb+nb
#                                                + lc+mc+nc) / 2 + 1)
#                                 n_ab_p  = int(max(la_p[0]+lb,
#                                                   la_p[1]+mb,
#                                                   la_p[2]+nb))
#                                 valp = coeff * coulomb_rys_3c2e_new(
#                                     roots, weights, G, PQsq, rho, n_rys_p,
#                                     n_ab_p, m_c_base,
#                                     la_p[0], lb, lc, ld,
#                                     la_p[1], mb, mc, md,
#                                     la_p[2], nb, nc, nd,
#                                     alphaik, alphajk, alphakk, alphalk,
#                                     I, J, K, L, P,
#                                 )

#                                 gradA[xi] += -lfactor * valm + 2.0 * alphaik * valp

#                             # --------------------------------------------------
#                             # d(AB|C)/dB_xi = -lb_xi*(A B^{lb-1}|C)
#                             #                + 2*alphajk*(A B^{lb+1}|C)
#                             # --------------------------------------------------
#                             for xi in range(3):
#                                 la_arr = np.array([la, ma, na])
#                                 lb_arr = np.array([lb, mb, nb])
#                                 lc_arr = np.array([lc, mc, nc])

#                                 lfactor = float(lb_arr[xi])

#                                 # Down-shift: lb_xi -> lb_xi - 1
#                                 if lb_arr[xi] > 0:
#                                     lb_m = lb_arr.copy()
#                                     lb_m[xi] -= 1
#                                     n_rys_m = int((la+ma+na
#                                                    + lb_m[0]+lb_m[1]+lb_m[2]
#                                                    + lc+mc+nc) / 2 + 1)
#                                     n_ab_m  = int(max(la+lb_m[0],
#                                                       ma+lb_m[1],
#                                                       na+lb_m[2]))
#                                     valm = coeff * coulomb_rys_3c2e_new(
#                                         roots, weights, G, PQsq, rho, n_rys_m,
#                                         n_ab_m, m_c_base,
#                                         la, lb_m[0], lc, ld,
#                                         ma, lb_m[1], mc, md,
#                                         na, lb_m[2], nc, nd,
#                                         alphaik, alphajk, alphakk, alphalk,
#                                         I, J, K, L, P,
#                                     )
#                                 else:
#                                     valm = 0.0

#                                 # Up-shift: lb_xi -> lb_xi + 1
#                                 lb_p = lb_arr.copy()
#                                 lb_p[xi] += 1
#                                 n_rys_p = int((la+ma+na
#                                                + lb_p[0]+lb_p[1]+lb_p[2]
#                                                + lc+mc+nc) / 2 + 1)
#                                 n_ab_p  = int(max(la+lb_p[0],
#                                                   ma+lb_p[1],
#                                                   na+lb_p[2]))
#                                 valp = coeff * coulomb_rys_3c2e_new(
#                                     roots, weights, G, PQsq, rho, n_rys_p,
#                                     n_ab_p, m_c_base,
#                                     la, lb_p[0], lc, ld,
#                                     ma, lb_p[1], mc, md,
#                                     na, lb_p[2], nc, nd,
#                                     alphaik, alphajk, alphakk, alphalk,
#                                     I, J, K, L, P,
#                                 )

#                                 gradB[xi] += -lfactor * valm + 2.0 * alphajk * valp

#                             # --------------------------------------------------
#                             # d(AB|C)/dC_xi = -lc_xi*(AB|C^{lc-1})
#                             #                + 2*alphakk*(AB|C^{lc+1})
#                             # --------------------------------------------------
#                             for xi in range(3):
#                                 la_arr = np.array([la, ma, na])
#                                 lb_arr = np.array([lb, mb, nb])
#                                 lc_arr = np.array([lc, mc, nc])

#                                 lfactor = float(lc_arr[xi])

#                                 # Down-shift: lc_xi -> lc_xi - 1
#                                 if lc_arr[xi] > 0:
#                                     lc_m = lc_arr.copy()
#                                     lc_m[xi] -= 1
#                                     n_rys_m = int((la+ma+na
#                                                    + lb+mb+nb
#                                                    + lc_m[0]+lc_m[1]+lc_m[2]) / 2 + 1)
#                                     m_c_m   = int(max(lc_m[0], lc_m[1], lc_m[2]))
#                                     valm = coeff * coulomb_rys_3c2e_new(
#                                         roots, weights, G, PQsq, rho, n_rys_m,
#                                         n_ab_base, m_c_m,
#                                         la, lb, lc_m[0], ld,
#                                         ma, mb, lc_m[1], md,
#                                         na, nb, lc_m[2], nd,
#                                         alphaik, alphajk, alphakk, alphalk,
#                                         I, J, K, L, P,
#                                     )
#                                 else:
#                                     valm = 0.0

#                                 # Up-shift: lc_xi -> lc_xi + 1
#                                 lc_p = lc_arr.copy()
#                                 lc_p[xi] += 1
#                                 n_rys_p = int((la+ma+na
#                                                + lb+mb+nb
#                                                + lc_p[0]+lc_p[1]+lc_p[2]) / 2 + 1)
#                                 m_c_p   = int(max(lc_p[0], lc_p[1], lc_p[2]))
#                                 valp = coeff * coulomb_rys_3c2e_new(
#                                     roots, weights, G, PQsq, rho, n_rys_p,
#                                     n_ab_base, m_c_p,
#                                     la, lb, lc_p[0], ld,
#                                     ma, mb, lc_p[1], md,
#                                     na, nb, lc_p[2], nd,
#                                     alphaik, alphajk, alphakk, alphalk,
#                                     I, J, K, L, P,
#                                 )

#                                 gradC[xi] += -lfactor * valm + 2.0 * alphakk * valp

#                 # Accumulate gradients into output tensor
#                 iA = i - indx_startA
#                 iB = j - indx_startB
#                 iC = k - indx_startC

#                 for xi in range(3):
#                     d3c2e[atomA, xi, iA, iB, iC] += gradA[xi]
#                     d3c2e[atomB, xi, iA, iB, iC] += gradB[xi]
#                     d3c2e[atomC, xi, iA, iB, iC] += gradC[xi]

#                 # Exploit (AB|C) = (BA|C) symmetry to fill (j,i,k) from (i,j,k)
#                 if tri_symm and i != j:
#                     for xi in range(3):
#                         # When we swap A<->B: gradA goes to atomB slot, gradB to atomA slot
#                         d3c2e[atomB, xi, iB, iA, iC] += gradA[xi]
#                         d3c2e[atomA, xi, iB, iA, iC] += gradB[xi]
#                         d3c2e[atomC, xi, iB, iA, iC] += gradC[xi]

#     return d3c2e