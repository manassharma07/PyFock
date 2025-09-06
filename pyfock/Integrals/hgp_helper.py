import numpy as np
from numba import njit 
from .integral_helpers import comb

'''
Most of the functions here are adapted from the following repository (BSD license) using Python language
https://github.com/rpmuller/pyquante2/

More specifically from this file:
https://github.com/rpmuller/pyquante2/blob/master/pyquante2/ints/hgp.py
'''
@njit(cache=True,fastmath=True, error_model='numpy', nogil=True)
def gaussian_product_center(alphaa,xyza,alphab,xyzb):
    return (alphaa*xyza + alphab*xyzb)/(alphaa+alphab)

@njit(cache=True,fastmath=True, error_model='numpy', nogil=True)
def hgp_hrr(xyza,norma,lmna,alphaa,
        xyzb,normb,lmnb,alphab,
        xyzc,normc,lmnc,alphac,
        xyzd,normd,lmnd,alphad):

    la,ma,na = lmna
    lb,mb,nb = lmnb
    lc,mc,nc = lmnc
    ld,md,nd = lmnd
    xa,ya,za = xyza
    xb,yb,zb = xyzb
    xc,yc,zc = xyzc
    xd,yd,zd = xyzd
    if lb > 0:
        return (hgp_hrr(xyza,norma,(la+1,ma,na),alphaa,
                    xyzb,normb,(lb-1,mb,nb),alphab,
                    xyzc,normc,(lc,mc,nc),alphac,
                    xyzd,normd,(ld,md,nd),alphad)
                + (xa-xb)*hgp_hrr(xyza,norma,(la,ma,na),alphaa,
                              xyzb,normb,(lb-1,mb,nb),alphab,
                              xyzc,normc,(lc,mc,nc),alphac,
                              xyzd,normd,(ld,md,nd),alphad)
                )
    elif mb > 0:
        return (hgp_hrr(xyza,norma,(la,ma+1,na),alphaa,
                    xyzb,normb,(lb,mb-1,nb),alphab,
                    xyzc,normc,(lc,mc,nc),alphac,
                    xyzd,normd,(ld,md,nd),alphad)
                + (ya-yb)*hgp_hrr(xyza,norma,(la,ma,na),alphaa,
                              xyzb,normb,(lb,mb-1,nb),alphab,
                              xyzc,normc,(lc,mc,nc),alphac,
                              xyzd,normd,(ld,md,nd),alphad)
                )
    elif nb > 0:
        return (hgp_hrr(xyza,norma,(la,ma,na+1),alphaa,
                    xyzb,normb,(lb,mb,nb-1),alphab,
                    xyzc,normc,(lc,mc,nc),alphac,
                    xyzd,normd,(ld,md,nd),alphad)
                + (za-zb)*hgp_hrr(xyza,norma,(la,ma,na),alphaa,
                              xyzb,normb,(lb,mb,nb-1),alphab,
                              xyzc,normc,(lc,mc,nc),alphac,
                              xyzd,normd,(ld,md,nd),alphad)
                )
    elif ld > 0:
        return (hgp_hrr(xyza,norma,(la,ma,na),alphaa,
                    xyzb,normb,(lb,mb,nb),alphab,
                    xyzc,normc,(lc+1,mc,nc),alphac,
                    xyzd,normd,(ld-1,md,nd),alphad)
                + (xc-xd)*hgp_hrr(xyza,norma,(la,ma,na),alphaa,
                              xyzb,normb,(lb,mb,nb),alphab,
                              xyzc,normc,(lc,mc,nc),alphac,
                              xyzd,normd,(ld-1,md,nd),alphad)
                )
    elif md > 0:
        return (hgp_hrr(xyza,norma,(la,ma,na),alphaa,
                    xyzb,normb,(lb,mb,nb),alphab,
                    xyzc,normc,(lc,mc+1,nc),alphac,
                    xyzd,normd,(ld,md-1,nd),alphad)
                + (yc-yd)*hgp_hrr(xyza,norma,(la,ma,na),alphaa,
                              xyzb,normb,(lb,mb,nb),alphab,
                              xyzc,normc,(lc,mc,nc),alphac,
                              xyzd,normd,(ld,md-1,nd),alphad)
                )
    elif nd > 0:
        return (hgp_hrr(xyza,norma,(la,ma,na),alphaa,
                    xyzb,normb,(lb,mb,nb),alphab,
                    xyzc,normc,(lc,mc,nc+1),alphac,
                    xyzd,normd,(ld,md,nd-1),alphad)
                + (zc-zd)*hgp_hrr(xyza,norma,(la,ma,na),alphaa,
                              xyzb,normb,(lb,mb,nb),alphab,
                              xyzc,normc,(lc,mc,nc),alphac,
                              xyzd,normd,(ld,md,nd-1),alphad)
                )
    return hgp_vrr(xyza,norma,(la,ma,na),alphaa,
               xyzb,normb,alphab,
               xyzc,normc,(lc,mc,nc),alphac,
               xyzd,normd,alphad,0)
@njit(cache=True,fastmath=True, error_model='numpy', nogil=True)
def hgp_vrr(xyza,norma,lmna,alphaa,
        xyzb,normb,alphab,
        xyzc,normc,lmnc,alphac,
        xyzd,normd,alphad,M):

    la,ma,na = lmna
    lc,mc,nc = lmnc
    xa,ya,za = xyza
    xb,yb,zb = xyzb
    xc,yc,zc = xyzc
    xd,yd,zd = xyzd

    px,py,pz = xyzp = gaussian_product_center(alphaa,xyza,alphab,xyzb)
    qx,qy,qz = xyzq = gaussian_product_center(alphac,xyzc,alphad,xyzd)
    zeta,eta = float(alphaa+alphab),float(alphac+alphad)
    wx,wy,wz = xyzw = gaussian_product_center(zeta,xyzp,eta,xyzq)

    rab2 = np.power(xa-xb,2) + np.power(ya-yb,2) + np.power(za-zb,2)
    Kab = np.sqrt(2)*np.power(np.pi,1.25)/(alphaa+alphab)\
          *np.exp(-alphaa*alphab/(alphaa+alphab)*rab2)
    rcd2 = np.power(xc-xd,2) + np.power(yc-yd,2) + np.power(zc-zd,2)
    Kcd = np.sqrt(2)*np.power(np.pi,1.25)/(alphac+alphad)\
          *np.exp(-alphac*alphad/(alphac+alphad)*rcd2)
    rpq2 = np.power(px-qx,2) + np.power(py-qy,2) + np.power(pz-qz,2)
    T = zeta*eta/(zeta+eta)*rpq2

    mtot = la+ma+na+lc+mc+nc+M

    Fgterms = [0]*(mtot+1)
    # Fgterms[mtot] = Fgamma(mtot,T)
    for im in range(mtot-1,-1,-1):
        Fgterms[im]=(2.*T*Fgterms[im+1]+np.exp(-T))/(2.*im+1)

    # Todo: setup this as a regular array

    # Store the vrr values as a 7 dimensional array
    # vrr_terms[la,ma,na,lc,mc,nc,m]
    vrr_terms = {}
    for im in range(mtot+1):
        vrr_terms[0,0,0,0,0,0,im] = (
            norma*normb*normc*normd*Kab*Kcd/np.sqrt(zeta+eta)*Fgterms[im]
            )

    # Todo: use itertools.product() for the nested for loops
    for i in range(la):
        for im in range(mtot-i):
            vrr_terms[i+1,0,0, 0,0,0, im] = (
                (px-xa)*vrr_terms[i,0,0, 0,0,0, im]
                + (wx-px)*vrr_terms[i,0,0, 0,0,0, im+1]
                )
            if i:
                vrr_terms[i+1,0,0, 0,0,0, im] += (
                    i/2./zeta*( vrr_terms[i-1,0,0, 0,0,0, im]
                               - eta/(zeta+eta)*vrr_terms[i-1,0,0, 0,0,0, im+1]
                               ))

    for j in range(ma):
        for i in range(la+1):
            for im in range(mtot-i-j):
                vrr_terms[i,j+1,0, 0,0,0, im] = (
                    (py-ya)*vrr_terms[i,j,0, 0,0,0, im]
                    + (wy-py)*vrr_terms[i,j,0, 0,0,0, im+1]
                    )
                if j:
                    vrr_terms[i,j+1,0, 0,0,0, im] += (
                        j/2./zeta*(vrr_terms[i,j-1,0, 0,0,0, im]
                                  - eta/(zeta+eta)
                                  *vrr_terms[i,j-1,0, 0,0,0, im+1]
                                  ))


    for k in range(na):
        for j in range(ma+1):
            for i in range(la+1):
                for im in range(mtot-i-j-k):
                    vrr_terms[i,j,k+1, 0,0,0, im] = (
                        (pz-za)*vrr_terms[i,j,k, 0,0,0, im]
                        + (wz-pz)*vrr_terms[i,j,k, 0,0,0, im+1]
                        )
                    if k:
                        vrr_terms[i,j,k+1, 0,0,0, im] += (
                            k/2./zeta*(vrr_terms[i,j,k-1, 0,0,0, im]
                                      - eta/(zeta+eta)
                                      *vrr_terms[i,j,k-1, 0,0,0, im+1]
                                      ))

    for q in range(lc):
        for k in range(na+1):
            for j in range(ma+1):
                for i in range(la+1):
                    for im in range(mtot-i-j-k-q):
                        vrr_terms[i,j,k, q+1,0,0, im] = (
                            (qx-xc)*vrr_terms[i,j,k, q,0,0, im]
                            + (wx-qx)*vrr_terms[i,j,k, q,0,0, im+1]
                            )
                        if q:
                            vrr_terms[i,j,k, q+1,0,0, im] += (
                                q/2./eta*(vrr_terms[i,j,k, q-1,0,0, im]
                                         - zeta/(zeta+eta)
                                         *vrr_terms[i,j,k, q-1,0,0, im+1]
                                         ))
                        if i:
                            vrr_terms[i,j,k, q+1,0,0, im] += (
                                i/2./(zeta+eta)*vrr_terms[i-1,j,k, q,0,0, im+1]
                                )

    for r in range(mc):
        for q in range(lc+1):
            for k in range(na+1):
                for j in range(ma+1):
                    for i in range(la+1):
                        for im in range(mtot-i-j-k-q-r):
                            vrr_terms[i,j,k, q,r+1,0, im] = (
                                (qy-yc)*vrr_terms[i,j,k, q,r,0, im]
                                + (wy-qy)*vrr_terms[i,j,k, q,r,0, im+1]
                                )
                            if r:
                                vrr_terms[i,j,k, q,r+1,0, im] += (
                                    r/2./eta*(vrr_terms[i,j,k, q,r-1,0, im]
                                             - zeta/(zeta+eta)
                                             *vrr_terms[i,j,k, q,r-1,0, im+1]
                                             ))
                            if j:
                                vrr_terms[i,j,k, q,r+1,0, im] += (
                                    j/2./(zeta+eta)*vrr_terms[i,j-1,k,q,r,0,im+1]
                                    )

    for s in range(nc):
        for r in range(mc+1):
            for q in range(lc+1):
                for k in range(na+1):
                    for j in range(ma+1):
                        for i in range(la+1):
                            for im in range(mtot-i-j-k-q-r-s):
                                vrr_terms[i,j,k,q,r,s+1,im] = (
                                    (qz-zc)*vrr_terms[i,j,k,q,r,s,im]
                                    + (wz-qz)*vrr_terms[i,j,k,q,r,s,im+1]
                                    )
                                if s:
                                    vrr_terms[i,j,k,q,r,s+1,im] += (
                                        s/2./eta*(vrr_terms[i,j,k,q,r,s-1,im]
                                                 - zeta/(zeta+eta)
                                                 *vrr_terms[i,j,k,q,r,s-1,im+1]
                                                 ))
                                if k:
                                    vrr_terms[i,j,k,q,r,s+1,im] += (
                                        k/2./(zeta+eta)*vrr_terms[i,j,k-1,q,r,s,im+1]
                                        )
    return vrr_terms[la,ma,na,lc,mc,nc,M]
