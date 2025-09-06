import numpy as np
from numba import njit , prange

from .integral_helpers import c2k, vlriPartial, Fboys, fastFactorial

def nuc_mat_grad_symm(basis, mol, slice=None, sqrt_ints4c2e_diag=None):
    #Here the lists are converted to numpy arrays for better use with Numba.
    #Once these conversions are done we pass these to a Numba decorated
    #function that uses prange, etc. to calculate the matrix efficiently.

    # This function calculates the nuclear matrix for a given basis object.
    # The basis object holds the information of basis functions like: exponents, coeffs, etc.
    # It is possible to only calculate a slice (block) of the complete matrix.
    # slice is a 4 element list whose first and second elements give the range of the rows to be calculated.
    # the third and fourth element give the range of columns to be calculated.
    # slice = [start_row, end_row, start_col, end_col]
    # The integrals are performed using the formulas

    #We convert the required properties to numpy arrays as this is what Numba likes.
    bfs_coords = np.array([basis.bfs_coords])
    bfs_contr_prim_norms = np.array([basis.bfs_contr_prim_norms])
    bfs_lmn = np.array([basis.bfs_lmn])
    bfs_nprim = np.array([basis.bfs_nprim])
    coordsBohrs = np.array([mol.coordsBohrs])
    Z = np.array([mol.Zcharges])
    natoms = mol.natoms
        

    #The remaining properties like bfs_coeffs are a list of lists of unequal sizes.
    #Numba won't be able to work with these efficiently.
    #So, we convert them to a numpy 2d array by applying a trick,
    #that the second dimension is that of the largest list. So that
    #it can accomadate all the lists.
    maxnprim = max(basis.bfs_nprim)
    bfs_coeffs = np.zeros([basis.bfs_nao, maxnprim])
    bfs_expnts = np.zeros([basis.bfs_nao, maxnprim])
    bfs_prim_norms = np.zeros([basis.bfs_nao, maxnprim])
    for i in range(basis.bfs_nao):
        for j in range(basis.bfs_nprim[i]):
            bfs_coeffs[i,j] = basis.bfs_coeffs[i][j]
            bfs_expnts[i,j] = basis.bfs_expnts[i][j]
            bfs_prim_norms[i,j] = basis.bfs_prim_norms[i][j]
        


    if slice is None:
        slice = [0, basis.bfs_nao, 0, basis.bfs_nao]
        
    #Limits for the calculation of overlap integrals
    a = int(slice[0]) #row start index
    b = int(slice[1]) #row end index
    c = int(slice[2]) #column start index
    d = int(slice[3]) #column end index

    # print([a,b,c,d])

    if sqrt_ints4c2e_diag is None:
        #Create dummy array
        sqrt_ints4c2e_diag = np.zeros((1,1), dtype=np.float64)
        isSchwarz = False
    else:
        isSchwarz = True
    
    V = nuc_mat_grad_symm_internal(bfs_coords[0], bfs_contr_prim_norms[0], bfs_lmn[0], bfs_nprim[0], bfs_coeffs, bfs_prim_norms, bfs_expnts, a, b, c, d, Z[0], coordsBohrs[0], natoms, sqrt_ints4c2e_diag, isSchwarz)
    
    return V 

@njit(parallel=True, cache=True, fastmath=True, error_model="numpy")
def nuc_mat_grad_symm_internal(bfs_coords, bfs_contr_prim_norms, bfs_lmn, bfs_nprim, bfs_coeffs, bfs_prim_norms, bfs_expnts, start_row, end_row, start_col, end_col, Z, coordsMol, natoms, sqrt_ints4c2e_diag, isSchwarz = False):
    # This function calculates the nuclear potential matrix and uses the symmetry property to only calculate half-ish the elements
    # and get the remaining half by symmetry.
    # The basis object holds the information of basis functions like: exponents, coeffs, etc.
    # The integrals are performed using the formulas https://pubs.acs.org/doi/full/10.1021/acs.jchemed.8b00255

    # (A|-Z_C/r_{iC}|B) = 
    #Using numba-scipy allows us to use scipy.special.gamma and gamminc,
    #however, this prevents the caching functionality. Nevertheless,
    #apart form the compilation overhead, it allows to perform calculaitons significantly faster and with good 
    #accuracy.


    # Infer the matrix shape from the start and end indices
    num_rows = end_row - start_row 
    num_cols = end_col - start_col 
    matrix_shape = (natoms, 3, num_rows, num_cols)

    
    # Check if the slice of the matrix requested falls in the lower/upper triangle or in both the triangles
    upper_tri = False
    lower_tri = False
    both_tri_symm = False
    both_tri_nonsymm = False
    if end_row <= start_col:
        upper_tri = True
    elif start_row >= end_col:
        lower_tri = True
    elif start_row==start_col and end_row==end_col:
        both_tri_symm = True
    else:
        both_tri_nonsymm = True
    
    # print(both_tri_symm)

    # Initialize the matrix with zeros
    V = np.zeros(matrix_shape) 
    PI = 3.141592653589793
    PIx2 = 6.283185307179586 #2*PI
    
    
    #Loop over BFs
    for i in prange(start_row, end_row): 
        I = bfs_coords[i]
        Ni = bfs_contr_prim_norms[i]
        lmni = bfs_lmn[i]
        la, ma, na = lmni
        for j in prange(start_col, end_col):
            
            
            if lower_tri or upper_tri or (both_tri_symm and j<=i) or both_tri_nonsymm:
                if isSchwarz:
                    sqrt_ij = sqrt_ints4c2e_diag[i,j]
                    if sqrt_ij*sqrt_ij<1e-13:
                        continue
                result = 0.0
                J = bfs_coords[j]
                IJ = I - J  
                IJsq = np.sum(IJ**2)
                
                Nj = bfs_contr_prim_norms[j]

                temp_NiNj = Ni*Nj
                
                lmnj = bfs_lmn[j]
                
                lb, mb, nb = lmnj

                facl = np.zeros(la+lb+1)
                facm = np.zeros(ma+mb+1)
                facn = np.zeros(na+nb+1)
                F_ = np.zeros((la+lb+ma+mb+na+nb+1)) 
                vntk = np.zeros((na+nb+1, (na+nb)//2+1, (na+nb)//2+1)) # Good for upto j shells; j orbitals have an angular momentum of 7;
                vmsj = np.zeros((ma+mb+1, (ma+mb)//2+1, (ma+mb)//2+1)) # Good for upto j shells; j orbitals have an angular momentum of 7;

                #Loop over primitives
                for ik in range(bfs_nprim[i]):     #Parallelising over primitives doesn't seem to make a difference
                    alphaik = bfs_expnts[i,ik]
                    dik = bfs_coeffs[i,ik]
                    Nik = bfs_prim_norms[i,ik]
                    temp_alphaikIjsq = alphaik*IJsq
                    temp_alphaikI = alphaik*I
                    temp_NiNjNikdik = temp_NiNj*Nik*dik
                    for jk in range(bfs_nprim[j]):
                        
                        alphajk = bfs_expnts[j,jk]
                        gamma = alphaik + alphajk
                        gamma_inv = 1/gamma
                        # screenfactor = np.exp(-alphaik*alphajk*gamma_inv*IJsq)
                        screenfactor = np.exp(-temp_alphaikIjsq*alphajk*gamma_inv)
                        if abs(screenfactor)<1.0e-8:   
                        # Going lower than E-8 doesn't change the max erroor. There could still be some effects from error compounding but the max error doesnt budge.
                        #TODO: This is quite low. But since this is the slowest part.
                        #But I had to do this because this is a very slow part of the program.
                        #Will have to check how the accuracy is affected and if the screening factor
                        #can be reduced further.
                            continue
                        

                        
                        djk = bfs_coeffs[j,jk] 
                        
                        Njk = bfs_prim_norms[j,jk]

                        # screenfactor_2 = djk*Njk*temp_NiNjNikdik*screenfactor*gamma_inv*np.sqrt(gamma_inv)*15.5031383401
                        # if abs(screenfactor_2)<1.0e-10: # The threshold used here should be the same as Schwarz screening threshold (for 4c2e and 3c2e ERIs)
                        #     continue 
                        
                        epsilon = 0.25*gamma_inv
                        P = (temp_alphaikI + alphajk*J)*gamma_inv
                        PI = P - I
                        PJ = P - J
                        tempfac = (PIx2*gamma_inv)*screenfactor

                        for l in range(0, la+lb+1):
                            facl[l] = c2k(l,la,lb,PI[0],PJ[0])
                        for m in range(0, ma+mb+1):
                            facm[m] = c2k(m,ma,mb,PI[1],PJ[1])
                        for n in range(0, na+nb+1):
                            facn[n] = c2k(n,na,nb,PI[2],PJ[2])


                        Vc = 0.0
                        #Loop over nuclei
                        for iatom in prange(natoms): #Parallelising over atoms seems to be faster for Cholestrol.xyz with def2-QZVPPD (628 sec)
                            Rc = coordsMol[iatom]
                            Zc = Z[iatom]
                            PC = P - Rc
                            temp_gamma_sum_PCsq = gamma*np.sum(PC**2)

                            fac1 = -Zc*tempfac
                            #print(fac1)
                            sum_Vl = 0.0

                            for li in range(la+lb+ma+mb+na+nb + 1):
                                F_[li] = Fboys(li,temp_gamma_sum_PCsq)
                            for m in range(0,ma+mb+1):
                                for s in range(0, int(m/2)+1):
                                    for j1 in range(0, int((m-2*s)/2)+1):
                                        vmsj[m,s,j1] = vlriPartial(PC[1],m,s,j1)*epsilon**(s+j1)*facm[m]
                            for n in range(0,na+nb+1):
                                for t in range(0, int(n/2)+1):
                                    for k in range(0, int((n-2*t)/2)+1):
                                        vntk[n,t,k] = vlriPartial(PC[2],n,t,k)*epsilon**(t+k)*facn[n] 
                            
                            
                            for l in range(0,la+lb+1):
                                # facl = c2k(l,la,lb,PI[0],PJ[0])
                                # fac_l = fastFactorial(l)
                                for r in range(0, int(l/2)+1):
                                    # fac_r = fastFactorial(r)
                                    for i1 in range(0, int((l-2*r)/2)+1):
                                        v_lri = vlriPartial(PC[0],l,r,i1)*epsilon**(r+i1)*facl[l]
                                        # v_lri = (-1)**l*((-1)**i1*fac_l*PC[0]**(l-2*r-2*i1)/(fac_r*fastFactorial(i1)*fastFactorial(l-2*r-2*i1)))*epsilon**(r+i1)*facl[l]
                                        sum_Vm = 0.0
                                        for m in range(0,ma+mb+1):
                                            # facm = c2k(m,ma,mb,PI[1],PJ[1])
                                            # fac_m = fastFactorial(m)
                                            for s in range(0, int(m/2)+1):
                                                # fac_s = fastFactorial(s)
                                                for j1 in range(0, int((m-2*s)/2)+1):
                                                    # v_msj = vlriPartial(PC[1],m,s,j1)*epsilon**(s+j1)*facm[m]
                                                    v_msj = vmsj[m,s,j1]
                                                    # v_msj = (-1)**m*((-1)**j1*fac_m*PC[1]**(m-2*s-2*j1)/(fac_s*fastFactorial(j1)*fastFactorial(m-2*s-2*j1)))*epsilon**(s+j1)*facm[m]
                                                    sum_Vn = 0.0
                                                    for n in range(0,na+nb+1):
                                                        # facn = c2k(n,na,nb,PI[2],PJ[2])
                                                        # fac_n = fastFactorial(n)
                                                        for t in range(0, int(n/2)+1):
                                                            # fac_t = fastFactorial(t)
                                                            for k in range(0, int((n-2*t)/2)+1):
                                                                # v_ntk = vlriPartial(PC[2],n,t,k)*epsilon**(t+k)*facn[n]
                                                                v_ntk = vntk[n,t,k]
                                                                # v_ntk = (-1)**n*((-1)**k*fac_n*PC[2]**(n-2*t-2*k)/(fac_t*fastFactorial(k)*fastFactorial(n-2*t-2*k)))*epsilon**(t+k)*facn[n]
                                                                # F = Fboys(l+m+n-2*(r+s+t)-(i1+j1+k),temp_gamma_sum_PCsq) 
                                                                F = F_[l+m+n-2*(r+s+t)-(i1+j1+k)]
                                                                
                                                                sum_Vn += v_ntk*F
                                                    sum_Vm += v_msj*sum_Vn
                                        sum_Vl += v_lri*sum_Vm
                                
                            Vc += sum_Vl*fac1
                        result += Vc*djk*Njk*temp_NiNjNikdik
                V[i - start_row, j - start_col] = result
            
      
    if both_tri_symm:
        #We save time by evaluating only the lower diagonal elements and then use symmetry Vi,j=Vj,i 
        for i in prange(start_row, end_row):
            for j in prange(start_col, end_col):
                if j>i:
                    V[i-start_row, j-start_col] = V[j-start_col, i-start_row]
            
    return V       

# This version parallelizes over atoms
@njit(parallel=True, cache=True, fastmath=True, error_model="numpy")
def nuc_mat_grad_symm_internal2(bfs_coords, bfs_contr_prim_norms, bfs_lmn, bfs_nprim, bfs_coeffs, bfs_prim_norms, bfs_expnts, start_row, end_row, start_col, end_col, Z, coordsMol, natoms):
    # This function calculates the nuclear potential matrix and uses the symmetry property to only calculate half-ish the elements
    # and get the remaining half by symmetry.
    # The basis object holds the information of basis functions like: exponents, coeffs, etc.
    # The integrals are performed using the formulas https://pubs.acs.org/doi/full/10.1021/acs.jchemed.8b00255

    # (A|-Z_C/r_{iC}|B) = 
    #Using numba-scipy allows us to use scipy.special.gamma and gamminc,
    #however, this prevents the caching functionality. Nevertheless,
    #apart form the compilation overhead, it allows to perform calculaitons significantly faster and with good 
    #accuracy.

    # Infer the matrix shape from the start and end indices
    num_rows = end_row - start_row 
    num_cols = end_col - start_col 
    matrix_shape = (natoms, 3, num_rows, num_cols)

    
    # Check if the slice of the matrix requested falls in the lower/upper triangle or in both the triangles
    upper_tri = False
    lower_tri = False
    both_tri_symm = False
    both_tri_nonsymm = False
    if end_row <= start_col:
        upper_tri = True
    elif start_row >= end_col:
        lower_tri = True
    elif start_row==start_col and end_row==end_col:
        both_tri_symm = True
    else:
        both_tri_nonsymm = True
    
    # print(both_tri_symm)

    # Initialize the matrix with zeros
    V = np.zeros(matrix_shape) 
    PI = 3.141592653589793
    PIx2 = 6.283185307179586 #2*PI
    
    #Loop over nuclei
    for iatom in prange(natoms): #Parallelising over atoms seems to be faster for Cholestrol.xyz with def2-QZVPPD (628 sec)
        Rc = coordsMol[iatom]
        Zc = Z[iatom]
        
        #Loop over BFs
        for i in range(start_row, end_row): 
            I = bfs_coords[i]
            Ni = bfs_contr_prim_norms[i]
            lmni = bfs_lmn[i]
            la, ma, na = lmni
            #Loop over primitives
            for ik in range(bfs_nprim[i]):     #Parallelising over primitives doesn't seem to make a difference
                alphaik = bfs_expnts[i,ik]
                dik = bfs_coeffs[i,ik]
                Nik = bfs_prim_norms[i,ik]
                for j in range(start_col, end_col):
                    if lower_tri or upper_tri or (both_tri_symm and j<=i) or both_tri_nonsymm:
                        result = 0.0
                        J = bfs_coords[j]
                        IJ = I - J  
                        IJsq = np.sum(IJ**2)
                        
                        Nj = bfs_contr_prim_norms[j]
                        
                        lmnj = bfs_lmn[j]
                        
                        lb, mb, nb = lmnj
                        
                        for jk in range(bfs_nprim[j]):
                            
                            alphajk = bfs_expnts[j,jk]
                            gamma = alphaik + alphajk
                            gamma_inv = 1/gamma
                            screenfactor = np.exp(-alphaik*alphajk*gamma_inv*IJsq)
                            if abs(screenfactor)<1.0e-8:   
                            # Going lower than E-8 doesn't change the max erroor. There could still be some effects from error compounding but the max error doesnt budge.
                            #TODO: This is quite low. But since this is the slowest part.
                            #But I had to do this because this is a very slow part of the program.
                            #Will have to check how the accuracy is affected and if the screening factor
                            #can be reduced further.
                                continue

                            
                            djk = bfs_coeffs[j,jk] 
                            
                            Njk = bfs_prim_norms[j,jk]
                            
                            epsilon = 0.25*gamma_inv
                            P = (alphaik*I + alphajk*J)*gamma_inv
                            PC = P - Rc
                            PI = P - I
                            PJ = P - J
                            tempfac = (PIx2*gamma_inv)*screenfactor

                            Vc = 0.0
                            

                            fac1 = -Zc*tempfac
                            #print(fac1)
                            sum_Vl = 0.0
                            
                            
                            for l in range(0,la+lb+1):
                                facl = c2k(l,la,lb,PI[0],PJ[0])
                                for r in range(0, int(l/2)+1):
                                    for i1 in range(0, int((l-2*r)/2)+1):
                                        v_lri = vlriPartial(PC[0],l,r,i1)*epsilon**(r+i1)*facl
                                        sum_Vm = 0.0
                                        for m in range(0,ma+mb+1):
                                            facm = c2k(m,ma,mb,PI[1],PJ[1])
                                            for s in range(0, int(m/2)+1):
                                                for j1 in range(0, int((m-2*s)/2)+1):
                                                    v_msj = vlriPartial(PC[1],m,s,j1)*epsilon**(s+j1)*facm
                                                    sum_Vn = 0.0
                                                    for n in range(0,na+nb+1):
                                                        facn = c2k(n,na,nb,PI[2],PJ[2])
                                                        for t in range(0, int(n/2)+1):
                                                            for k in range(0, int((n-2*t)/2)+1):
                                                                v_ntk = vlriPartial(PC[2],n,t,k)*epsilon**(t+k)*facn
                                                                F = Fboys(l+m+n-2*(r+s+t)-(i1+j1+k),gamma*np.sum(PC**2)) 
                                                                
                                                                sum_Vn += v_ntk*F
                                                    sum_Vm += v_msj*sum_Vn
                                        sum_Vl += v_lri*sum_Vm
                                
                            Vc += sum_Vl*fac1
                            result += Vc*dik*djk*Nik*Njk*Ni*Nj
                        V[i - start_row, j - start_col] += result
            
      
    if both_tri_symm:
        #We save time by evaluating only the lower diagonal elements and then use symmetry Vi,j=Vj,i 
        for i in prange(start_row, end_row):
            for j in prange(start_col, end_col):
                if j>i:
                    V[i-start_row, j-start_col] = V[j-start_col, i-start_row]
            
    return V       
    