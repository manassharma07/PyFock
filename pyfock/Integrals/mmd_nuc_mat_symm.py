import numpy as np
from numba import njit , prange

from .integral_helpers import c2k, vlriPartial, Fboys, hermite_gauss_coeff, aux_hermite_int

def mmd_nuc_mat_symm(basis, mol, slice=None):
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
    
    V = mmd_nuc_mat_symm_internal(bfs_coords[0], bfs_contr_prim_norms[0], bfs_lmn[0], bfs_nprim[0], bfs_coeffs, bfs_prim_norms, bfs_expnts, a, b, c, d, Z[0], coordsBohrs[0], natoms)
    
    return V 

@njit(parallel=True, cache=False, fastmath=True, error_model="numpy")
def mmd_nuc_mat_symm_internal(bfs_coords, bfs_contr_prim_norms, bfs_lmn, bfs_nprim, bfs_coeffs, bfs_prim_norms, bfs_expnts, start_row, end_row, start_col, end_col, Z, coordsMol, natoms):
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
    matrix_shape = (num_rows, num_cols)

    
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
                result = 0.0
                J = bfs_coords[j]
                IJ = I - J  
                IJsq = np.sum(IJ**2)
                
                Nj = bfs_contr_prim_norms[j]
                
                lmnj = bfs_lmn[j]
                
                lb, mb, nb = lmnj
                #Loop over primitives
                for ik in range(bfs_nprim[i]):     #Parallelising over primitives doesn't seem to make a difference
                    alphaik = bfs_expnts[i,ik]
                    dik = bfs_coeffs[i,ik]
                    Nik = bfs_prim_norms[i,ik]
                    for jk in range(bfs_nprim[j]):
                        
                        alphajk = bfs_expnts[j,jk]
                        gamma = alphaik + alphajk
                        gamma_inv = 1/gamma
                        temp_gamma = alphaik*alphajk*gamma_inv
                        screenfactor = np.exp(-temp_gamma*IJsq)
                        if abs(screenfactor)<1.0e-8:   
                        # Going lower than E-8 doesn't change the max erroor. There could still be some effects from error compounding but the max error doesnt budge.
                        #TODO: This is quite low. But since this is the slowest part.
                        #But I had to do this because this is a very slow part of the program.
                        #Will have to check how the accuracy is affected and if the screening factor
                        #can be reduced further.
                            continue

                        
                        djk = bfs_coeffs[j,jk] 
                        
                        Njk = bfs_prim_norms[j,jk]
                        
                        # epsilon = 0.25*gamma_inv
                        P = (alphaik*I + alphajk*J)*gamma_inv
                        # PI = P - I
                        # PJ = P - J
                        tempfac = (PIx2*gamma_inv)

                        Vc = 0.0
                        #Loop over nuclei
                        for iatom in range(natoms): #Parallelising over atoms seems to be faster for Cholestrol.xyz with def2-QZVPPD (628 sec)
                            Rc = coordsMol[iatom]
                            Zc = Z[iatom]
                            PC = P - Rc
                            # RPC = np.linalg.norm(PC)
                            RPC = np.sqrt(np.sum(PC**2))

                            fac1 = -Zc*tempfac
                            #print(fac1)
                            sum_Vl = 0.0
                            
                            
                            for t in range(la+lb+1):
                                for u in range(ma+mb+1):
                                    for v in range(na+nb+1):
                                        sum_Vl += hermite_gauss_coeff(la,lb,t,IJ[0],alphaik,alphajk,gamma,temp_gamma) * \
                                            hermite_gauss_coeff(ma,mb,u,IJ[1],alphaik,alphajk,gamma,temp_gamma) * \
                                            hermite_gauss_coeff(na,nb,v,IJ[2],alphaik,alphajk,gamma,temp_gamma) * \
                                            aux_hermite_int(t,u,v,0,gamma,PC[0],PC[1],PC[2],RPC)
                                
                            Vc += sum_Vl*fac1
                        result += Vc*dik*djk*Nik*Njk*Ni*Nj
                V[i - start_row, j - start_col] = result
            
      
    if both_tri_symm:
        #We save time by evaluating only the lower diagonal elements and then use symmetry Vi,j=Vj,i 
        for i in prange(start_row, end_row):
            for j in prange(start_col, end_col):
                if j>i:
                    V[i-start_row, j-start_col] = V[j-start_col, i-start_row]
            
    return V             
    