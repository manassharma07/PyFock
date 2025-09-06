import numpy as np
from numba import njit , prange

from .integral_helpers import calcS

def kin_mat_symm(basis, slice=None):
    # Here the lists are converted to numpy arrays for better use with Numba.
    # Once these conversions are done we pass these to a Numba decorated
    # function that uses prange, etc. to calculate the matrix efficiently.

    # This function calculates the kinetic energy matrix for a given basis object.
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

        
    T = kin_mat_symm_internal(bfs_coords[0], bfs_contr_prim_norms[0], bfs_lmn[0], bfs_nprim[0], bfs_coeffs, bfs_prim_norms, bfs_expnts, a, b, c, d)
    
    return T 

@njit(parallel=True, cache=True)
def kin_mat_symm_internal(bfs_coords, bfs_contr_prim_norms, bfs_lmn, bfs_nprim, bfs_coeffs, bfs_prim_norms, bfs_expnts, start_row, end_row, start_col, end_col):
    # This function calculates the kinetic energy matrix and uses the symmetry property to only calculate half-ish the elements
    # and get the remaining half by symmetry.
    # The reason we need this extra function is because we want the callable function to be simple and not require so many 
    # arguments. But when using Numba to optimize, we can't have too many custom objects and stuff. Numba likes numpy arrays
    # so passing those is okay. But lists and custom objects are not okay.
    # This function calculates the kinetic matrix for a given basis object.
    # The basis object holds the information of basis functions like: exponents, coeffs, etc.
    # It is possible to only calculate a slice (block) of the complete matrix.
    # slice is a 4 element list whose first and second elements give the range of the rows to be calculated.
    # the third and fourth element give the range of columns to be calculated.
    # The integrals are performed using the formulas here https://pubs.acs.org/doi/full/10.1021/acs.jchemed.8b00255

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


    # Initialize the matrix with zeros
    T = np.zeros(matrix_shape) 


    for i in prange(start_row, end_row):
        I = bfs_coords[i]
        Ni = bfs_contr_prim_norms[i]
        lmni = bfs_lmn[i]
        for j in range(start_col, end_col): #Because we are only evaluating the lower triangular matrix.
            if lower_tri or upper_tri or (both_tri_symm and j<=i) or both_tri_nonsymm:
                result_sum = 0.0
                J = bfs_coords[j]
                IJ = I - J  
                Nj = bfs_contr_prim_norms[j]
                lmnj = bfs_lmn[j]
                #Some factors to save FLOPS
                fac1 = np.sum(IJ**2)
                fac2 = Ni*Nj
                fac3 = (2*(lmnj[0]+lmnj[1]+lmnj[2])+3)
                fac4 = (lmnj[0]*(lmnj[0]-1))
                fac5 = (lmnj[1]*(lmnj[1]-1))
                fac6 = (lmnj[2]*(lmnj[2]-1))
                for ik in range(bfs_nprim[i]):
                    for jk in range(bfs_nprim[j]):
                        alphaik = bfs_expnts[i,ik]
                        alphajk = bfs_expnts[j,jk]
                        gamma = alphaik + alphajk
                        temp_1 = np.exp(-alphaik*alphajk/gamma*fac1)
                        if (abs(temp_1)<1.0e-9):
                            continue

                        dik = bfs_coeffs[i,ik]
                        djk = bfs_coeffs[j,jk] 
                        Nik = bfs_prim_norms[i,ik]
                        Njk = bfs_prim_norms[j,jk]
                        
                        P = (alphaik*I + alphajk*J)/gamma
                        PI = P - I
                        PJ = P - J

                        temp = dik*djk  #coeff of primitives as read from basis set
                        temp = temp*Nik*Njk #normalization factors of primitives
                        temp = temp*fac2 #normalization factor of the contraction of primitives

                        

                        Sx = calcS(lmni[0],lmnj[0],gamma,PI[0],PJ[0])
                        Sy = calcS(lmni[1],lmnj[1],gamma,PI[1],PJ[1])
                        Sz = calcS(lmni[2],lmnj[2],gamma,PI[2],PJ[2])
                        overlap1 = Sx*Sy*Sz

                        Sx = calcS(lmni[0],lmnj[0]+2,gamma,PI[0],PJ[0])
                        overlap2 = Sx*Sy*Sz

                        Sx = calcS(lmni[0],lmnj[0],gamma,PI[0],PJ[0])
                        Sy = calcS(lmni[1],lmnj[1]+2,gamma,PI[1],PJ[1])
                        overlap3 = Sx*Sy*Sz

                        Sy = calcS(lmni[1],lmnj[1],gamma,PI[1],PJ[1])
                        Sz = calcS(lmni[2],lmnj[2]+2,gamma,PI[2],PJ[2])
                        overlap4 = Sx*Sy*Sz

                        Sx = calcS(lmni[0],lmnj[0]-2,gamma,PI[0],PJ[0])
                        Sz = calcS(lmni[2],lmnj[2],gamma,PI[2],PJ[2])
                        overlap5 = Sx*Sy*Sz

                        Sx = calcS(lmni[0],lmnj[0],gamma,PI[0],PJ[0])
                        Sy = calcS(lmni[1],lmnj[1]-2,gamma,PI[1],PJ[1])
                        overlap6 = Sx*Sy*Sz

                        Sy = calcS(lmni[1],lmnj[1],gamma,PI[1],PJ[1])
                        Sz = calcS(lmni[2],lmnj[2]-2,gamma,PI[2],PJ[2])
                        overlap7 = Sx*Sy*Sz

                        part1 = overlap1*alphajk*fac3
                        part2 = 2*alphajk*alphajk*(overlap2+overlap3+overlap4)
                        part3 = fac4*overlap5
                        part4 = fac5*overlap6
                        part5 = fac6*overlap7

                        result = temp*(part1 - part2 - 0.5*(part3+part4+part5))*temp_1
                        
                        result_sum += result
                T[i - start_row, j - start_col] = result_sum
                
            
    if both_tri_symm:
        #We save time by evaluating only the lower diagonal elements and then use symmetry Ti,j=Tj,i 
        for i in prange(start_row, end_row):
            for j in range(start_col, end_col):
                if j>i:
                    T[i-start_row, j-start_col] = T[j-start_col, i-start_row]
                
    return T