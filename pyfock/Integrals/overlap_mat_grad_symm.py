import numpy as np
from numba import njit , prange

from .integral_helpers import calcS

def overlap_mat_grad_symm(basis, slice=None):
    # Here the lists are converted to numpy arrays for better use with Numba.
    # Once these conversions are done we pass these to a Numba decorated
    # function that uses prange, etc. to calculate the matrix efficiently.

    # This function calculates the overlap matrix for a given basis object.
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
    bfs_atoms = np.array([basis.bfs_atoms])
    natoms = max(basis.bfs_atoms) + 1
        

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
        
    dS = overlap_mat_grad_symm_internal_new(natoms, bfs_atoms[0], bfs_coords[0], bfs_contr_prim_norms[0], bfs_lmn[0], bfs_nprim[0], bfs_coeffs, bfs_prim_norms, bfs_expnts, a, b, c, d)
    return dS

@njit(parallel=True, fastmath=True, cache=True)
def overlap_mat_grad_symm_internal(natoms, bfs_atoms, bfs_coords, bfs_contr_prim_norms, bfs_lmn, bfs_nprim, bfs_coeffs, bfs_prim_norms, bfs_expnts, start_row, end_row, start_col, end_col):
    # This function calculates the overlap matrix and uses the symmetry property to only calculate half-ish the elements
    # and get the remaining half by symmetry.
    # The reason we need this extra function is because we want the callable function to be simple and not require so many 
    # arguments. But when using Numba to optimize, we can't have too many custom objects and stuff. Numba likes numpy arrays
    # so passing those is okay. But lists and custom objects are not okay.
    # This function calculates the overlap matrix for a given basis object.
    # The basis object holds the information of basis functions like: exponents, coeffs, etc.
    # It is possible to only calculate a slice (block) of the complete matrix.
    # slice is a 4 element list whose first and second elements give the range of the rows to be calculated.
    # the third and fourth element give the range of columns to be calculated.
    # The integrals are performed using the formulas here https://pubs.acs.org/doi/full/10.1021/acs.jchemed.8b00255

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

    # is_ibf_on_atom = False
    # is_jbf_on_atom = False

    # Initialize the matrix with zeros
    dS = np.zeros(matrix_shape) 

    for iatom in prange(natoms):
        for dir in range(3):
            for i in range(start_row, end_row):
                I = bfs_coords[i]
                lmni = bfs_lmn[i]
                Ni = bfs_contr_prim_norms[i]
                if iatom==bfs_atoms[i]:
                    is_ibf_on_atom = True
                else:
                    is_ibf_on_atom = False
                for j in range(start_col, end_col): #Because we are only evaluating the lower triangular matrix.
                    if lower_tri or upper_tri or (both_tri_symm and j<=i) or both_tri_nonsymm:
                        result = 0.0
                        if iatom==bfs_atoms[j]:
                            is_jbf_on_atom = True
                        else:
                            is_jbf_on_atom = False

                        # if not is_ibf_on_atom and not is_jbf_on_atom:
                        #     continue
                        if is_ibf_on_atom or is_jbf_on_atom:
                            
                            J = bfs_coords[j]
                            IJ = I - J  
                            tempfac = np.sum(IJ**2)
                            
                            Nj = bfs_contr_prim_norms[j]
                            
                            lmnj = bfs_lmn[j]
                            for ik in range(bfs_nprim[i]):
                                alphaik = bfs_expnts[i][ik]
                                dik = bfs_coeffs[i][ik]
                                Nik = bfs_prim_norms[i][ik]
                                for jk in range(bfs_nprim[j]):

                                    
                                    alphajk = bfs_expnts[j][jk]
                                    gamma = alphaik + alphajk
                                    screenfactor = np.exp(-alphaik*alphajk/gamma*tempfac)
                                    if (abs(screenfactor)<1.0e-12):
                                        continue
                                    
                                    djk = bfs_coeffs[j][jk] 
                                    Njk = bfs_prim_norms[j][jk]
                                    
                                    P = (alphaik*I + alphajk*J)/gamma
                                    PI = P - I
                                    PJ = P - J
                                    
                                    temp = dik*djk
                                    temp = temp*Nik*Njk
                                    temp = temp*Ni*Nj
                                    if is_ibf_on_atom and not is_jbf_on_atom:
                                        if dir==0: #x
                                            lfactor = lmni[0]
                                            Sx = calcS(lmni[0]-1,lmnj[0],gamma,PI[0],PJ[0])
                                            Sy = calcS(lmni[1],lmnj[1],gamma,PI[1],PJ[1])
                                            Sz = calcS(lmni[2],lmnj[2],gamma,PI[2],PJ[2])
                                        if dir==1: #y
                                            lfactor = lmni[1]
                                            Sx = calcS(lmni[0],lmnj[0],gamma,PI[0],PJ[0])
                                            Sy = calcS(lmni[1]-1,lmnj[1],gamma,PI[1],PJ[1])
                                            Sz = calcS(lmni[2],lmnj[2],gamma,PI[2],PJ[2])
                                        if dir==2: #z
                                            lfactor = lmni[2]
                                            Sx = calcS(lmni[0],lmnj[0],gamma,PI[0],PJ[0])
                                            Sy = calcS(lmni[1],lmnj[1],gamma,PI[1],PJ[1])
                                            Sz = calcS(lmni[2]-1,lmnj[2],gamma,PI[2],PJ[2])
                                        tempA = -lfactor*temp*screenfactor*Sx*Sy*Sz
                                        if dir==0: #x
                                            Sx = calcS(lmni[0]+1,lmnj[0],gamma,PI[0],PJ[0])
                                        if dir==1: #y
                                            Sy = calcS(lmni[1]+1,lmnj[1],gamma,PI[1],PJ[1])
                                        if dir==2: #z
                                            Sz = calcS(lmni[2]+1,lmnj[2],gamma,PI[2],PJ[2])
                                        tempB = 2*alphaik*temp*screenfactor*Sx*Sy*Sz
                                        result += tempA + tempB
                                    if not is_ibf_on_atom and is_jbf_on_atom:
                                        if dir==0: #x
                                            lfactor = lmnj[0]
                                            Sx = calcS(lmni[0],lmnj[0]-1,gamma,PI[0],PJ[0])
                                            Sy = calcS(lmni[1],lmnj[1],gamma,PI[1],PJ[1])
                                            Sz = calcS(lmni[2],lmnj[2],gamma,PI[2],PJ[2])
                                        if dir==1: #y
                                            lfactor = lmnj[1]
                                            Sx = calcS(lmni[0],lmnj[0],gamma,PI[0],PJ[0])
                                            Sy = calcS(lmni[1],lmnj[1]-1,gamma,PI[1],PJ[1])
                                            Sz = calcS(lmni[2],lmnj[2],gamma,PI[2],PJ[2])
                                        if dir==2: #z
                                            lfactor = lmnj[2]
                                            Sx = calcS(lmni[0],lmnj[0],gamma,PI[0],PJ[0])
                                            Sy = calcS(lmni[1],lmnj[1],gamma,PI[1],PJ[1])
                                            Sz = calcS(lmni[2],lmnj[2]-1,gamma,PI[2],PJ[2])
                                        tempA = -lfactor*temp*screenfactor*Sx*Sy*Sz
                                        if dir==0: #x
                                            Sx = calcS(lmni[0],lmnj[0]+1,gamma,PI[0],PJ[0])
                                        if dir==1: #y
                                            Sy = calcS(lmni[1],lmnj[1]+1,gamma,PI[1],PJ[1])
                                        if dir==2: #z
                                            Sz = calcS(lmni[2],lmnj[2]+1,gamma,PI[2],PJ[2])
                                        tempB = 2*alphajk*temp*screenfactor*Sx*Sy*Sz
                                        result += tempA + tempB
                                    if is_ibf_on_atom and is_jbf_on_atom:
                                        continue
                                        # Apply chain rule and compute the derivative of bra
                                        if dir==0: #x
                                            lfactor = lmni[0]
                                            Sx = calcS(lmni[0]-1,lmnj[0],gamma,PI[0],PJ[0])
                                            Sy = calcS(lmni[1],lmnj[1],gamma,PI[1],PJ[1])
                                            Sz = calcS(lmni[2],lmnj[2],gamma,PI[2],PJ[2])
                                        if dir==1: #y
                                            lfactor = lmni[1]
                                            Sx = calcS(lmni[0],lmnj[0],gamma,PI[0],PJ[0])
                                            Sy = calcS(lmni[1]-1,lmnj[1],gamma,PI[1],PJ[1])
                                            Sz = calcS(lmni[2],lmnj[2],gamma,PI[2],PJ[2])
                                        if dir==2: #z
                                            lfactor = lmni[2]
                                            Sx = calcS(lmni[0],lmnj[0],gamma,PI[0],PJ[0])
                                            Sy = calcS(lmni[1],lmnj[1],gamma,PI[1],PJ[1])
                                            Sz = calcS(lmni[2]-1,lmnj[2],gamma,PI[2],PJ[2])
                                        tempA = -lfactor*temp*screenfactor*Sx*Sy*Sz
                                        if dir==0: #x
                                            Sx = calcS(lmni[0]+1,lmnj[0],gamma,PI[0],PJ[0])
                                        if dir==1: #y
                                            Sy = calcS(lmni[1]+1,lmnj[1],gamma,PI[1],PJ[1])
                                        if dir==2: #z
                                            Sz = calcS(lmni[2]+1,lmnj[2],gamma,PI[2],PJ[2])
                                        tempB = 2*alphaik*temp*screenfactor*Sx*Sy*Sz
                                        result += tempA + tempB

                                        # Now compute the derivative of the ket
                                        if dir==0: #x
                                            lfactor = lmnj[0]
                                            Sx = calcS(lmni[0],lmnj[0]-1,gamma,PI[0],PJ[0])
                                            Sy = calcS(lmni[1],lmnj[1],gamma,PI[1],PJ[1])
                                            Sz = calcS(lmni[2],lmnj[2],gamma,PI[2],PJ[2])
                                        if dir==1: #y
                                            lfactor = lmnj[1]
                                            Sx = calcS(lmni[0],lmnj[0],gamma,PI[0],PJ[0])
                                            Sy = calcS(lmni[1],lmnj[1]-1,gamma,PI[1],PJ[1])
                                            Sz = calcS(lmni[2],lmnj[2],gamma,PI[2],PJ[2])
                                        if dir==2: #z
                                            lfactor = lmnj[2]
                                            Sx = calcS(lmni[0],lmnj[0],gamma,PI[0],PJ[0])
                                            Sy = calcS(lmni[1],lmnj[1],gamma,PI[1],PJ[1])
                                            Sz = calcS(lmni[2],lmnj[2]-1,gamma,PI[2],PJ[2])
                                        tempA = -lfactor*temp*screenfactor*Sx*Sy*Sz
                                        if dir==0: #x
                                            Sx = calcS(lmni[0],lmnj[0]+1,gamma,PI[0],PJ[0])
                                        if dir==1: #y
                                            Sy = calcS(lmni[1],lmnj[1]+1,gamma,PI[1],PJ[1])
                                        if dir==2: #z
                                            Sz = calcS(lmni[2],lmnj[2]+1,gamma,PI[2],PJ[2])
                                        tempB = 2*alphajk*temp*screenfactor*Sx*Sy*Sz
                                        result += tempA + tempB
                                
                            dS[iatom, dir, i - start_row, j - start_col] = result
                            if both_tri_symm:
                                dS[iatom, dir, j - start_col, i - start_row] = result

    # if both_tri_symm:
    #     #We save time by evaluating only the lower diagonal elements and then use symmetry Si,j=Sj,i 
    #     for i in prange(start_row, end_row):
    #         for j in prange(start_col, end_col):
    #             if j>i:
    #                 dS[:, :, i-start_row, j-start_col] = dS[:, :, j-start_col, i-start_row]
    return dS

@njit(parallel=True, fastmath=True, cache=True)
def overlap_mat_grad_symm_internal_new(natoms, bfs_atoms, bfs_coords, bfs_contr_prim_norms, bfs_lmn, bfs_nprim, bfs_coeffs, bfs_prim_norms, bfs_expnts, start_row, end_row, start_col, end_col):
    # This function calculates the overlap matrix and uses the symmetry property to only calculate half-ish the elements
    # and get the remaining half by symmetry.
    # The reason we need this extra function is because we want the callable function to be simple and not require so many 
    # arguments. But when using Numba to optimize, we can't have too many custom objects and stuff. Numba likes numpy arrays
    # so passing those is okay. But lists and custom objects are not okay.
    # This function calculates the overlap matrix for a given basis object.
    # The basis object holds the information of basis functions like: exponents, coeffs, etc.
    # It is possible to only calculate a slice (block) of the complete matrix.
    # slice is a 4 element list whose first and second elements give the range of the rows to be calculated.
    # the third and fourth element give the range of columns to be calculated.
    # The integrals are performed using the formulas here https://pubs.acs.org/doi/full/10.1021/acs.jchemed.8b00255

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

    # is_ibf_on_atom = False
    # is_jbf_on_atom = False

    # Initialize the matrix with zeros
    dS = np.zeros(matrix_shape) 

        
    for i in prange(start_row, end_row):
        I = bfs_coords[i]
        lmni = bfs_lmn[i]
        Ni = bfs_contr_prim_norms[i]
        for j in range(start_col, end_col): #Because we are only evaluating the lower triangular matrix.
            if lower_tri or upper_tri or (both_tri_symm and j<=i) or both_tri_nonsymm:
                result = np.zeros(3)#0.0

                if bfs_atoms[i]==bfs_atoms[j]:
                    continue
                    
                J = bfs_coords[j]
                IJ = I - J  
                tempfac = np.sum(IJ**2)
                
                Nj = bfs_contr_prim_norms[j]
                
                lmnj = bfs_lmn[j]
                for ik in range(bfs_nprim[i]):
                    alphaik = bfs_expnts[i][ik]
                    dik = bfs_coeffs[i][ik]
                    Nik = bfs_prim_norms[i][ik]
                    for jk in range(bfs_nprim[j]):

                        
                        alphajk = bfs_expnts[j][jk]
                        gamma = alphaik + alphajk
                        screenfactor = np.exp(-alphaik*alphajk/gamma*tempfac)
                        if (abs(screenfactor)<1.0e-12):
                            continue
                        
                        djk = bfs_coeffs[j][jk] 
                        Njk = bfs_prim_norms[j][jk]
                        
                        P = (alphaik*I + alphajk*J)/gamma
                        PI = P - I
                        PJ = P - J
                        
                        temp = dik*djk
                        temp = temp*Nik*Njk
                        temp = temp*Ni*Nj
                        
                        for dir in range(3):
                            if dir==0: #x
                                lfactor = lmni[0]
                                Sx = calcS(lmni[0]-1,lmnj[0],gamma,PI[0],PJ[0])
                                Sy = calcS(lmni[1],lmnj[1],gamma,PI[1],PJ[1])
                                Sz = calcS(lmni[2],lmnj[2],gamma,PI[2],PJ[2])
                            if dir==1: #y
                                lfactor = lmni[1]
                                Sx = calcS(lmni[0],lmnj[0],gamma,PI[0],PJ[0])
                                Sy = calcS(lmni[1]-1,lmnj[1],gamma,PI[1],PJ[1])
                                Sz = calcS(lmni[2],lmnj[2],gamma,PI[2],PJ[2])
                            if dir==2: #z
                                lfactor = lmni[2]
                                Sx = calcS(lmni[0],lmnj[0],gamma,PI[0],PJ[0])
                                Sy = calcS(lmni[1],lmnj[1],gamma,PI[1],PJ[1])
                                Sz = calcS(lmni[2]-1,lmnj[2],gamma,PI[2],PJ[2])
                            tempA = -lfactor*temp*screenfactor*Sx*Sy*Sz
                            if dir==0: #x
                                Sx = calcS(lmni[0]+1,lmnj[0],gamma,PI[0],PJ[0])
                            if dir==1: #y
                                Sy = calcS(lmni[1]+1,lmnj[1],gamma,PI[1],PJ[1])
                            if dir==2: #z
                                Sz = calcS(lmni[2]+1,lmnj[2],gamma,PI[2],PJ[2])
                            tempB = 2*alphaik*temp*screenfactor*Sx*Sy*Sz
                            result[dir] += tempA + tempB
                        
                        
                    
                            dS[bfs_atoms[i], dir, i - start_row, j - start_col] = result[dir]
                            dS[bfs_atoms[j], dir, i - start_row, j - start_col] = -result[dir]
                            if both_tri_symm:
                                dS[bfs_atoms[i], dir, j - start_col, i - start_row] = result[dir]
                                dS[bfs_atoms[j], dir, j - start_col, i - start_row] = -result[dir]

    # if both_tri_symm:
    #     #We save time by evaluating only the lower diagonal elements and then use symmetry Si,j=Sj,i 
    #     for i in prange(start_row, end_row):
    #         for j in prange(start_col, end_col):
    #             if j>i:
    #                 dS[:, :, i-start_row, j-start_col] = dS[:, :, j-start_col, i-start_row]
    return dS