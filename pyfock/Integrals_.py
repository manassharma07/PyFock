# Integrals.py
# Author: Manas Sharma (manassharma07@live.com)
# This is a part of CrysX (https://bragitoff.com/crysx)
#
# 
#			 .d8888b.                            Y88b   d88P      
#			d88P  Y88b                            Y88b d88P       
#			888    888                             Y88o88P        
#			888        888d888 888  888 .d8888b     Y888P         
#			888        888P"   888  888 88K         d888b         
#			888    888 888     888  888 "Y8888b.   d88888b        
#			Y88b  d88P 888     Y88b 888      X88  d88P Y88b       
#			 "Y8888P"  888      "Y88888  88888P' d88P   Y88b      
#			                        888                           
#			                   Y8b d88P                           
#			                    "Y88P"    
from threadpoolctl import ThreadpoolController, threadpool_info, threadpool_limits
import numpy as np
import os
import ctypes
import scipy 
import sys
import math 
import functools
import numba_scipy
from scipy.special import factorial2, binom, hyp1f1, gammainc, gamma
from scipy.sparse import csc_matrix
from . import Basis
from opt_einsum import contract
from opt_einsum import contract_expression
try:
    from numba import vectorize,jit,njit,prange,set_num_threads,get_num_threads   
    NUMBA_EXISTS = True
except Exception as e:
    NUMBA_EXISTS = False
    pass
try:
    from joblib import Parallel, delayed
    JOBLIB_EXISTS = True
except Exception as e:
    JOBLIB_EXISTS = False
    pass
try:
    import pylibxc
    LIBXC_EXISTS = True
except Exception as e:
    LIBXC_EXISTS = False
    pass
try:
    import numexpr
    NUMEXPR_EXISTS = True
except Exception as e:
    NUMEXPR_EXISTS = False
    pass
from timeit import default_timer as timer
from . import densfuncs
from autograd import grad
import autograd.numpy as npautograd
import gc
import psutil 
import sparse
#The functions with @njit decorator are outside the class Integrals.
#these provide acceleration by using the JIT feature of Numba.

#NOTE: For a function to work efficiently with Numba and take its advantage, it has to be cleverly written.
# First of all, it should only take in numpy arrays as arguments.
# The sizes of the lists should be predefined and not like Python.
# The loops can usually be replaced with prange to leverage parallel execution.
# Take care of race-condition when using prange.
# While numpy functions are supported, scipy functions are not supported so write your own.


# BIG TODO: Rewrite factorial and double factorial functions to use loops rather than recursion.
# Loops would definitely be faster.
#@njit(cache=True)
def fac1(n):
     if n <= 0:
         return 1
     else:
         return n * doublefactorial(n-1)
#@njit(cache=True)
def fastFactorial1(n):
    # UPDATE: Look up table versio is extremely slowww
    # return fac(n)
    # LOOKUP_TABLE = [
    # 1, 1, 2, 6, 24, 120, 720, 5040, 40320,
    # 362880, 3628800, 39916800, 479001600,
    # 6227020800, 87178291200, 1307674368000,
    # 20922789888000, 355687428096000, 6402373705728000,
    # 121645100408832000, 2432902008176640000]
    # if n<=20:
    #     return LOOKUP_TABLE[n]
    # else:
    # Update
    # loop is working the best
        if n<= 1:
            return 1
        else:
            factorial = 1
            for i in range(2, n+1):
                factorial *= i
            return factorial

#@njit(cache=True)
def comb1(x, y):
    # binom = fac(x) // fac(y) // fac(x - y)
    binom = fastFactorial2(x) // fastFactorial2(y) // fastFactorial2(x - y)
    return binom
#@njit(cache=True)
def doublefactorial1(n):
# Double Factorial Implementation based on recursion
     if n <= 0:
         return 1
     else:
         return n * doublefactorial(n-2)
        
# def doublefactorial1(n):
# # Double factorial implementation based on loops
# # Seems to be faster
# # The overlap matrix gets f****d up with this
#     res = 1
#     for i in range(n, -1, -2):
#         if(i == 0 or i == 1):
#             return res
#         else:
#             res *= i

#@njit(cache=True)   
def c2kNumba1(k,la,lb,PA,PB):
    temp = 0.0
    for i in range(la+1):
        factor1 = comb(la,i)
        factor2 = PA**(la-i)
        for j in range(lb+1):
            if (i+j)==k :
                temp +=  factor1*comb(lb,j)*factor2*PB**(lb-j)
    return temp
#@njit(cache=True)
def calcSNumba1(la,lb,gamma,PA,PB):
    temp = 0.0
    fac1 = np.sqrt(np.pi/gamma)
    fac2 = 2*gamma
    for k in range(0, int((la+lb)/2)+1):
        temp +=  c2kNumba(2*k,la,lb,PA,PB)*fac1*doublefactorial(2*k-1)/(fac2)**k
    return temp

#@njit(parallel=True, cache=True)
def matMulNumba1(A, B):
    assert A.shape[1] == B.shape[0]
    res = np.zeros((A.shape[0], B.shape[1]), )
    for i in range(A.shape[0]):
        for j in range(B.shape[1]):
            res[i,j] += A[i,:] * B[:,j]
            # for k in prange(A.shape[1]):
            #     res[i,j] += A[i,k] * B[k,j]
    return res
#@njit(parallel=True, cache=True)
def calcVtempNumba1(z, ao_val, nao):
    # Try to simulate this
    # v_temp = z @ ao_value_block  # The fastest uptil now
    # v_temp += v_temp.T 
    
    V = np.zeros((nao, nao))
    V_temp = np.dot(z, ao_val)
    for i in prange(nao):
        for j in prange(i+1):
            V[i,j] = V_temp[i,j] + V_temp[j,i]
            V[j,i] = V[i,j]
    
    # New try (Extreeeeeemmmlyyyyy slowwwwwwwwwww)
    # ao_val_trans = ao_val.T
    # ncoords = ao_val.shape[0]
    # for i in prange(nao):
    #     if np.abs(z[i,:]).max()<1e-8:
    #         continue
    #     for j in prange(i+1):
    #         if np.abs(ao_val[j,:]).max()<1e-8:
    #             continue
            
    #         val = 0.0
    #         for k in prange(ncoords):
    #             val += z[i,k]*ao_val[k,j] + z[k,i]*ao_val[j,k]
            
    #         V[i,j] = val
    #         V[j,i] = val
    

    return V

#@njit(parallel=True, cache=True)
# def calcVtempNumba1(z, ao_val):
#     assert z.shape[1] == ao_val.shape[0]
#     res = np.zeros((z.shape[0], ao_val.shape[1]))
#     # for k in range(ao_val.shape[0]):
#     #     for i in range(z.shape[0]):
#     #         if np.abs(z[i,k])<1E-10:
#     #             continue
#     #         for j in range(ao_val.shape[1]):
#     #             if np.abs(ao_val[k,j])< 1E-10:
#     #                 continue
#     #             res[i,j]+=z[i,k]*ao_val[k,j]
#     for i in range(z.shape[0]):
#         for j in range(ao_val.shape[1]):
#             val = 0.0
#             for k in prange(ao_val.shape[0]):
#                 if np.abs(z[i,k])<1E-10:
#                     continue
#                 if np.abs(ao_val[k,j])< 1E-10:
#                     continue
#                 val+=z[i,k]*ao_val[k,j]
#             res[i,j] = val
#     return res

#@njit(parallel=True, cache=True)
def evalRhoNumba1_old(bf_values, densmat, coords):
    # This is the Numba decorated faster version of evalRho.
    # The arguments of this function are different than evalRho.
    # Evaluates the value of density at a given grid ('coord') nx3 array
    # For this we need the density matrix as well as the information 
    # the basis functions in the basis object.
    # rho at the grid point m is given as:
    # \rho^m = \sum_{\mu}\sum_{\nu} D_{\mu \nu} \mu^m \nu^m
    rho = np.zeros((coords.shape[0]))
    ncoords = coords.shape[0] 
    nbfs = bf_values.shape[1]

    # #Approach 1 (has the coords loop as the outer most)
    # #Not very efficient!
    # #Loop over grid points
    # for m in prange(ncoords):
    #     #Loop over BFs
    #     for i in prange(nbfs):
    #         mu = bf_values[m,i]
    #         if abs(mu)<1.0e-6:  #This condition helps in accelerating the speed by a factor of almost 2.
    #             continue
    #         for j in prange(nbfs):
    #             nu = bf_values[m,j]
    #             if abs(nu)<1.0e-6: #A value of 6-8 is good for an accuracy og 7-8 decimal places.
    #                 continue
    #             if abs(densmat[i,j])<1.0e-9:
    #                 continue
    #             rho[m] = rho[m] + densmat[i,j]*mu*nu
    #             #rho[m] = rho[m] + densmat[i,j]*bf_values[m,i]*bf_values[m,j]

    #New try
    #!!!einsum Not supported by NUMBA!!!
    #rho = np.einsum('ij,mi,mj->m',densmat,bf_values,bf_values) 
    #temp = np.dot(bf_values, densmat)

    #New try
    #Loop over coords should be the innermost loop
    #Loop over BFs
    for i in range(nbfs):
        for j in range(nbfs):
            if abs(densmat[i,j])<1.0e-9: #A value of 10-14 is good for an accuracy of 7-8 decimal places.
                continue
            dens = densmat[i,j]
            #rho = rho
            #Loop over grid points
            for m in prange(ncoords):
                mu = bf_values[m,i]
                if abs(mu)<1.0e-6:  #This condition helps in accelerating the speed by a factor of almost 10-100.
                    continue
                nu = bf_values[m,j]
                if abs(nu)<1.0e-6: #A value of 10-14 is good for an accuracy of 7-8 decimal places.
                    continue
                rho[m] = rho[m] + dens*mu*nu
                # y = rho[m] + dens*mu*nu
                # rho[m] = y


    return rho 

#@njit(parallel=True, cache=True)
def evalRhoNumba1(bf_values, densmat):
    # This is the Numba decorated faster version of evalRho.
    # The arguments of this function are different than evalRho.
    # Evaluates the value of density at a given grid ('coord') nx3 array
    # For this we need the density matrix as well as the information 
    # the basis functions in the basis object.
    # rho at the grid point m is given as:
    # \rho^m = \sum_{\mu}\sum_{\nu} D_{\mu \nu} \mu^m \nu^m
    rho = np.zeros((bf_values.shape[0]))
    ncoords = bf_values.shape[0] 
    nbfs = bf_values.shape[1]

    # #Approach 1 (has the coords loop as the outer most)
    # #Not very efficient!
    # #Loop over grid points
    # for m in prange(ncoords):
    #     #Loop over BFs
    #     for i in prange(nbfs):
    #         mu = bf_values[m,i]
    #         if abs(mu)<1.0e-6:  #This condition helps in accelerating the speed by a factor of almost 2.
    #             continue
    #         for j in prange(nbfs):
    #             nu = bf_values[m,j]
    #             if abs(nu)<1.0e-6: #A value of 6-8 is good for an accuracy og 7-8 decimal places.
    #                 continue
    #             if abs(densmat[i,j])<1.0e-9:
    #                 continue
    #             rho[m] = rho[m] + densmat[i,j]*mu*nu
    #             #rho[m] = rho[m] + densmat[i,j]*bf_values[m,i]*bf_values[m,j]

    #New try
    #!!!einsum Not supported by NUMBA!!!
    #rho = np.einsum('ij,mi,mj->m',densmat,bf_values,bf_values) 
    #temp = np.dot(bf_values, densmat)

    #New try (VERYYY SLOWWW)
    #Loop over coords should be the innermost loop
    #Loop over BFs
    # for i in prange(nbfs):
    #     for j in prange(nbfs):
    #         if abs(densmat[i,j])<1.0e-9: #A value of 10-14 is good for an accuracy of 7-8 decimal places.
    #             continue
    #         dens = densmat[i,j]
            
    #         mu = bf_values[:,i]
    #         nu = bf_values[:,j]
    #         rho = rho + dens*mu*nu

    # Lets try to simulate this
    # tempo = bf_values @ densmat
    # rho_block = contract('mi, mi -> m', bf_values, tempo)
    # Finally the following is as fast as the above snippet (with @ and opt_einsum.contract)
    # temp_value = np.dot(bf_values, densmat)
    # for m in prange(ncoords):
    #     rho_temp = 0.0
    #     for i in range(nbfs):
    #         rho_temp += bf_values[m,i]*temp_value[m,i]
    #     rho[m] = rho_temp

    # The following should have been faster than above, but is pretty much only as fast
    # temp_value = np.dot(bf_values, densmat)
    # for m in prange(ncoords):
    #     rho[m] = np.dot(bf_values[m,:], temp_value[m,:])

    # The following is the fastest because here we utilize symmetry information and also skip calculation for small values
    #Loop over BFs
    for m in prange(ncoords):
        rho_temp = 0.0
        for i in prange(nbfs):
            mu = bf_values[m,i]
            if abs(mu)<1.0e-8: #A value of 8 is good for an accuracy of 7-8 decimal places.
                    continue
            for j in prange(i+1):
                dens = densmat[i,j]
                if abs(dens)<1.0e-8: #A value of 9 is good for an accuracy of 7-8 decimal places.
                    continue
                if i==j: # Diagonal terms
                    nu = mu
                    rho_temp += dens*mu*nu 
                else: # Non-diagonal terms
                    nu = bf_values[m,j]
                    if abs(nu)<1.0e-8: #A value of 8 is good for an accuracy of 7-8 decimal places.
                        continue
                    rho_temp += 2*dens*mu*nu 
        rho[m] = rho_temp


    return rho 

#@njit(cache=True)
def evalGTOandGradNumba1(alpha, coeff, lmn, x, y, z):
    # A very low-level way to calculate the ao values as well as theor gradients simultaneously, without 
    # running similar calls again and again.
    # value = np.zeros((4))
    # Prelims
    # x = coord[0]-coordCenter[0]
    # y = coord[1]-coordCenter[1]
    # z = coord[2]-coordCenter[2]
    xl = x**lmn[0]
    ym = y**lmn[1]
    zn = z**lmn[2]
    exp = np.exp(-alpha*(x**2+y**2+z**2))
    factor2 = coeff*exp

    # AO Value
    # value[0] = factor2*xl*ym*zn
    value0 = factor2*xl*ym*zn
    # Grad x
    if np.abs(x-0)<1e-14:
        # value[1] = 0.0
        value1 = 0.0
    else:
        xl = x**(lmn[0]-1)
        factor = (lmn[0]-2*alpha*x**2)
        # value[1] = factor2*xl*ym*zn*factor
        value1 = factor2*xl*ym*zn*factor
    # Grad y
    if np.abs(y-0)<1e-10:
        # value[2] = 0.0
        value2 = 0.0
    else:
        xl = x**lmn[0]
        ym = y**(lmn[1]-1)
        factor = (lmn[1]-2*alpha*y**2)
        # value[2] = factor2*xl*ym*zn*factor 
        value2 = factor2*xl*ym*zn*factor 
    # Grad z 
    if np.abs(z-0)<1e-14:
        # value[3] = 0.0
        value3 = 0.0
    else:
        zn = z**(lmn[2]-1)
        xl = x**lmn[0]
        ym = y**lmn[1]  
        factor = (lmn[2]-2*alpha*z**2)
        # value[3] = factor2*xl*ym*zn*factor
        value3 = factor2*xl*ym*zn*factor
    # return value
    return value0, value1, value2, value3

#@njit(cache=True)
def evalGTONumba1(alpha, coeff, lmn, x, y, z):
    #This function evaluates the value of a given Gaussian primitive 
    # with given values of alpha (exponent), coefficient, and angular momentum
    # centered at 'coordCenter' (3 comp, numpy array). The value is calculated at
    # a given 'coord'.

    # instead of writing the whole formula in 1 line, it was suggested to break it down so that Numba can compile them efficiently.
    # Seems to provide a small speedup.
    xl = x**lmn[0]
    ym = y**lmn[1]
    zn = z**lmn[2]
    exp = np.exp(-alpha*(x**2+y**2+z**2))
    value = coeff*xl*ym*zn*exp


    # oneliner
    # value = coeff*((coord[0]-coordCenter[0])**lmn[0]*(coord[1]-coordCenter[1])**lmn[1]*(coord[2]-coordCenter[2])**lmn[2])*np.exp(-alpha*((coord[0]-coordCenter[0])**2+(coord[1]-coordCenter[1])**2+(coord[2]-coordCenter[2])**2))
    
    
    return value

#@njit(cache=True)
def evalGTOgradxNumba1(alpha, coeff, coordCenter, lmn, coord):
    # Returns the gradient of a given Gaussian primitive along x
    # x = coord[0]-coordCenter[0]
    # # The following is hack because the gradient can blow up if the coord is the same as Gaussian center
    # if np.abs(x-0)<1e-14:
    #     return 0.0
    # value = coeff*((x)**(lmn[0]-1)*(coord[1]-coordCenter[1])**lmn[1]*(coord[2]-coordCenter[2])**lmn[2])*(lmn[0]-2*alpha*(coordCenter[0]-coord[0])**2)*np.exp(-alpha*((x)**2+(coord[1]-coordCenter[1])**2+(coord[2]-coordCenter[2])**2))
    
    # NEW way (Here we split up the code in multiple lines rather than a single line as we had before)
    # This helps NUMBA to compile more efficiently
    x = (coord[0]-coordCenter[0])
    # The following is hack because the gradient can blow up if the coord is the same as Gaussian center
    if np.abs(x-0)<1e-14:
        return 0.0
    y = (coord[1]-coordCenter[1])
    z = (coord[2]-coordCenter[2])
    xl = x**(lmn[0]-1)
    ym = y**lmn[1]
    zn = z**lmn[2]
    exp = np.exp(-alpha*(x**2+y**2+z**2))
    factor = (lmn[0]-2*alpha*x**2)
    value = coeff*xl*ym*zn*factor*exp
    
    
    return value

#@njit(cache=True)
def evalGTOgradyNumba1(alpha, coeff, coordCenter, lmn, coord):
    # Returns the gradient of a given Gaussian primitive along y
    # y = coord[1]-coordCenter[1]
    # # The following is hack because the gradient can blow up if the coord is the same as Gaussian center
    # if np.abs(y-0)<1e-14:
    #     return 0.0
    # value = coeff*((coord[0]-coordCenter[0])**lmn[0]*(y)**(lmn[1]-1)*(coord[2]-coordCenter[2])**lmn[2])*(lmn[1]-2*alpha*(coordCenter[1]-coord[1])**2)*np.exp(-alpha*((coord[0]-coordCenter[0])**2+(y)**2+(coord[2]-coordCenter[2])**2))
    
    # NEW way (Here we split up the code in multiple lines rather than a single line as we had before)
    # This helps NUMBA to compile more efficiently
    x = (coord[0]-coordCenter[0])
    y = (coord[1]-coordCenter[1])
    # The following is hack because the gradient can blow up if the coord is the same as Gaussian center
    if np.abs(y-0)<1e-14:
        return 0.0
    z = (coord[2]-coordCenter[2])
    xl = x**lmn[0]
    ym = y**(lmn[1]-1)
    zn = z**lmn[2]
    exp = np.exp(-alpha*(x**2+y**2+z**2))
    factor = (lmn[1]-2*alpha*y**2)
    value = coeff*xl*ym*zn*factor*exp
    
    return value

#@njit(cache=True)
def evalGTOgradzNumba1(alpha, coeff, coordCenter, lmn, coord):
    # Returns the gradient of a given Gaussian primitive along z
    # z = coord[2]-coordCenter[2]
    # # The following is hack because the gradient can blow up if the coord is the same as Gaussian center
    # if np.abs(z-0)<1e-14:
    #     return 0.0
    # value = coeff*((coord[0]-coordCenter[0])**lmn[0]*(coord[1]-coordCenter[1])**lmn[1]*(z)**(lmn[2]-1))*(lmn[2]-2*alpha*(coordCenter[2]-coord[2])**2)*np.exp(-alpha*((coord[0]-coordCenter[0])**2+(coord[1]-coordCenter[1])**2+(z)**2))
    
    # NEW way (Here we split up the code in multiple lines rather than a single line as we had before)
    # This helps NUMBA to compile more efficiently
    x = (coord[0]-coordCenter[0])
    y = (coord[1]-coordCenter[1])
    z = (coord[2]-coordCenter[2])
    # The following is hack because the gradient can blow up if the coord is the same as Gaussian center
    if np.abs(z-0)<1e-14:
        return 0.0
    xl = x**lmn[0]
    ym = y**lmn[1]
    zn = z**(lmn[2]-1)
    exp = np.exp(-alpha*(x**2+y**2+z**2))
    factor = (lmn[2]-2*alpha*z**2)
    value = coeff*xl*ym*zn*factor*exp
    
    return value

#@njit(parallel=True, cache=True)
#TODO this is incomplete
def evalBFiNumba(coordi, Ni, Nik, dik, lmni, alphaik, coord):
    #This function evaluates the value of a given Basis function at a given grid point (coord)
    #'coord' is a 3 element 1d-array
    value = 0.0
    #Loop over primitives
    # nprim = 9
    for ik in prange(basis.bfs_nprim[i]):
        dik = basis.bfs_coeffs[i][ik] 
        Nik = basis.bfs_prim_norms[i][ik]
        alphaik = basis.bfs_expnts[i][ik]
        value = value + evalGTONumba(alphaik, Ni*Nik*dik, coordi, lmni, coord)
        # print(Ni*Nik*dik)
        # print(Integrals.evalGTO(alphaik, Ni*Nik*dik, coordi, lmni, coord))
            

    return value

#@njit(parallel=True, cache=True)
def evalBFsgradNumba1(bfs_coords, bfs_contr_prim_norms, bfs_nprim, bfs_lmn, bfs_coeffs, bfs_prim_norms, bfs_expnts, coord):
    #This function evaluates the value of all the given Basis functions on the grid (coord).
    # 'coord' should be a nx3 array
        
    nao = bfs_coords.shape[0]
    ncoord = coord.shape[0]
    result = np.zeros((3, ncoord, nao))

    #TODO I guess a better way would be to iterate over BFs first, instead of coords.
    #Also, it would be better to skip the loop over coords altoghether and use numpy instead for the evaluation, by just passing
    #the coords array to evalGTO. I am pretty sure this would increase the speed substantially. This could be wrong though,
    #since we are in fact using JIT so maybe no more optimization is possible. But worth a try.

    #Loop over grid points
    for k in prange(ncoord):
        #Loop over BFs
        for i in range(nao):
            valuex = 0.0
            valuey = 0.0
            valuez = 0.0
            coordi = bfs_coords[i]
            Ni = bfs_contr_prim_norms[i]
            lmni = bfs_lmn[i]
            for ik in range(bfs_nprim[i]):
                dik = bfs_coeffs[i][ik] 
                Nik = bfs_prim_norms[i][ik]
                alphaik = bfs_expnts[i][ik]
                valuex = valuex + evalGTOgradxNumba2(alphaik, Ni*Nik*dik, coordi, lmni, coord[k])
                valuey = valuey + evalGTOgradyNumba2(alphaik, Ni*Nik*dik, coordi, lmni, coord[k])
                valuez = valuez + evalGTOgradzNumba2(alphaik, Ni*Nik*dik, coordi, lmni, coord[k])
            result[0,k,i] = valuex
            result[1,k,i] = valuey
            result[2,k,i] = valuez
    
    # Loop over BFs (This is lowerrr and could also have race condition)
    # for i in range(nao):
    #     coordi = bfs_coords[i]
    #     Ni = bfs_contr_prim_norms[i]
    #     lmni = bfs_lmn[i]
    #     #Loop over primitives
    #     for ik in range(bfs_nprim[i]):
    #         dik = bfs_coeffs[i][ik] 
    #         Nik = bfs_prim_norms[i][ik]
    #         alphaik = bfs_expnts[i][ik]
    #         #Loop over grid points
    #         for k in prange(ncoord):
    #             result[0,k,i] += evalGTOgradxNumba2(alphaik, Ni*Nik*dik, coordi, lmni, coord[k])
    #             result[1,k,i] += evalGTOgradyNumba2(alphaik, Ni*Nik*dik, coordi, lmni, coord[k])
    #             result[2,k,i] += evalGTOgradzNumba2(alphaik, Ni*Nik*dik, coordi, lmni, coord[k])
            
                
    return result


#@njit(parallel=True, cache=True)
def evalBFsandgradNumba1(bfs_coords, bfs_contr_prim_norms, bfs_nprim, bfs_lmn, bfs_coeffs, bfs_prim_norms, bfs_expnts, bfs_radius_cutoff, coord):
    #This function evaluates the value of all the given Basis functions on the grid (coord).
    # 'coord' should be a nx3 array
        
    nao = bfs_coords.shape[0]
    ncoord = coord.shape[0]
    # result = np.zeros((4, ncoord, nao))
    result1 = np.zeros((ncoord,nao))
    result2 = np.zeros((3,ncoord,nao))

    #TODO I guess a better way would be to iterate over BFs first, instead of coords.
    #Also, it would be better to skip the loop over coords altoghether and use numpy instead for the evaluation, by just passing
    #the coords array to evalGTO. I am pretty sure this would increase the speed substantially. This could be wrong though,
    #since we are in fact using JIT so maybe no more optimization is possible. But worth a try.

    #Loop over grid points
    for k in prange(ncoord):
        coord_grid = coord[k]
        #Loop over BFs
        for i in range(nao):
            value_ao = 0.0
            valuex = 0.0
            valuey = 0.0
            valuez = 0.0
            # values[0] = 0.0
            # values[1] = 0.0
            # values[2] = 0.0
            # values[3] = 0.0
            coord_bf = bfs_coords[i]
            x = coord_grid[0]-coord_bf[0]
            y = coord_grid[1]-coord_bf[1]
            z = coord_grid[2]-coord_bf[2]
            if (np.sqrt(x**2+y**2+z**2)>bfs_radius_cutoff[i]):
                continue
            Ni = bfs_contr_prim_norms[i]
            lmni = bfs_lmn[i]
            #cutoff_radius = 0 #Cutoff radius for this basis function
            for ik in range(bfs_nprim[i]):
                dik = bfs_coeffs[i, ik] 
                Nik = bfs_prim_norms[i, ik]
                alphaik = bfs_expnts[i, ik]
                a,b,c,d = evalGTOandGradNumba(alphaik, Ni*Nik*dik, lmni, x, y, z)
                value_ao = value_ao + a
                valuex = valuex + b
                valuey = valuey + c
                valuez = valuez + d
            # result[0,k,i] = value_ao
            # result[1,k,i] = valuex
            # result[2,k,i] = valuey
            # result[3,k,i] = valuez
            result1[k,i] = value_ao
            result2[0,k,i] = valuex
            result2[1,k,i] = valuey
            result2[2,k,i] = valuez
            
            
    
    # Loop over BFs (This is lowerrr and could also have race condition)
    # for i in range(nao):
    #     coordi = bfs_coords[i]
    #     Ni = bfs_contr_prim_norms[i]
    #     lmni = bfs_lmn[i]
    #     #Loop over primitives
    #     for ik in range(bfs_nprim[i]):
    #         dik = bfs_coeffs[i][ik] 
    #         Nik = bfs_prim_norms[i][ik]
    #         alphaik = bfs_expnts[i][ik]
    #         #Loop over grid points
    #         for k in prange(ncoord):
    #             result[0,k,i] += evalGTOgradxNumba2(alphaik, Ni*Nik*dik, coordi, lmni, coord[k])
    #             result[1,k,i] += evalGTOgradyNumba2(alphaik, Ni*Nik*dik, coordi, lmni, coord[k])
    #             result[2,k,i] += evalGTOgradzNumba2(alphaik, Ni*Nik*dik, coordi, lmni, coord[k])
            
                
    # return result
    return result1, result2


def evalBFsandRho_serialNumba1(bfs_coords, bfs_contr_prim_norms, bfs_nprim, bfs_lmn, bfs_coeffs, bfs_prim_norms, bfs_expnts, bfs_radius_cutoff, coord, densmat):
    #This function evaluates the value of all the given Basis functions on the grid (coord).
    # 'coord' should be a nx3 array
        
    nao = bfs_coords.shape[0]
    ncoord = coord.shape[0]
    result = np.zeros((ncoord, nao))
    rho = np.zeros((nao))

    #Loop over grid points
    for k in range(ncoord):
        rho_temp = 0.0
        coord_grid = coord[k]
        #Loop over BFs
        for i in range(nao):
            value = 0.0
            coord_bf = bfs_coords[i]
            x = coord_grid[0]-coord_bf[0]
            y = coord_grid[1]-coord_bf[1]
            z = coord_grid[2]-coord_bf[2]
            if (np.sqrt(x**2+y**2+z**2)<=bfs_radius_cutoff[i]):
                # continue
                Ni = bfs_contr_prim_norms[i]
                lmni = bfs_lmn[i]
                for ik in range(bfs_nprim[i]):
                    dik = bfs_coeffs[i][ik] 
                    Nik = bfs_prim_norms[i][ik]
                    alphaik = bfs_expnts[i][ik]
                    value += evalGTONumba(alphaik, Ni*Nik*dik, lmni, x, y, z)
            result[k,i] = value
        # Rho at grid points
        for ii in range(nao):
            mu = result[k,ii]
            if abs(mu)<1.0e-8: #A value of 8 is good for an accuracy of 7-8 decimal places.
                continue
            for j in range(ii+1):
                dens = densmat[ii,j]
                if abs(dens)<1.0e-8: #A value of 9 is good for an accuracy of 7-8 decimal places.
                    continue
                if ii==j: # Diagonal terms
                    nu = mu
                    rho_temp += dens*mu*nu 
                else: # Non-diagonal terms
                    nu = result[k,j]
                    if abs(nu)<1.0e-8: #A value of 8 is good for an accuracy of 7-8 decimal places.
                        continue
                    rho_temp += 2*dens*mu*nu 
        rho[k] = rho_temp
    

    return result, rho


def evalBFsandRhoNumba1(bfs_coords, bfs_contr_prim_norms, bfs_nprim, bfs_lmn, bfs_coeffs, bfs_prim_norms, bfs_expnts, bfs_radius_cutoff, coord, densmat):
    #This function evaluates the value of all the given Basis functions on the grid (coord).
    # 'coord' should be a nx3 array
        
    nao = bfs_coords.shape[0]
    ncoord = coord.shape[0]
    result = np.zeros((ncoord, nao))
    rho = np.zeros((nao))

    #TODO I guess a better way would be to iterate over BFs first, instead of coords.
    #Also, it would be better to skip the loop over coords altoghether and use numpy instead for the evaluation, by just passing
    #the coords array to evalGTO. I am pretty sure this would increase the speed substantially. This could be wrong though,
    #since we are in fact using JIT so maybe no more optimization is possible. But worth a try.

    #Loop over grid points
    for k in prange(ncoord):
        rho_temp = 0.0
        coord_grid = coord[k]
        #Loop over BFs
        for i in range(nao):
            value = 0.0
            coord_bf = bfs_coords[i]
            x = coord_grid[0]-coord_bf[0]
            y = coord_grid[1]-coord_bf[1]
            z = coord_grid[2]-coord_bf[2]
            if (np.sqrt(x**2+y**2+z**2)<=bfs_radius_cutoff[i]):
                # continue
                Ni = bfs_contr_prim_norms[i]
                lmni = bfs_lmn[i]
                for ik in range(bfs_nprim[i]):
                    dik = bfs_coeffs[i][ik] 
                    Nik = bfs_prim_norms[i][ik]
                    alphaik = bfs_expnts[i][ik]
                    value += evalGTONumba(alphaik, Ni*Nik*dik, lmni, x, y, z)
            result[k,i] = value
        # Rho at grid points
        for ii in range(nao):
            mu = result[k,ii]
            if abs(mu)<1.0e-8: #A value of 8 is good for an accuracy of 7-8 decimal places.
                    continue
            for j in range(ii+1):
                dens = densmat[ii,j]
                if abs(dens)<1.0e-8: #A value of 9 is good for an accuracy of 7-8 decimal places.
                    continue
                if ii==j: # Diagonal terms
                    nu = mu
                    rho_temp += dens*mu*nu 
                else: # Non-diagonal terms
                    nu = result[k,j]
                    if abs(mu)<1.0e-8: #A value of 8 is good for an accuracy of 7-8 decimal places.
                        continue
                    rho_temp += 2*dens*mu*nu 
        rho[k] = rho_temp
    

    return result, rho

#@njit(parallel=True, cache=True)
def evalBFsNumba1(bfs_coords, bfs_contr_prim_norms, bfs_nprim, bfs_lmn, bfs_coeffs, bfs_prim_norms, bfs_expnts, bfs_radius_cutoff, coord):
    #This function evaluates the value of all the given Basis functions on the grid (coord).
    # 'coord' should be a nx3 array
        
    nao = bfs_coords.shape[0]
    ncoord = coord.shape[0]
    result = np.zeros((ncoord, nao))

    #TODO I guess a better way would be to iterate over BFs first, instead of coords.
    #Also, it would be better to skip the loop over coords altoghether and use numpy instead for the evaluation, by just passing
    #the coords array to evalGTO. I am pretty sure this would increase the speed substantially. This could be wrong though,
    #since we are in fact using JIT so maybe no more optimization is possible. But worth a try.

    #Loop over grid points
    for k in prange(ncoord):
        coord_grid = coord[k]
        #Loop over BFs
        for i in range(nao):
            value = 0.0
            coord_bf = bfs_coords[i]
            x = coord_grid[0]-coord_bf[0]
            y = coord_grid[1]-coord_bf[1]
            z = coord_grid[2]-coord_bf[2]
            if (np.sqrt(x**2+y**2+z**2)>bfs_radius_cutoff[i]):
                continue
            Ni = bfs_contr_prim_norms[i]
            lmni = bfs_lmn[i]
            for ik in range(bfs_nprim[i]):
                dik = bfs_coeffs[i][ik] 
                Nik = bfs_prim_norms[i][ik]
                alphaik = bfs_expnts[i][ik]
                value += evalGTONumba(alphaik, Ni*Nik*dik, lmni, x, y, z)
            result[k,i] = value

    #New try (doesn't work)
    #Loop over BFs
    # for i in prange(nao):
    #     coordi = bfs_coords[i]
    #     Ni = bfs_contr_prim_norms[i]
    #     lmni = bfs_lmn[i]
    #     for ik in prange(bfs_nprim[i]):
    #         dik = bfs_coeffs[i][ik] 
    #         Nik = bfs_prim_norms[i][ik]
    #         alphaik = bfs_expnts[i][ik]
    #         result[:,i] = result[:,i] + evalGTONumba(alphaik, Ni*Nik*dik, coordi, lmni, coord)

    #New try to make the loop over coords as the innermost
    #UPDATE: Somehow this is slower than the existing implementation
    #This is strange and interesting as the similar thing greatly accelerated rho calcualiton
    #but it slows down ao calculation.
    #Loop over BFs
    # for i in range(nao):
    #     coordi = bfs_coords[i]
    #     Ni = bfs_contr_prim_norms[i]
    #     lmni = bfs_lmn[i]
    #     #Loop over primitives
    #     for ik in range(bfs_nprim[i]):
    #         dik = bfs_coeffs[i][ik] 
    #         Nik = bfs_prim_norms[i][ik]
    #         alphaik = bfs_expnts[i][ik]
    #         #Loop over grid points
    #         for k in prange(ncoord):
    #             result[k,i] += evalGTONumba(alphaik, Ni*Nik*dik, coordi, lmni, coord[k])



    return result

def evalBFsSparseNumba1(bfs_coords, bfs_contr_prim_norms, bfs_nprim, bfs_lmn, bfs_coeffs, bfs_prim_norms, bfs_expnts, bfs_radius_cutoff, coord, bf_indices):
    #This function evaluates the value of all the given Basis functions on the grid (coord).
    # 'coord' should be a nx3 array
        
    nao = bf_indices.shape[0]
    ncoord = coord.shape[0]
    result = np.zeros((ncoord, nao))

    #TODO I guess a better way would be to iterate over BFs first, instead of coords.
    #Also, it would be better to skip the loop over coords altoghether and use numpy instead for the evaluation, by just passing
    #the coords array to evalGTO. I am pretty sure this would increase the speed substantially. This could be wrong though,
    #since we are in fact using JIT so maybe no more optimization is possible. But worth a try.

    #Loop over grid points
    for k in prange(ncoord):
        coord_grid = coord[k]
        #Loop over BFs
        for i in range(nao):
            # Actual index in original basis set
            ibf = bf_indices[i]
            value = 0.0
            coord_bf = bfs_coords[ibf]
            x = coord_grid[0]-coord_bf[0]
            y = coord_grid[1]-coord_bf[1]
            z = coord_grid[2]-coord_bf[2]
            Ni = bfs_contr_prim_norms[ibf]
            lmni = bfs_lmn[ibf]
            for ik in range(bfs_nprim[ibf]):
                dik = bfs_coeffs[ibf][ik] 
                Nik = bfs_prim_norms[ibf][ik]
                alphaik = bfs_expnts[ibf][ik]
                value += evalGTONumba(alphaik, Ni*Nik*dik, lmni, x, y, z)
            result[k,i] = value

    return result

def nonZeroBFIndicesNumba1_old(coords, bf_values, threshold):
    nbfs = bf_values.shape[1]
    ncoords = coords.shape[0]
    count = 0
    indices = np.zeros((nbfs), dtype='uint16')
    # Loop over the basis functions 
    for ibf in range(nbfs):
        # Loop over the grid points and check if the value of the basis function is greater than the threshold
        for igrd in range(ncoords):
            if np.abs(bf_values[igrd, ibf]) > threshold:
                indices[count] = ibf
                count = count + 1
                break

    # Return the indices array and the number of non-zero bfs
    return indices, count

def nonZeroBFIndicesNumba1(coords, bfs_coords, bfs_radius_cutoff):
    nbfs = bfs_coords.shape[0]
    ncoords = coords.shape[0]
    count = 0
    indices = np.zeros((nbfs), dtype='uint16')
    # Loop over the basis functions 
    for ibf in range(nbfs):
        coord_bf = bfs_coords[ibf]
        # Loop over the grid points and check if the value of the basis function is greater than the threshold
        for igrd in range(ncoords):
            coord_grid = coords[igrd]
            x = coord_grid[0]-coord_bf[0]
            y = coord_grid[1]-coord_bf[1]
            z = coord_grid[2]-coord_bf[2]
            if (np.sqrt(x**2+y**2+z**2)<bfs_radius_cutoff[ibf]):
                indices[count] = ibf
                count = count + 1
                break

    # Return the indices array and the number of non-zero bfs
    return indices, count
    

#@njit(parallel=True,cache=True)
def calcZ1(F, ao_values):
    mN = F.shape[0]
    iN = ao_values.shape[1]
    z = np.zeros((mN, iN))
    #v = np.zeros((iN, iN))
    #Loop over BFs
    for i in prange(iN):
        for m in prange(mN):
            z[m,i] = 0.5*F[m]*ao_values[m,i]
    return z
            
#@njit(parallel=True,cache=True)
def overlapMatNumba1(bfs_coords, bfs_contr_prim_norms, bfs_lmn, bfs_nprim, bfs_coeffs, bfs_prim_norms, bfs_expnts,a,b,c,d):
    #This function calculates th overlap matrix for a given slice without any use of symmetry.
    #NOTE: Somehow this is faster than OS scheme when using NUMBA.
    #The reason we need this extra function is because we want the callable function to be simple and not require so many 
    #arguments. But when using Numba to optimize, we can't have too many custom objects and stuff. Numba likes numpy arrays
    # so passing those is okay. But lists and custom objects are not okay.
    m = b-a 
    n = d-c
    S = np.zeros((m,n)) #The difference in syntax is due to Numba
    for i in prange(a,b):
        I = bfs_coords[i]
        lmni = bfs_lmn[i]
        Ni = bfs_contr_prim_norms[i]
        for j in prange(c,d): #Symmetry won't be used and the complete slice would be evaluated.
            S[i,j] = 0.0
            
            J = bfs_coords[j]
            IJ = I - J  
            tempfac = np.sum(IJ**2)
            
            Nj = bfs_contr_prim_norms[j]
            
            lmnj = bfs_lmn[j]
            for ik in prange(bfs_nprim[i]):
                alphaik = bfs_expnts[i][ik]
                dik = bfs_coeffs[i][ik]
                Nik = bfs_prim_norms[i][ik]
                for jk in prange(bfs_nprim[j]):
                    
                    alphajk = bfs_expnts[j][jk]
                    gamma = alphaik + alphajk
                    screenfactor = np.exp(-alphaik*alphajk/gamma*tempfac)
                    if (abs(screenfactor)<1.0e-12):
                        #print('skipped')
                        continue
                    
                    djk = bfs_coeffs[j][jk] 
                    Njk = bfs_prim_norms[j][jk]
                    
                    P = (alphaik*I + alphajk*J)/gamma
                    PI = P - I
                    PJ = P - J
                    Sx = calcSNumba(lmni[0],lmnj[0],gamma,PI[0],PJ[0])
                    Sy = calcSNumba(lmni[1],lmnj[1],gamma,PI[1],PJ[1])
                    Sz = calcSNumba(lmni[2],lmnj[2],gamma,PI[2],PJ[2])
                    temp = dik*djk
                    temp = temp*Nik*Njk
                    temp = temp*Ni*Nj
                    temp = temp*screenfactor*Sx*Sy*Sz
                    #temp = temp*np.exp(-alphaik*alphajk/gamma*np.sum(IJ**2))*Sx*Sy*Sz
                    #I have a hunch that np.sum can be slower when called form NUMBA
                    #UPDATE: Although, the difference wasn't much, it seems that np.sum was faster.
                    #temp = temp*np.exp(-alphaik*alphajk/gamma*(IJ[0]**2+IJ[1]**2+IJ[2]**2))*Sx*Sy*Sz
                    S[i,j] = S[i,j] + temp
    return S


#@njit(parallel=True,cache=True)
def overlapMatSymmNumba1(bfs_coords, bfs_contr_prim_norms, bfs_lmn, bfs_nprim, bfs_coeffs, bfs_prim_norms, bfs_expnts,a,b,c,d):
    #NOTE: Somehow this is faster than OS scheme when using NUMBA.
    # This function calculates the overlap matrix and uses the symmetry property to only calculate half-ish the elements
    # and get the remaining half by symmetry.
    #The reason we need this extra function is because we want the callable function to be simple and not require so many 
    #arguments. But when using Numba to optimize, we can't have too many custom objects and stuff. Numba likes numpy arrays
    # so passing those is okay. But lists and custom objects are not okay.
    m = b-a  #I feel like slices are meaningless for symmetric matrix calculation. As symmetry is only defined for the complete matrix.
    n = d-c
    S = np.zeros((m,n)) #The difference in syntax is due to Numba
    for i in prange(a,b):
        I = bfs_coords[i]
        lmni = bfs_lmn[i]
        Ni = bfs_contr_prim_norms[i]
        for j in prange(c,i+1): #Because we are only evaluating the lower triangular matrix.
            S[i,j] = 0.0
            
            J = bfs_coords[j]
            IJ = I - J  
            tempfac = np.sum(IJ**2)
            
            Nj = bfs_contr_prim_norms[j]
            
            lmnj = bfs_lmn[j]
            for ik in prange(bfs_nprim[i]):
                alphaik = bfs_expnts[i][ik]
                dik = bfs_coeffs[i][ik]
                Nik = bfs_prim_norms[i][ik]
                for jk in prange(bfs_nprim[j]):
                    
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
                    Sx = calcSNumba(lmni[0],lmnj[0],gamma,PI[0],PJ[0])
                    Sy = calcSNumba(lmni[1],lmnj[1],gamma,PI[1],PJ[1])
                    Sz = calcSNumba(lmni[2],lmnj[2],gamma,PI[2],PJ[2])
                    temp = dik*djk
                    temp = temp*Nik*Njk
                    temp = temp*Ni*Nj
                    temp = temp*screenfactor*Sx*Sy*Sz
                    #temp = temp*np.exp(-alphaik*alphajk/gamma*np.sum(IJ**2))*Sx*Sy*Sz
                    #I have a hunch that np.sum can be slower when called form NUMBA
                    #UPDATE: Although, the difference wasn't much, it seems that np.sum was faster.
                    #temp = temp*np.exp(-alphaik*alphajk/gamma*(IJ[0]**2+IJ[1]**2+IJ[2]**2))*Sx*Sy*Sz
                    S[i,j] = S[i,j] + temp
            S[j,i] = S[i,j] #We save time by evaluating only the lower diagonal elements and then use symmetry Si,j=Sj,i
    return S

#@njit(parallel=True,cache=True)
def overlapMatOSNumba1(bfs_coords, bfs_contr_prim_norms, bfs_lmn, bfs_nprim, bfs_coeffs, bfs_prim_norms, bfs_expnts,a,b,c,d):
    #NOTE: For some reason the OS implementation is slower then the regular explicit calculation.
    #The reason we need this extra function is because we want the callable function to be simple and not require so many 
    #arguments. But when using Numba to optimize, we can't have too many custom objects and stuff. Numba likes numpy arrays
    # so passing those is okay. But lists and custom objects are not okay.
    m = b-a
    n = d-c
    S = np.zeros((m,n)) #The difference in syntax is due to Numba
    for i in prange(a,b):
        for j in prange(c,i+1): #Because we are only evaluating the lower triangular matrix.
            S[i,j] = 0.0
            I = bfs_coords[i]
            J = bfs_coords[j]
            IJ = I - J  
            IJsq = IJ**2
            Ni = bfs_contr_prim_norms[i]
            Nj = bfs_contr_prim_norms[j]
            lmni = bfs_lmn[i]
            lmnj = bfs_lmn[j]
            for ik in prange(bfs_nprim[i]):
                alphaik = bfs_expnts[i][ik]
                dik = bfs_coeffs[i][ik]
                Nik = bfs_prim_norms[i][ik]
                for jk in prange(bfs_nprim[j]):
                    
                    alphajk = bfs_expnts[j][jk]
                    gamma = alphaik + alphajk
                    mu = alphaik*alphajk/gamma
                    Kab = np.exp(-mu*IJsq)
                    screenFactor = np.exp(-mu*np.sum(IJsq))
                    if (abs(screenFactor)<1.0e-12):
                        #print('skipped')
                        continue
                    
                    djk = bfs_coeffs[j][jk] 
                    Njk = bfs_prim_norms[j][jk]
                    
                    P = (alphaik*I + alphajk*J)/gamma
                    PI = P - I
                    PJ = P - J

                    # The follwoing are calculated using recursive funcitons technique
                    # Sx = overlapPrimitivesOSNumba2(lmni[0],lmnj[0],gamma,PI[0],PJ[0],Kab[0])
                    # Sy = overlapPrimitivesOSNumba2(lmni[1],lmnj[1],gamma,PI[1],PJ[1],Kab[1])
                    # Sz = overlapPrimitivesOSNumba2(lmni[2],lmnj[2],gamma,PI[2],PJ[2],Kab[2])

                    #The follwoing are calculated using OS scheme but using loops instead of recursion
                    Sx = overlapPrimitivesOS_2Numba2(lmni[0],lmnj[0],gamma,PI[0],PJ[0],Kab[0])
                    Sy = overlapPrimitivesOS_2Numba2(lmni[1],lmnj[1],gamma,PI[1],PJ[1],Kab[1])
                    Sz = overlapPrimitivesOS_2Numba2(lmni[2],lmnj[2],gamma,PI[2],PJ[2],Kab[2])

                    temp = dik*djk
                    temp = temp*Nik*Njk
                    temp = temp*Ni*Nj
                    factor = np.sqrt(np.pi/gamma)
                    temp = temp*Sx*Sy*Sz*Kab[0]*Kab[1]*Kab[2]*factor*factor*factor
        
                    S[i,j] = S[i,j] + temp
            S[j,i] = S[i,j] #We save time by evaluating only the lower diagonal elements and then use symmetry Si,j=Sj,i
    return S
    
#@njit(parallel=True,cache=True)
def overlapPrimitivesOSNumba1(i, j, p, PA, PB, Kab):
    #Calculates and returns the overlap integral of two Gaussian primitives using OBARA-SAIKA scheme.
    #This could have been used in the overlapMat function, but wasn't.
    #But instead it would probably be used to evaluate the kinMat (KE mat).

    if i==0 and j==0:
        return 1  #Kab*np.sqrt(np.pi/p)
    elif i==1 and j==0:
        return PA*overlapPrimitivesOSNumba2(0,0,p,PA,PB,Kab)
    elif j==0:
        return PA*overlapPrimitivesOSNumba2(i-1,0,p,PA,PB,Kab) + 0.5/p*((i-1)*overlapPrimitivesOSNumba2(i-2,0,p,PA,PB,Kab))
    elif i==0 and j==1:
        return PB*overlapPrimitivesOSNumba2(0,0,p,PA,PB,Kab)
    elif i==0:
        return PB*overlapPrimitivesOSNumba2(0,j-1,p,PA,PB,Kab) + 0.5/p*((j-1)*overlapPrimitivesOSNumba2(0,j-2,p,PA,PB,Kab))
    elif i==1 and j==1: 
        return PA*overlapPrimitivesOSNumba2(0,1,p,PA,PB,Kab) + 0.5/p*(overlapPrimitivesOSNumba2(0,0,p,PA,PB,Kab))
    elif i==1:
        return PB*overlapPrimitivesOSNumba2(1,j-1,p,PA,PB,Kab) + 0.5/p*(overlapPrimitivesOSNumba2(0,j-1,p,PA,PB,Kab) + (j-1)*overlapPrimitivesOSNumba2(1,j-2,p,PA,PB,Kab))
    elif j==1:
        return PA*overlapPrimitivesOSNumba2(i-1,1,p,PA,PB,Kab) + 0.5/p*((i-1)*overlapPrimitivesOSNumba2(i-2,1,p,PA,PB,Kab) + overlapPrimitivesOSNumba2(i-1,0,p,PA,PB,Kab))
    # elif i==2 and j==2:
    #     return PA*overlapPrimitivesOSNumba2(1,2,p,PA,PB,Kab) + 0.5/p*(overlapPrimitivesOSNumba2(0,2,p,PA,PB,Kab) + 2*overlapPrimitivesOSNumba2(1,1,p,PA,PB,Kab))
    # elif j==2:
    #     return PA*overlapPrimitivesOSNumba2(i-1,2,p,PA,PB,Kab) + 0.5/p*((i-1)*overlapPrimitivesOSNumba2(i-2,2,p,PA,PB,Kab) + 2*overlapPrimitivesOSNumba2(i-1,1,p,PA,PB,Kab))
    # elif i==2:
    #     return PB*overlapPrimitivesOSNumba2(2,j-1,p,PA,PB,Kab) + 0.5/p*(2*overlapPrimitivesOSNumba2(1,j-1,p,PA,PB,Kab) + (j-1)*overlapPrimitivesOSNumba2(2,j-2,p,PA,PB,Kab))
    else:
        return PA*overlapPrimitivesOSNumba2(i-1,j,p,PA,PB,Kab) + 0.5/p*((i-1)*overlapPrimitivesOSNumba2(i-2,j,p,PA,PB,Kab) + j*overlapPrimitivesOSNumba2(i-1,j-1,p,PA,PB,Kab))

def overlapPrimitivesOS_2Numba1(i, j, p, PA, PB, Kab):
    #Calculates and returns the overlap integral of two Gaussian primitives using OBARA-SAIKA scheme.
    #This function uses loops instead fo recursion of function.
    #This could have been used in the overlapMat function, but wasn't.
    #But instead it would probably be used to evaluate the kinMat (KE mat).

    S = np.empty((i+3,j+3))
    S[0,:] = 0.0
    S[:,0] = 0.0
    S[1,1] = 1.0 #Kab*np.sqrt(np.pi/p)
    for ii in range(1,i+2):
        for jj in range(1,j+2):
            temp = 0.5/p*((ii-1)*S[ii-1,jj] + (jj-1)*S[ii,jj-1])
            S[ii+1,jj] = PA*S[ii,jj] + temp
            S[ii,jj+1] = PB*S[ii,jj] + temp

        
    # Sij = S[i+1,j+1]
    # return Sij
    return S[i+1,j+1]

#@njit(parallel=True,cache=True)
def overlapPrimitives1(alphaA, coordA, lmnA, alphaB, coordB, lmnB, AB, PA, PB, gamma):
    #Calculates and returns the overlap integral of two Gaussian primitives.
    #This could have been used in the overlapMat function, but wasn't.
    #But instead it would probably be used to evaluate the kinMat (KE mat).
        
    #AB = coordA - coordB
    #gamma = alphaA + alphaB
    #P = (alphaA*coordA + alphaB*coordB)/gamma
    #PA = P - coordA
    #PB = P - coordB
    Sx = calcSNumba(lmnA[0],lmnB[0],gamma,PA[0],PB[0])
    Sy = calcSNumba(lmnA[1],lmnB[1],gamma,PA[1],PB[1])
    Sz = calcSNumba(lmnA[2],lmnB[2],gamma,PA[2],PB[2])
    out = np.exp(-alphaA*alphaB/gamma*np.sum(AB**2))*Sx*Sy*Sz

    return out

#@njit(parallel=True,cache=True)
def kinMatNumba1(bfs_coords, bfs_contr_prim_norms, bfs_lmn, bfs_nprim, bfs_coeffs, bfs_prim_norms, bfs_expnts,a,b,c,d):
    #This function calculates th Kinetic energy matrix for a given slice without any use of symmetry.
    #The reason we need this extra function is because we want the callable function to be simple and not require so many 
    #arguments. But when using Numba to optimize, we can't have too many custom objects and stuff. Numba likes numpy arrays
    # so passing those is okay. But lists and custom objects are not okay.
    # This function calculates the kinetic matrix for a given basis object.
    # The basis object holds the information of basis functions like: exponents, coeffs, etc.
    # It is possible to only calculate a slice (block) of the complete matrix.
    # slice is a 4 element list whose first and second elements give the range of the rows to be calculated.
    # the third and fourth element give the range of columns to be calculated.
    # The integrals are performed using the formulas here https://pubs.acs.org/doi/full/10.1021/acs.jchemed.8b00255

    #If the user doesn't provide a slice then calculate the complete kinetic matrix for all the BFs
    m = b-a
    n = d-c
    T = np.zeros((m,n)) #The difference in syntax is due to Numba
    for i in prange(a,b):
        for j in prange(c,d): #Because we are only evaluating the lower triangular matrix.
            T[i,j]=0
            I = bfs_coords[i]
            J = bfs_coords[j]
            IJ = I - J  
            Ni = bfs_contr_prim_norms[i]
            Nj = bfs_contr_prim_norms[j]
            lmni = bfs_lmn[i]
            lmnj = bfs_lmn[j]
            #Some factors to save FLOPS
            fac1 = np.sum(IJ**2)
            fac2 = Ni*Nj
            fac3 = (2*(lmnj[0]+lmnj[1]+lmnj[2])+3)
            fac4 = (lmnj[0]*(lmnj[0]-1))
            fac5 = (lmnj[1]*(lmnj[1]-1))
            fac6 = (lmnj[2]*(lmnj[2]-1))
            for ik in prange(bfs_nprim[i]):
                for jk in prange(bfs_nprim[j]):
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

                    

                    Sx = calcSNumba(lmni[0],lmnj[0],gamma,PI[0],PJ[0])
                    Sy = calcSNumba(lmni[1],lmnj[1],gamma,PI[1],PJ[1])
                    Sz = calcSNumba(lmni[2],lmnj[2],gamma,PI[2],PJ[2])
                    overlap1 = Sx*Sy*Sz

                    Sx = calcSNumba(lmni[0],lmnj[0]+2,gamma,PI[0],PJ[0])
                    #Sy = calcSNumba(lmni[1],lmnj[1],gamma,PI[1],PJ[1])
                    #Sz = calcSNumba(lmni[2],lmnj[2],gamma,PI[2],PJ[2])
                    overlap2 = Sx*Sy*Sz

                    Sx = calcSNumba(lmni[0],lmnj[0],gamma,PI[0],PJ[0])
                    Sy = calcSNumba(lmni[1],lmnj[1]+2,gamma,PI[1],PJ[1])
                    #Sz = calcSNumba(lmni[2],lmnj[2],gamma,PI[2],PJ[2])
                    overlap3 = Sx*Sy*Sz

                    #Sx = calcSNumba(lmni[0],lmnj[0],gamma,PI[0],PJ[0])
                    Sy = calcSNumba(lmni[1],lmnj[1],gamma,PI[1],PJ[1])
                    Sz = calcSNumba(lmni[2],lmnj[2]+2,gamma,PI[2],PJ[2])
                    overlap4 = Sx*Sy*Sz

                    Sx = calcSNumba(lmni[0],lmnj[0]-2,gamma,PI[0],PJ[0])
                    #Sy = calcSNumba(lmni[1],lmnj[1],gamma,PI[1],PJ[1])
                    Sz = calcSNumba(lmni[2],lmnj[2],gamma,PI[2],PJ[2])
                    overlap5 = Sx*Sy*Sz

                    Sx = calcSNumba(lmni[0],lmnj[0],gamma,PI[0],PJ[0])
                    Sy = calcSNumba(lmni[1],lmnj[1]-2,gamma,PI[1],PJ[1])
                    #Sz = calcSNumba(lmni[2],lmnj[2],gamma,PI[2],PJ[2])
                    overlap6 = Sx*Sy*Sz

                    #Sx = calcSNumba(lmni[0],lmnj[0],gamma,PI[0],PJ[0])
                    Sy = calcSNumba(lmni[1],lmnj[1],gamma,PI[1],PJ[1])
                    Sz = calcSNumba(lmni[2],lmnj[2]-2,gamma,PI[2],PJ[2])
                    overlap7 = Sx*Sy*Sz
                    

                    # overlap1 = overlapPrimitives(alphaik, I, lmni, alphajk, J, lmnj, IJ, PI, PJ, gamma)
                    # overlap2 = overlapPrimitives(alphaik, I, lmni, alphajk, J, [lmnj[0]+2,lmnj[1],lmnj[2]], IJ, PI, PJ, gamma)
                    # overlap3 = overlapPrimitives(alphaik, I, lmni, alphajk, J, [lmnj[0],lmnj[1]+2,lmnj[2]], IJ, PI, PJ, gamma)
                    # overlap4 = overlapPrimitives(alphaik, I, lmni, alphajk, J, [lmnj[0],lmnj[1],lmnj[2]+2], IJ, PI, PJ, gamma)
                    # overlap5 = overlapPrimitives(alphaik, I, lmni, alphajk, J, [lmnj[0]-2,lmnj[1],lmnj[2]], IJ, PI, PJ, gamma)
                    # overlap6 = overlapPrimitives(alphaik, I, lmni, alphajk, J, [lmnj[0],lmnj[1]-2,lmnj[2]], IJ, PI, PJ, gamma)
                    # overlap7 = overlapPrimitives(alphaik, I, lmni, alphajk, J, [lmnj[0],lmnj[1],lmnj[2]-2], IJ, PI, PJ, gamma)

                    part1 = overlap1*alphajk*fac3
                    part2 = 2*alphajk*alphajk*(overlap2+overlap3+overlap4)
                    part3 = fac4*overlap5
                    part4 = fac5*overlap6
                    part5 = fac6*overlap7

                    result = temp*(part1 - part2 - 0.5*(part3+part4+part5))*temp_1
                    
                    T[i,j] = T[i,j] + result
    return T

#@njit(parallel=True,cache=True)
def kinMatSymmNumba1(bfs_coords, bfs_contr_prim_norms, bfs_lmn, bfs_nprim, bfs_coeffs, bfs_prim_norms, bfs_expnts,a,b,c,d):
    # This function calculates the kinetic energy matrix and uses the symmetry property to only calculate half-ish the elements
    # and get the remaining half by symmetry.
    #The reason we need this extra function is because we want the callable function to be simple and not require so many 
    #arguments. But when using Numba to optimize, we can't have too many custom objects and stuff. Numba likes numpy arrays
    # so passing those is okay. But lists and custom objects are not okay.
    # This function calculates the kinetic matrix for a given basis object.
    # The basis object holds the information of basis functions like: exponents, coeffs, etc.
    # It is possible to only calculate a slice (block) of the complete matrix.
    # slice is a 4 element list whose first and second elements give the range of the rows to be calculated.
    # the third and fourth element give the range of columns to be calculated.
    # The integrals are performed using the formulas here https://pubs.acs.org/doi/full/10.1021/acs.jchemed.8b00255

    #If the user doesn't provide a slice then calculate the complete kinetic matrix for all the BFs
    m = b-a
    n = d-c
    T = np.zeros((m,n)) #The difference in syntax is due to Numba
    for i in prange(a,b):
        for j in prange(c,i+1): #Because we are only evaluating the lower triangular matrix.
            T[i,j]=0
            I = bfs_coords[i]
            J = bfs_coords[j]
            IJ = I - J  
            Ni = bfs_contr_prim_norms[i]
            Nj = bfs_contr_prim_norms[j]
            lmni = bfs_lmn[i]
            lmnj = bfs_lmn[j]
            #Some factors to save FLOPS
            fac1 = np.sum(IJ**2)
            fac2 = Ni*Nj
            fac3 = (2*(lmnj[0]+lmnj[1]+lmnj[2])+3)
            fac4 = (lmnj[0]*(lmnj[0]-1))
            fac5 = (lmnj[1]*(lmnj[1]-1))
            fac6 = (lmnj[2]*(lmnj[2]-1))
            for ik in prange(bfs_nprim[i]):
                for jk in prange(bfs_nprim[j]):
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

                    

                    Sx = calcSNumba(lmni[0],lmnj[0],gamma,PI[0],PJ[0])
                    Sy = calcSNumba(lmni[1],lmnj[1],gamma,PI[1],PJ[1])
                    Sz = calcSNumba(lmni[2],lmnj[2],gamma,PI[2],PJ[2])
                    overlap1 = Sx*Sy*Sz

                    Sx = calcSNumba(lmni[0],lmnj[0]+2,gamma,PI[0],PJ[0])
                    #Sy = calcSNumba(lmni[1],lmnj[1],gamma,PI[1],PJ[1])
                    #Sz = calcSNumba(lmni[2],lmnj[2],gamma,PI[2],PJ[2])
                    overlap2 = Sx*Sy*Sz

                    Sx = calcSNumba(lmni[0],lmnj[0],gamma,PI[0],PJ[0])
                    Sy = calcSNumba(lmni[1],lmnj[1]+2,gamma,PI[1],PJ[1])
                    #Sz = calcSNumba(lmni[2],lmnj[2],gamma,PI[2],PJ[2])
                    overlap3 = Sx*Sy*Sz

                    #Sx = calcSNumba(lmni[0],lmnj[0],gamma,PI[0],PJ[0])
                    Sy = calcSNumba(lmni[1],lmnj[1],gamma,PI[1],PJ[1])
                    Sz = calcSNumba(lmni[2],lmnj[2]+2,gamma,PI[2],PJ[2])
                    overlap4 = Sx*Sy*Sz

                    Sx = calcSNumba(lmni[0],lmnj[0]-2,gamma,PI[0],PJ[0])
                    #Sy = calcSNumba(lmni[1],lmnj[1],gamma,PI[1],PJ[1])
                    Sz = calcSNumba(lmni[2],lmnj[2],gamma,PI[2],PJ[2])
                    overlap5 = Sx*Sy*Sz

                    Sx = calcSNumba(lmni[0],lmnj[0],gamma,PI[0],PJ[0])
                    Sy = calcSNumba(lmni[1],lmnj[1]-2,gamma,PI[1],PJ[1])
                    #Sz = calcSNumba(lmni[2],lmnj[2],gamma,PI[2],PJ[2])
                    overlap6 = Sx*Sy*Sz

                    #Sx = calcSNumba(lmni[0],lmnj[0],gamma,PI[0],PJ[0])
                    Sy = calcSNumba(lmni[1],lmnj[1],gamma,PI[1],PJ[1])
                    Sz = calcSNumba(lmni[2],lmnj[2]-2,gamma,PI[2],PJ[2])
                    overlap7 = Sx*Sy*Sz
                    

                    # overlap1 = overlapPrimitives(alphaik, I, lmni, alphajk, J, lmnj, IJ, PI, PJ, gamma)
                    # overlap2 = overlapPrimitives(alphaik, I, lmni, alphajk, J, [lmnj[0]+2,lmnj[1],lmnj[2]], IJ, PI, PJ, gamma)
                    # overlap3 = overlapPrimitives(alphaik, I, lmni, alphajk, J, [lmnj[0],lmnj[1]+2,lmnj[2]], IJ, PI, PJ, gamma)
                    # overlap4 = overlapPrimitives(alphaik, I, lmni, alphajk, J, [lmnj[0],lmnj[1],lmnj[2]+2], IJ, PI, PJ, gamma)
                    # overlap5 = overlapPrimitives(alphaik, I, lmni, alphajk, J, [lmnj[0]-2,lmnj[1],lmnj[2]], IJ, PI, PJ, gamma)
                    # overlap6 = overlapPrimitives(alphaik, I, lmni, alphajk, J, [lmnj[0],lmnj[1]-2,lmnj[2]], IJ, PI, PJ, gamma)
                    # overlap7 = overlapPrimitives(alphaik, I, lmni, alphajk, J, [lmnj[0],lmnj[1],lmnj[2]-2], IJ, PI, PJ, gamma)

                    part1 = overlap1*alphajk*fac3
                    part2 = 2*alphajk*alphajk*(overlap2+overlap3+overlap4)
                    part3 = fac4*overlap5
                    part4 = fac5*overlap6
                    part5 = fac6*overlap7

                    result = temp*(part1 - part2 - 0.5*(part3+part4+part5))*temp_1
                    
                    T[i,j] = T[i,j] + result
            T[j,i] = T[i,j] #We save time by evaluating only the lower diagonal elements and then use symmetry Si,j=Sj,i
    return T

#@njit(parallel=True,cache=True)
# def vlriNumba1(la,lb,Ai,Bi,Ci,gamma,l,r,i):
#     epsilon = 1/(4*gamma)
#     return (-1)**l*c2kNumba(l,la,lb,Ai,Bi)*((-1)**i*fac(l)*Ci**(l-2*r-2*i)*epsilon**(r+i)/(fac(r)*fac(i)*fac(l-2*r-2*i)))

#@njit(parallel=True,cache=True)
def vlriNumbaPartial1(Ci, l,r,i):
    # return (-1)**l*((-1)**i*fac(l)*Ci**(l-2*r-2*i)/(fac(r)*fac(i)*fac(l-2*r-2*i)))
    return (-1)**l*((-1)**i*fastFactorial2(l)*Ci**(l-2*r-2*i)/(fastFactorial2(r)*fastFactorial2(i)*fastFactorial2(l-2*r-2*i)))

#@njit(parallel=True,cache=True)
def calcCgammincNumba1(a):
    p = [9.4368392235e-3,-1.0782666481E-04,-5.8969657295E-06,2.8939523781E-07,1.0043326298E-01,5.5637848465E-01]
    q = [1.1464706419E-01,2.6963429121E+00,-2.9647038257E+00,2.1080724954E+00]
    r = [0.0,1.1428716184E+00,-6.6981186438E-03,1.0480765092E-04]
    s = [1.0356711153E+00,2.3423452308E+00,-3.6174503174E-01,-3.1376557650E+00,2.9092306039E+00]
    a2 = a*a
    a3 = a*a*a
    a4 = a*a*a*a
    c = np.empty(4)
    c[0] = 1 + p[0]*a + p[1]*a2 + p[2]*a3 + p[3]*a4 + p[4]*(np.exp(-p[5]*a)-1)
    c[1] = q[0] + q[1]/a + q[2]/(a2) + q[3]/a3
    c[2] = r[0] + r[1]*a + r[2]*a2 + r[3]*a3
    c[3] = s[0] + s[1]/a + s[2]/a2 + s[3]/a3 + s[4]/a4
    return c

#@njit(parallel=True,cache=True)
def incGammaNumba1(a,x):
    out = np.exp(-x)*x**a
    c = calcCgammincNumba(a)
    out = out*(1/a+c[0]*x/(a*(a+1))+(c[0]*x)*(c[0]*x)/(a*(a+1)*(a+2)))
    Wx = 0.5 + 0.5*math.tanh(c[1]*(x-c[2]))
    #out = out*(1-Wx) + gammaNumba(a)*Wx*(1-c4**(-x))
    out = out*(1-Wx) + gammaNumba(a)*Wx*(1-c[3]**(-x))
    return out

#@njit(parallel=True,cache=True)
def gammaNumba1(x): 
#For numerical integration in the calculaiton of gamma function
    # _a =( 1.00000000000000000000, 0.57721566490153286061, -0.65587807152025388108,
    #      -0.04200263503409523553, 0.16653861138229148950, -0.04219773455554433675,
    #      -0.00962197152787697356, 0.00721894324666309954, -0.00116516759185906511,
    #      -0.00021524167411495097, 0.00012805028238811619, -0.00002013485478078824,
    #      -0.00000125049348214267, 0.00000113302723198170, -0.00000020563384169776,
    #       0.00000000611609510448, 0.00000000500200764447, -0.00000000118127457049,
    #       0.00000000010434267117, 0.00000000000778226344, -0.00000000000369680562,
    #       0.00000000000051003703, -0.00000000000002058326, -0.00000000000000534812,
    #       0.00000000000000122678, -0.00000000000000011813, 0.00000000000000000119,
    #       0.00000000000000000141, -0.00000000000000000023, 0.00000000000000000002
    #    )
    _a_reverse =(-2.3e-19, 1.41e-18, 1.19e-18, -1.1813e-16, 1.22678e-15, -5.34812e-15, -2.058326e-14, 5.1003703e-13, -3.69680562e-12, 7.78226344e-12, 1.0434267117e-10, -1.18127457049e-09, 5.00200764447e-09, 6.11609510448e-09, -2.0563384169776e-07, 1.1330272319817e-06, -1.25049348214267e-06, -2.013485478078824e-05, 0.0001280502823881162, -0.00021524167411495098, -0.0011651675918590652, 0.0072189432466631, -0.009621971527876973, -0.04219773455554433, 0.16653861138229148, -0.04200263503409524, -0.6558780715202539, 0.5772156649015329, 1.0)
    y  = x - 1.0
    sm = _a_reverse[0]
    n = len(_a_reverse)
    for i in range(n):
        sm = sm * y + _a_reverse[i]
    return 1.0 / sm

#F3 in boys_test.py ()
#@njit(cache=True)
def FboysNumba1_old(v,x):
    # Return wrong values at T very close at zero, why u is defined to be nearly 0
    if x >= 0 and x < 0.0000001:
        F = 1/(2*v+1) #- x/(2*v+3)
    else:
        part1 = np.sqrt(np.pi)/(4**v * x**(v+1/2)) * math.erf(np.sqrt(x))
        part2 = 0 
        for k in range(0, v):
            part2 += fac(v-k)/(4**k * fac(2*v-2*k)*x**(k+1))
        F = fac(2*v)/(2*fac(v)) * (part1 - np.exp(-x)*part2)
    return F

def FboysNumba1_old(v,x):
    if x >= 0 and x < 0.0000001:
        F = 1/(2*v+1) - x/(2*v+3)
    else:
        F = 0.5*x**(-(v+0.5))*incGammaNumba(v+0.5,x)*gammaNumba(v+0.5)
    return F


# F2 in boys_test.py (was supposed to be faster)
#@njit(cache=True)
def FboysNumba1(v,x):
    # From: https://pubs.acs.org/doi/full/10.1021/acs.jchemed.8b00255
    #from scipy.special import gammainc, gamma
    if x >= 0 and x < 0.0000001:
        F = 1/(2*v+1) - x/(2*v+3)
    else:
        F = 0.5*x**(-(v+0.5))*gammainc(v+0.5,x)*gamma(v+0.5)
        # F = 0.5*x**(-(v+0.5))*numba_gamma(v+0.5)#*gammainc(v+0.5,x)
    return F
#@njit(cache=True)
def FboysNumba1_jjgoings(v,x):
    #from scipy.special import hyp1f1
    F = hyp1f1(v+0.5,v+1.5,-x)/(2.0*v+1.0)
    return F

#@njit(parallel=True,cache=True)
def nucMatSymmNumba1(bfs_coords, bfs_contr_prim_norms, bfs_lmn, bfs_nprim, bfs_coeffs, bfs_prim_norms, bfs_expnts,a,b,c,d, Z, coordsMol, natoms):
    # This function calculates the nuclear potential matrix for a given basis object and mol object.
    # This function calculates the nuclear potential matrix and uses the symmetry property to only calculate half-ish the elements
    # and get the remaining half by symmetry.
    # The basis object holds the information of basis functions like: exponents, coeffs, etc.
    # The mol object is used to get information about the charges of nuclei and their positions.
    # It is here, that we see the advantage of having the mol and basis objects be supplied separately.
    # This allows to calculate the nuclear matrix of one molecule in the basis of another.
    # It is possible to only calculate a slice (block) of the complete matrix.
    # slice is a 4 element list whose first and second elements give the range of the rows to be calculated.
    # the third and fourth element give the range of columns to be calculated.
    # The integrals are performed using the formulas https://pubs.acs.org/doi/full/10.1021/acs.jchemed.8b00255

    # (A|-Z_C/r_{iC}|B) = 
    #NOTE: This is more slower than the calculation of overlap and kinetic matrices 
    #when compared with PySCF.
    #Benchmark: Cholestrol.xyz def2-QZVPPD (3963 BFs) 766 sec vs 30 sec of PySCF
    #Using numba-scipy allows us to use scipy.special.gamma and gamminc,
    #however, this prevents the caching functionality. Nevertheless,
    #apart form the compilation overhead, it allows to perform calculaitons significantly faster and with good 
    #accuracy.
    #Benchmark: Cholestrol.xyz def2-QZVPPD (3963 BFs) 760 sec vs 30 sec of PySCF using scipy gamma and 10^-8 screening

    #If the user doesn't provide a slice then calculate the complete kinetic matrix for all the BFs
    m = b-a
    n = d-c
    V = np.zeros((m,n)) #The difference in syntax is due to Numba
    PI = 3.141592653589793
    PIx2 = 6.283185307179586 #2*PI
    
    
    #Loop pver BFs
    for i in prange(a,b):
        I = bfs_coords[i]
        Ni = bfs_contr_prim_norms[i]
        lmni = bfs_lmn[i]
        la, ma, na = lmni
        for j in prange(c,i+1):
            
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
                    PI = P - I
                    PJ = P - J
                    tempfac = (PIx2*gamma_inv)*screenfactor

                    Vc = 0.0
                    #Loop over nuclei
                    for iatom in prange(natoms): #Parallelising over atoms seems to be faster for Cholestrol.xyz with def2-QZVPPD (628 sec)
                        Rc = coordsMol[iatom]
                        Zc = Z[iatom]
                        PC = P - Rc

                        fac1 = -Zc*tempfac
                        #print(fac1)
                        sum_Vl = 0.0
                        
                        
                        for l in range(0,la+lb+1):
                            facl = c2kNumba(l,la,lb,PI[0],PJ[0])
                            for r in range(0, int(l/2)+1):
                                for i1 in range(0, int((l-2*r)/2)+1):
                                    v_lri = vlriNumbaPartial(PC[0],l,r,i1)*epsilon**(r+i1)*facl
                                    sum_Vm = 0.0
                                    for m in range(0,ma+mb+1):
                                        facm = c2kNumba(m,ma,mb,PI[1],PJ[1])
                                        for s in range(0, int(m/2)+1):
                                            for j1 in range(0, int((m-2*s)/2)+1):
                                                v_msj = vlriNumbaPartial(PC[1],m,s,j1)*epsilon**(s+j1)*facm
                                                sum_Vn = 0.0
                                                for n in range(0,na+nb+1):
                                                    facn = c2kNumba(n,na,nb,PI[2],PJ[2])
                                                    for t in range(0, int(n/2)+1):
                                                        for k in range(0, int((n-2*t)/2)+1):
                                                            v_ntk = vlriNumbaPartial(PC[2],n,t,k)*epsilon**(t+k)*facn
                                                            # This version of Boys function is 3.1 times slower with 842 BFs (sto-6g & Icosahectane_C120H242)
                                                            # F = FboysNumba2_jjgoings(l+m+n-2*(r+s+t)-(i1+j1+k),gamma*np.sum(PC**2))
                                                            # Seems to befficient for now (F2 in boys_test.py)
                                                            F = FboysNumba2(l+m+n-2*(r+s+t)-(i1+j1+k),gamma*np.sum(PC**2))  
                                                            #TODO The numba implementation of Fboys gives wrong answers.
                                                            #Needs to fix that. 
                                                            
                                                            sum_Vn += v_ntk*F
                                                sum_Vm += v_msj*sum_Vn
                                    sum_Vl += v_lri*sum_Vm
                            
                        Vc += sum_Vl*fac1
                    #print(Vc)
                    V[i,j] += Vc*dik*djk*Nik*Njk*Ni*Nj
                    #print(dik*djk*Nik*Njk*Ni*Nj*Vc)       
                    #print(i,j)                 
            V[j,i] = V[i,j] #We save time by evaluating only the lower diagonal elements and then use symmetry Vi,j=Vj,i                
    
    return V

#@njit(parallel=True,cache=True)
def nucMatNumba1(bfs_coords, bfs_contr_prim_norms, bfs_lmn, bfs_nprim, bfs_coeffs, bfs_prim_norms, bfs_expnts,a,b,c,d, Z, coordsMol, natoms):
    # This function calculates the nuclear potential matrix for a given basis object and mol object.
    # The basis object holds the information of basis functions like: exponents, coeffs, etc.
    # The mol object is used to get information about the charges of nuclei and their positions.
    # It is here, that we see the advantage of having the mol and basis objects be supplied separately.
    # This allows to calculate the nuclear matrix of one molecule in the basis of another.
    # It is possible to only calculate a slice (block) of the complete matrix.
    # slice is a 4 element list whose first and second elements give the range of the rows to be calculated.
    # the third and fourth element give the range of columns to be calculated.
    # The integrals are performed using the formulas https://pubs.acs.org/doi/full/10.1021/acs.jchemed.8b00255

    # (A|-Z_C/r_{iC}|B) = 
    #NOTE: This is more slower than the calculation of overlap and kinetic matrices 
    #when compared with PySCF.
    #Benchmark: Cholestrol.xyz def2-QZVPPD (3963 BFs) 766 sec vs 30 sec of PySCF
    #Using numba-scipy allows us to use scipy.special.gamma and gamminc,
    #however, this prevents the caching functionality. Nevertheless,
    #apart form the compilation overhead, it allows to perform calculaitons significantly faster and with good 
    #accuracy.
    #Benchmark: Cholestrol.xyz def2-QZVPPD (3963 BFs) 760 sec vs 30 sec of PySCF using scipy gamma and 10^-8 screening
    #Update: After parallelising over the atoms, now NUMBA is much better in performance.

    #If the user doesn't provide a slice then calculate the complete nuclear matrix for all the BFs
    m = b-a
    n = d-c
    V = np.zeros((m,n)) #The difference in syntax is due to Numba
    PI = 3.141592653589793
    PIx2 = 6.283185307179586 #2*PI
    
    
    #Loop pver BFs
    for i in prange(a,b):
        I = bfs_coords[i]
        Ni = bfs_contr_prim_norms[i]
        lmni = bfs_lmn[i]
        la, ma, na = lmni
        for j in prange(c,d):
            
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
                    screenfactor = np.exp(-alphaik*alphajk*gamma_inv*IJsq)
                    if abs(screenfactor)<1.0e-8:   
                    #TODO: This is quite low. But since this is the slowest part.
                    #But I had to do this because this is a very slow part of the program.
                    #Will have to check how the accuracy is affected and if the screening factor
                    #can be reduced further.
                        continue

                    
                    djk = bfs_coeffs[j,jk] 
                    
                    Njk = bfs_prim_norms[j,jk]
                    
                    epsilon = 0.25*gamma_inv
                    P = (alphaik*I + alphajk*J)*gamma_inv
                    PI = P - I
                    PJ = P - J
                    tempfac = (PIx2*gamma_inv)*screenfactor

                    Vc = 0.0
                    #Loop over nuclei
                    for iatom in prange(natoms): #Parallelising over atoms seems to be faster for Cholestrol.xyz with def2-QZVPPD (628 sec)
                        Rc = coordsMol[iatom]
                        Zc = Z[iatom]
                        PC = P - Rc

                        fac1 = -Zc*tempfac
                        #print(fac1)
                        sum_Vl = 0.0
                        # vMax = la+lb+ma+mb+na+nb
                        # Fmax = FboysNumba2(vMax,gamma*np.sum(PC**2))
                        # Fboys_dict = dict()
                        
                        
                        for l in range(0,la+lb+1):
                            facl = c2kNumba(l,la,lb,PI[0],PJ[0])
                            for r in range(0, int(l/2)+1):
                                for i1 in range(0, int((l-2*r)/2)+1):
                                    v_lri = vlriNumbaPartial(PC[0],l,r,i1)*epsilon**(r+i1)*facl
                                    sum_Vm = 0.0
                                    for m in range(0,ma+mb+1):
                                        facm = c2kNumba(m,ma,mb,PI[1],PJ[1])
                                        for s in range(0, int(m/2)+1):
                                            for j1 in range(0, int((m-2*s)/2)+1):
                                                v_msj = vlriNumbaPartial(PC[1],m,s,j1)*epsilon**(s+j1)*facm
                                                sum_Vn = 0.0
                                                for n in range(0,na+nb+1):
                                                    facn = c2kNumba(n,na,nb,PI[2],PJ[2])
                                                    for t in range(0, int(n/2)+1):
                                                        for k in range(0, int((n-2*t)/2)+1):
                                                            v_ntk = vlriNumbaPartial(PC[2],n,t,k)*epsilon**(t+k)*facn
                                                            # This version of Boys function is 3.1 times slower with 842 BFs (sto-6g & Icosahectane_C120H242)
                                                            # F = FboysNumba2_jjgoings(l+m+n-2*(r+s+t)-(i1+j1+k),gamma*np.sum(PC**2))
                                                            # Seems to befficient for now (F2 in boys_test.py)
                                                            F = FboysNumba2(l+m+n-2*(r+s+t)-(i1+j1+k),gamma*np.sum(PC**2))  
                                                            # Using dictionary for Fboys
                                                            # if l+m+n-2*(r+s+t)-(i1+j1+k) in Fboys_dict:
                                                            #     F = Fboys_dict[l+m+n-2*(r+s+t)-(i1+j1+k)]
                                                            # else:
                                                            #     Fboys_dict[l+m+n-2*(r+s+t)-(i1+j1+k)] = F = FboysNumba2(l+m+n-2*(r+s+t)-(i1+j1+k),gamma*np.sum(PC**2)) 
                                                            # Using recursion for Fboys
                                                            # vvv = vMax 
                                                            # F=1
                                                            # while not vvv == (l+m+n-2*(r+s+t)-(i1+j1+k)):
                                                            #     F = FboysRecursiveNumba2(vvv, gamma*np.sum(PC**2), Fmax)
                                                            #     vvv -= 1
                                                            
                                                            sum_Vn += v_ntk*F
                                                sum_Vm += v_msj*sum_Vn
                                    sum_Vl += v_lri*sum_Vm
                            
                        Vc += sum_Vl*fac1
                    #print(Vc)
                    V[i,j] += Vc*dik*djk*Nik*Njk*Ni*Nj
                                   
    
    return V

def thetaNumba1(l,la,lb,PA,PB,gamma_,r):
    return c2kNumba(l,la,lb,PA,PB)*fastFactorial2(l)*(gamma_**(r-l))/(fastFactorial2(r)*fastFactorial2(l-2*r))

def gNumba1(lp,lq,rp,rq,i,la,lb,lc,ld,gammaP,gammaQ,PA,PB,QC,QD,PQ,delta):
        temp = ((-1)**lp)*thetaNumba2(lp,la,lb,PA,PB,gammaP,rp)*thetaNumba2(lq,lc,ld,QC,QD,gammaQ,rq)
        numerator = temp*((-1)**i)*((2*delta)**(2*(rp+rq)))*fastFactorial2(lp+lq-2*rp-2*rq)*(delta**i)*(PQ**(lp+lq-2*(rp+rq+i)))
        denominator = ((4*delta)**(lp+lq))*fastFactorial2(i)*fastFactorial2(lp+lq-2*(rp+rq+i))
        # print(numerator/temp)
        return (numerator/denominator)

def innerLoop4c2eNumba1(la,lb,lc,ld,ma,mb,mc,md,na,nb,nc,nd,gammaP,gammaQ,PI,PJ,QK,QL,PQ,PQsqBy4delta,delta):
    sum1 = 0.0
    for lp in range(0,la+lb+1):
        for rp in range(0, int(lp/2)+1):
            for lq in range(0, lc+ld+1):
                for rq in range(0, int(lq/2)+1):
                    for i1 in range(0,int((lp+lq-2*rp-2*rq)/2)+1):
                        gx = gNumba2(lp,lq,rp,rq,i1,la,lb,lc,ld,gammaP,gammaQ,PI[0],PJ[0],QK[0],QL[0],PQ[0],delta)
                        sum2 = 0.0
                        for mp in range(0,ma+mb+1):
                            for sp in range(0, int(mp/2)+1):
                                for mq in range(0, mc+md+1):
                                    for sq in range(0, int(mq/2)+1):
                                        for j1 in range(0,int((mp+mq-2*sp-2*sq)/2)+1):
                                            gy = gNumba2(mp,mq,sp,sq,j1,ma,mb,mc,md,gammaP,gammaQ,PI[1],PJ[1],QK[1],QL[1],PQ[1],delta)
                                            sum3 = 0.0                                                                   
                                            for np1 in range(0,na+nb+1):
                                                for tp in range(0, int(np1/2)+1):
                                                    for nq in range(0, nc+nd+1):
                                                        for tq in range(0, int(nq/2)+1):
                                                            for k1 in range(0,int((np1+nq-2*tp-2*tq)/2)+1):
                                                                gz = gNumba2(np1,nq,tp,tq,k1,na,nb,nc,nd,gammaP,gammaQ,PI[2],PJ[2],QK[2],QL[2],PQ[2],delta)
                                                                v = lp+lq+mp+mq+np1+nq-2*(rp+rq+sp+sq+tp+tq)-(i1+j1+k1)
                                                                F = FboysNumba2(v,PQsqBy4delta)
                                                                # F = FboysNumba2(v,np.sum(PQ**2)/(4*delta))
                                                                # F = FboysNumba1_jjgoings(v,PQsqBy4delta)
                                                                sum3 = sum3 + gz*F                                                                                   
                                            sum2 = sum2 + gy*sum3
                        sum1 = sum1 + gx*sum2
    return sum1



def fourCenterTwoElecNumba1(bfs_coords, bfs_contr_prim_norms, bfs_lmn, bfs_nprim, bfs_coeffs, bfs_prim_norms, bfs_expnts,indx_startA,indx_endA,indx_startB,indx_endB,indx_startC,indx_endC,indx_startD,indx_endD):
        # This function calculates the electron-electron Coulomb potential matrix for a given basis object and mol object.
        # The basis object holds the information of basis functions like: exponents, coeffs, etc.
        
        # It is possible to only calculate a slice (block) of the complete matrix.
        # slice is a 4 element list whose first and second elements give the range of the rows to be calculated.
        # the third and fourth element give the range of columns to be calculated.
        # The integrals are performed using the formulas https://pubs.acs.org/doi/full/10.1021/acs.jchemed.8b00255
        # Some useful resources:
        # https://chemistry.stackexchange.com/questions/71527/how-does-one-compute-the-number-of-unique-2-electron-integrals-for-a-given-basis
        # https://chemistry.stackexchange.com/questions/82532/practical-differences-between-storing-2-electron-integrals-and-calculating-them
        # Density matrix based screening Section (5.1.2): https://www.diva-portal.org/smash/get/diva2:740301/FULLTEXT02.pdf
        # Lots of notes on screening and efficiencies (3.3 pg 31): https://www.zora.uzh.ch/id/eprint/44716/1/Diss.123456.pdf
        # In the above resource in section 3.4 it also says that storing in 64 bit floating point precision is usually not required. 
        # So, the storage requirements can be reduced by 4 to 8 times.

       

    # returns (AB|CD) 
    
    m = indx_endA - indx_startA
    n = indx_endB - indx_startB
    o = indx_endC - indx_startC
    p = indx_endD - indx_startD
    fourC2E = np.zeros((m,n,o,p)) #The difference in syntax is due to Numba
        
       
        
    pi = 3.141592653589793
    pisq = 9.869604401089358  #PI^2
    twopisq = 19.739208802178716  #2*PI^2

    #Loop pver BFs
    for i in prange(indx_startA, indx_endA): #A
        I = bfs_coords[i]
        Ni = bfs_contr_prim_norms[i]
        lmni = bfs_lmn[i]
        la, ma, na = lmni
        nprimi = bfs_nprim[i]
        
        for j in prange(indx_startB, indx_endB): #B
            J = bfs_coords[j]
            IJ = I - J
            IJsq = np.sum(IJ**2)
            Nj = bfs_contr_prim_norms[j]
            lmnj = bfs_lmn[j]
            lb, mb, nb = lmnj
            tempcoeff1 = Ni*Nj
            nprimj = bfs_nprim[j]
            
            for k in prange(indx_startC, indx_endC): #C
                K = bfs_coords[k]
                Nk = bfs_contr_prim_norms[k]
                lmnk = bfs_lmn[k]
                lc, mc, nc = lmnk
                tempcoeff2 = tempcoeff1*Nk
                nprimk = bfs_nprim[k]
                
                for l in prange(indx_startD, indx_endD): #D
                    L = bfs_coords[l]
                    KL = K - L  
                    KLsq = np.sum(KL**2)
                    Nl = bfs_contr_prim_norms[l]
                    lmnl = bfs_lmn[l]  
                    ld, md, nd = lmnl
                    tempcoeff3 = tempcoeff2*Nl
                    npriml = bfs_nprim[l]
                    
                    #Loop over primitives
                    for ik in range(bfs_nprim[i]):   
                        dik = bfs_coeffs[i][ik]
                        Nik = bfs_prim_norms[i][ik]
                        alphaik = bfs_expnts[i][ik]
                        tempcoeff4 = tempcoeff3*dik*Nik
                        
                        for jk in range(bfs_nprim[j]):
                            alphajk = bfs_expnts[j][jk]
                            gammaP = alphaik + alphajk
                            screenfactorAB = np.exp(-alphaik*alphajk/gammaP*IJsq)
                            if abs(screenfactorAB)<1.0e-8:   
                                #TODO: Check for optimal value for screening
                                continue
                            djk = bfs_coeffs[j][jk] 
                            Njk = bfs_prim_norms[j][jk]      
                            P = (alphaik*I + alphajk*J)/gammaP
                            PI = P - I
                            PJ = P - J  
                            fac1 = twopisq/gammaP*screenfactorAB   
                            onefourthgammaPinv = 0.25/gammaP  
                            tempcoeff5 = tempcoeff4*djk*Njk  
                            
                            for kk in range(bfs_nprim[k]):
                                dkk = bfs_coeffs[k][kk]
                                Nkk = bfs_prim_norms[k][kk]
                                alphakk = bfs_expnts[k][kk]
                                tempcoeff6 = tempcoeff5*dkk*Nkk 
                                  
                                for lk in range(bfs_nprim[l]): 
                                    alphalk = bfs_expnts[l][lk]
                                    gammaQ = alphakk + alphalk
                                    screenfactorKL = np.exp(-alphakk*alphalk/gammaQ*KLsq)
                                    if abs(screenfactorKL)<1.0e-8:   
                                        #TODO: Check for optimal value for screening
                                        continue
                                    if abs(screenfactorAB*screenfactorKL)<1.0e-12:   
                                        #TODO: Check for optimal value for screening
                                        continue
                                    dlk = bfs_coeffs[l][lk] 
                                    Nlk = bfs_prim_norms[l][lk]     
                                    Q = (alphakk*K + alphalk*L)/gammaQ        
                                    PQ = P - Q
                                    
                                    QK = Q - K
                                    QL = Q - L
                                    tempcoeff7 = tempcoeff6*dlk*Nlk
                                      
                                    fac2 = fac1/gammaQ
                                    

                                    omega = (fac2)*np.sqrt(pi/(gammaP + gammaQ))*screenfactorKL
                                    delta = 0.25*(1/gammaQ) + onefourthgammaPinv          
                                    PQsqBy4delta = np.sum(PQ**2)/(4*delta)         
                                        
                                    sum1 = innerLoop4c2eNumba2(la,lb,lc,ld,ma,mb,mc,md,na,nb,nc,nd,gammaP,gammaQ,PI,PJ,QK,QL,PQ,PQsqBy4delta,delta)
                                    # sum1 = FboysNumba2(la+lb+lc+ld+ma+mb+mc+md+na+nb+nc+nd,PQsqBy4delta)
                                    # sum1 = 1.0
                                        
                                    fourC2E[i,j,k,l] = fourC2E[i,j,k,l] + omega*sum1*tempcoeff7
                                    
                                   
                            
        
    return fourC2E

def rys2c2eSymmNumba1(bfs_coords, bfs_contr_prim_norms, bfs_lmn, bfs_nprim, bfs_coeffs, bfs_prim_norms, bfs_expnts,a,b,c,d):
        # Two centered two electron integrals by hacking the 4c2e routines based on rys quadrature.
        # Based on Rys Quadrature from https://github.com/rpmuller/MolecularIntegrals.jl
        # This function calculates the electron-electron Coulomb potential matrix for a given basis object and mol object.
        # This uses 8 fold symmetry to only calculate the unique elements and assign the rest via symmetry
        # The basis object holds the information of basis functions like: exponents, coeffs, etc.
        
        # The integrals are performed using the formulas https://pubs.acs.org/doi/full/10.1021/acs.jchemed.8b00255
        # Some useful resources:
        # https://chemistry.stackexchange.com/questions/71527/how-does-one-compute-the-number-of-unique-2-electron-integrals-for-a-given-basis
        # https://chemistry.stackexchange.com/questions/82532/practical-differences-between-storing-2-electron-integrals-and-calculating-them
        # Density matrix based screening Section (5.1.2): https://www.diva-portal.org/smash/get/diva2:740301/FULLTEXT02.pdf
        # Lots of notes on screening and efficiencies (3.3 pg 31): https://www.zora.uzh.ch/id/eprint/44716/1/Diss.123456.pdf
        # In the above resource in section 3.4 it also says that storing in 64 bit floating point precision is usually not required. 
        # So, the storage requirements can be reduced by 4 to 8 times.

       

    # returns (A|C) 
    
    m = b-a
    n = d-c
    twoC2E = np.zeros((m,n)) 
    
       
        
    # pi = 3.141592653589793
    # pisq = 9.869604401089358  #PI^2
    # twopisq = 19.739208802178716  #2*PI^2
    J = L = np.zeros((3))
    Nj = Nl = 1
    lnmj = lmnl = np.zeros((3),dtype=np.int32)
    lb, mb, nb = int(0), int(0), int(0)
    ld, md, nd = int(0), int(0), int(0)
    alphajk = alphalk = 0.0
    djk, dlk = 1.0, 1.0
    Njk, Nlk = 1.0, 1.0
    #Loop pver BFs
    for i in prange(a, b): #A
        I = bfs_coords[i]
        # J = I
        # IJ = I #I - J
        P = I
        # IJsq = np.sum(IJ**2)
        Ni = bfs_contr_prim_norms[i]
        lmni = bfs_lmn[i]
        la, ma, na = lmni
        # nprimi = bfs_nprim[i]    
        
        for k in range(c, i+1): #C
            K = bfs_coords[k]
            # L = K
            # KL = K
            Q = K
            # KLsq = np.sum(KL**2)
            Nk = bfs_contr_prim_norms[k]
            lmnk = bfs_lmn[k]
            lc, mc, nc = lmnk
            tempcoeff1 = Ni*Nk
            # nprimk = bfs_nprim[k]
            

            norder = int((la+ma+na+lc+mc+nc)/2+1 ) 
            n = int(max(la,ma,na))
            m = int(max(lc,mc,nc))
            roots = np.zeros((norder))
            weights = np.zeros((norder))
            G = np.zeros((n+1,m+1))
                    
            PQ = P - Q
            PQsq = np.sum(PQ**2)
            
                    
            #Loop over primitives
            for ik in range(bfs_nprim[i]):   
                dik = bfs_coeffs[i][ik]
                Nik = bfs_prim_norms[i][ik]
                alphaik = bfs_expnts[i][ik]
                alphajk = 0.0
                tempcoeff2 = tempcoeff1*dik*Nik
                gammaP = alphaik
                        
                for kk in range(bfs_nprim[k]):
                    dkk = bfs_coeffs[k][kk]
                    Nkk = bfs_prim_norms[k][kk]
                    alphakk = bfs_expnts[k][kk]
                    alphalk = 0.0
                    tempcoeff3 = tempcoeff2*dkk*Nkk 
                    gammaQ = alphakk
                                  
                                     
                                    
                    rho = gammaP*gammaQ/(gammaP+gammaQ)
                    
                                    
                                      
                    twoC2E[i,k] += tempcoeff3*coulomb_rysNumba2(roots,weights,G,PQsq, rho, norder,n,m,la,lb,lc,ld,ma,mb,mc,md,na,nb,nc,nd,alphaik, alphajk, alphakk, alphalk,I,J,K,L)
                                    

                    
            twoC2E[k,i] = twoC2E[i,k]
                    
                                    
                                   
                            
        
    return twoC2E

def rys3c2eSymmNumba_tri1(bfs_coords, bfs_contr_prim_norms, bfs_lmn, bfs_nprim, bfs_coeffs, bfs_prim_norms, bfs_expnts,aux_bfs_coords, aux_bfs_contr_prim_norms, aux_bfs_lmn, aux_bfs_nprim, aux_bfs_coeffs, aux_bfs_prim_norms, aux_bfs_expnts,indx_startA,indx_endA,indx_startB,indx_endB,indx_startC,indx_endC):
        # Based on Rys Quadrature from https://github.com/rpmuller/MolecularIntegrals.jl
        # This function calculates the electron-electron Coulomb potential matrix for a given basis object and mol object.
        # This uses 8 fold symmetry to only calculate the unique elements and assign the rest via symmetry
        # The basis object holds the information of basis functions like: exponents, coeffs, etc.
        
        # The integrals are performed using the formulas https://pubs.acs.org/doi/full/10.1021/acs.jchemed.8b00255
        # Some useful resources:
        # https://chemistry.stackexchange.com/questions/71527/how-does-one-compute-the-number-of-unique-2-electron-integrals-for-a-given-basis
        # https://chemistry.stackexchange.com/questions/82532/practical-differences-between-storing-2-electron-integrals-and-calculating-them
        # Density matrix based screening Section (5.1.2): https://www.diva-portal.org/smash/get/diva2:740301/FULLTEXT02.pdf
        # Lots of notes on screening and efficiencies (3.3 pg 31): https://www.zora.uzh.ch/id/eprint/44716/1/Diss.123456.pdf
        # In the above resource in section 3.4 it also says that storing in 64 bit floating point precision is usually not required. 
        # So, the storage requirements can be reduced by 4 to 8 times.

       

    
    m = indx_endA - indx_startA
    n = indx_endB - indx_startB
    o = indx_endC - indx_startC
    
    threeC2E = np.zeros((int(m*(n+1)/2.0),o),dtype=np.float64) 
   
    
       
        
    pi = 3.141592653589793
    # pisq = 9.869604401089358  #PI^2
    twopisq = 19.739208802178716  #2*PI^2
    L = np.zeros((3))
    alphalk = 0.0
    ld, md, nd = int(0), int(0), int(0)
    #Loop over BFs
    for i in range(0, indx_endA): #A
        offset = int(i*(i+1)/2)
        I = bfs_coords[i]
        Ni = bfs_contr_prim_norms[i]
        lmni = bfs_lmn[i]
        la, ma, na = lmni
        # nprimi = bfs_nprim[i]
        
        for j in range(0, i+1): #B
            J = bfs_coords[j]
            IJ = I - J
            IJsq = np.sum(IJ**2)
            Nj = bfs_contr_prim_norms[j]
            lmnj = bfs_lmn[j]
            lb, mb, nb = lmnj
            tempcoeff1 = Ni*Nj
            # nprimj = bfs_nprim[j]
            # if i<j:
            #     triangle2ij = (j)*(j+1)/2+i
            # else:
            #     triangle2ij = (i)*(i+1)/2+j
            
            for k in prange(0, indx_endC): #C
                K = aux_bfs_coords[k]
                Nk = aux_bfs_contr_prim_norms[k]
                lmnk = aux_bfs_lmn[k]
                lc, mc, nc = lmnk
                tempcoeff2 = tempcoeff1*Nk
                # nprimk = bfs_nprim[k]
                
               
                  
                KL = K - L  
                KLsq = np.sum(KL**2)
                
                # npriml = bfs_nprim[l]

                norder = int((la+ma+na+lb+mb+nb+lc+mc+nc+ld+md+nd)/2 + 1 ) 
                val = 0.0
                if norder<=5: # Use rys quadrature
                    n = int(max(la+lb,ma+mb,na+nb))
                    m = int(max(lc+ld,mc+md,nc+nd))
                    roots = np.zeros((norder))
                    weights = np.zeros((norder))
                    G = np.zeros((n+1,m+1))
                        

                    
                        
                    #Loop over primitives
                    for ik in range(bfs_nprim[i]):   
                        dik = bfs_coeffs[i][ik]
                        Nik = bfs_prim_norms[i][ik]
                        alphaik = bfs_expnts[i][ik]
                        tempcoeff3 = tempcoeff2*dik*Nik
                            
                        for jk in range(bfs_nprim[j]):
                            alphajk = bfs_expnts[j][jk]
                            gammaP = alphaik + alphajk
                            screenfactorAB = np.exp(-alphaik*alphajk/gammaP*IJsq)
                            if abs(screenfactorAB)<1.0e-8:   
                                #Although this value of screening threshold seems very large
                                # it actually gives the best consistency with PySCF. Reducing it to 1e-15,
                                # actually worsened the agreement.
                                # I suspect that this is caused due to an error cancellation
                                # that happens with the nucmat calculation, as the same screening is 
                                # used there as well
                                continue
                            djk = bfs_coeffs[j][jk] 
                            Njk = bfs_prim_norms[j][jk]      
                            P = (alphaik*I + alphajk*J)/gammaP
                            # PI = P - I
                            # PJ = P - J  
                            # fac1 = twopisq/gammaP*screenfactorAB   
                            # onefourthgammaPinv = 0.25/gammaP  
                            tempcoeff4 = tempcoeff3*djk*Njk  
                                
                            for kk in range(aux_bfs_nprim[k]):
                                dkk = aux_bfs_coeffs[k][kk]
                                Nkk = aux_bfs_prim_norms[k][kk]
                                alphakk = aux_bfs_expnts[k][kk]
                                tempcoeff5 = tempcoeff4*dkk*Nkk 
                                      
                                    
                                gammaQ = alphakk #+ alphalk
                                screenfactorKL = np.exp(-alphakk*alphalk/gammaQ*KLsq)
                                if abs(screenfactorKL)<1.0e-8:   
                                    #Although this value of screening threshold seems very large
                                    # it actually gives the best consistency with PySCF. Reducing it to 1e-15,
                                    # actually worsened the agreement.
                                    # I suspect that this is caused due to an error cancellation
                                    # that happens with the nucmat calculation, as the same screening is 
                                    # used there as well
                                    continue
                                # if abs(screenfactorAB*screenfactorKL)<1.0e-10:   
                                #     #TODO: Check for optimal value for screening
                                #     continue
                                  
                                Q = K#(alphakk*K + alphalk*L)/gammaQ        
                                PQ = P - Q
                                PQsq = np.sum(PQ**2)
                                rho = gammaP*gammaQ/(gammaP+gammaQ)
                                        
                                        
                                # QK = Q - K
                                # QL = Q - L
                                        
                                        
                                          
                                val += tempcoeff5*coulomb_rysNumba2(roots,weights,G,PQsq, rho, norder,n,m,la,lb,lc,ld,ma,mb,mc,md,na,nb,nc,nd,alphaik, alphajk, alphakk, alphalk,I,J,K,L)
                                    
                else: # Analytical (Conventional)
                    #Loop over primitives
                    for ik in range(bfs_nprim[i]):   
                        dik = bfs_coeffs[i][ik]
                        Nik = bfs_prim_norms[i][ik]
                        alphaik = bfs_expnts[i][ik]
                        tempcoeff3 = tempcoeff2*dik*Nik
                            
                        for jk in range(bfs_nprim[j]):
                            alphajk = bfs_expnts[j][jk]
                            gammaP = alphaik + alphajk
                            screenfactorAB = np.exp(-alphaik*alphajk/gammaP*IJsq)
                            if abs(screenfactorAB)<1.0e-8:   
                                #Although this value of screening threshold seems very large
                                # it actually gives the best consistency with PySCF. Reducing it to 1e-15,
                                # actually worsened the agreement.
                                # I suspect that this is caused due to an error cancellation
                                # that happens with the nucmat calculation, as the same screening is 
                                # used there as well
                                continue
                            djk = bfs_coeffs[j][jk] 
                            Njk = bfs_prim_norms[j][jk]      
                            P = (alphaik*I + alphajk*J)/gammaP
                            PI = P - I
                            PJ = P - J  
                            fac1 = twopisq/gammaP*screenfactorAB   
                            onefourthgammaPinv = 0.25/gammaP  
                            tempcoeff4 = tempcoeff3*djk*Njk  
                                
                            for kk in range(aux_bfs_nprim[k]):
                                dkk = aux_bfs_coeffs[k][kk]
                                Nkk = aux_bfs_prim_norms[k][kk]
                                alphakk = aux_bfs_expnts[k][kk]
                                tempcoeff5 = tempcoeff4*dkk*Nkk 
                                      
                                
                                gammaQ = alphakk #+ alphalk
                                screenfactorKL = np.exp(-alphakk*alphalk/gammaQ*KLsq)
                                if abs(screenfactorKL)<1.0e-8:   
                                    #Although this value of screening threshold seems very large
                                    # it actually gives the best consistency with PySCF. Reducing it to 1e-15,
                                    # actually worsened the agreement.
                                    # I suspect that this is caused due to an error cancellation
                                    # that happens with the nucmat calculation, as the same screening is 
                                    # used there as well
                                    continue
                                if abs(screenfactorAB*screenfactorKL)<1.0e-10:   
                                    #TODO: Check for optimal value for screening
                                    continue
                                
                                Q = K#(alphakk*K + alphalk*L)/gammaQ        
                                PQ = P - Q
                                        
                                QK = Q - K
                                QL = Q #- L
                                
                                          
                                fac2 = fac1/gammaQ
                                        

                                omega = (fac2)*np.sqrt(pi/(gammaP + gammaQ))#*screenfactorKL
                                delta = 0.25*(1/gammaQ) + onefourthgammaPinv          
                                PQsqBy4delta = np.sum(PQ**2)/(4*delta)         
                                            
                                sum1 = innerLoop4c2eNumba2(la,lb,lc,ld,ma,mb,mc,md,na,nb,nc,nd,gammaP,gammaQ,PI,PJ,QK,QL,PQ,PQsqBy4delta,delta)
                                # sum1 = FboysNumba2(la+lb+lc+ld+ma+mb+mc+md+na+nb+nc+nd,PQsqBy4delta)
                                # sum1 = 1.0
                                            
                                val += omega*sum1*tempcoeff5

                threeC2E[j+offset,k] = val
                                    
                                   
                            
        
    return threeC2E

def rys3c2eSymmNumba_tri_schwarz1(bfs_coords, bfs_contr_prim_norms, bfs_lmn, bfs_nprim, bfs_coeffs, bfs_prim_norms, bfs_expnts,aux_bfs_coords, aux_bfs_contr_prim_norms, aux_bfs_lmn, aux_bfs_nprim, aux_bfs_coeffs, aux_bfs_prim_norms, aux_bfs_expnts, indicesA, indicesB, indicesC, nao, naux):
    # Calculates 3c2e integrals based on a provided list of significant triplets determined using Schwarz inequality
    # It is assumed that the provided list was made with triangular int3c2e in mind

    threeC2E = np.zeros((int(nao*(nao+1)/2.0),naux), dtype=np.float64) 

    ntriplets = indicesA.shape[0] # No. of significant triplets
   
        
    pi = 3.141592653589793
    # pisq = 9.869604401089358  #PI^2
    twopisq = 19.739208802178716  #2*PI^2
    L = np.zeros((3))
    alphalk = 0.0
    ld, md, nd = int(0), int(0), int(0)

    # Create arrays to avoid contention of allocator 
    # (https://stackoverflow.com/questions/70339388/using-numba-with-np-concatenate-is-not-efficient-in-parallel/70342014#70342014)
    
    #Loop over BFs
    for itemp in prange(ntriplets): 
        i = indicesA[itemp]
        j = indicesB[itemp]
        k = indicesC[itemp]
        offset = int(i*(i+1)/2)
        I = bfs_coords[i]
        Ni = bfs_contr_prim_norms[i]
        lmni = bfs_lmn[i]
        la, ma, na = lmni

        J = bfs_coords[j]
        IJ = I - J
        IJsq = np.sum(IJ**2)
        Nj = bfs_contr_prim_norms[j]
        lmnj = bfs_lmn[j]
        lb, mb, nb = lmnj
        tempcoeff1 = Ni*Nj
        
        K = aux_bfs_coords[k]
        Nk = aux_bfs_contr_prim_norms[k]
        lmnk = aux_bfs_lmn[k]
        lc, mc, nc = lmnk
        tempcoeff2 = tempcoeff1*Nk
                
               
                  
        KL = K - L
        KLsq = np.sum(KL**2)
                
                # npriml = bfs_nprim[l]

        norder = int((la+ma+na+lb+mb+nb+lc+mc+nc+ld+md+nd)/2 + 1 ) 
        val = 0.0
        if norder<=5: # Use rys quadrature
            n = int(max(la+lb,ma+mb,na+nb))
            m = int(max(lc+ld,mc+md,nc+nd))
            roots = np.zeros((norder))
            weights = np.zeros((norder))
            G = np.zeros((n+1,m+1))
                

            
                
            #Loop over primitives
            for ik in range(bfs_nprim[i]):   
                dik = bfs_coeffs[i,ik]
                Nik = bfs_prim_norms[i,ik]
                alphaik = bfs_expnts[i,ik]
                tempcoeff3 = tempcoeff2*dik*Nik
                    
                for jk in range(bfs_nprim[j]):
                    alphajk = bfs_expnts[j,jk]
                    gammaP = alphaik + alphajk
                    screenfactorAB = np.exp(-alphaik*alphajk/gammaP*IJsq)
                    if abs(screenfactorAB)<1.0e-8:   
                        #Although this value of screening threshold seems very large
                        # it actually gives the best consistency with PySCF. Reducing it to 1e-15,
                        # actually worsened the agreement.
                        # I suspect that this is caused due to an error cancellation
                        # that happens with the nucmat calculation, as the same screening is 
                        # used there as well
                        continue
                    djk = bfs_coeffs[j,jk] 
                    Njk = bfs_prim_norms[j,jk]      
                    P = (alphaik*I + alphajk*J)/gammaP
                    # PI = P - I
                    # PJ = P - J  
                    # fac1 = twopisq/gammaP*screenfactorAB   
                    # onefourthgammaPinv = 0.25/gammaP  
                    tempcoeff4 = tempcoeff3*djk*Njk  
                        
                    for kk in range(aux_bfs_nprim[k]):
                        dkk = aux_bfs_coeffs[k,kk]
                        Nkk = aux_bfs_prim_norms[k,kk]
                        alphakk = aux_bfs_expnts[k,kk]
                        tempcoeff5 = tempcoeff4*dkk*Nkk 
                              
                            
                        gammaQ = alphakk #+ alphalk
                        screenfactorKL = np.exp(-alphakk*alphalk/gammaQ*KLsq)
                        if abs(screenfactorKL)<1.0e-8:   
                            #Although this value of screening threshold seems very large
                            # it actually gives the best consistency with PySCF. Reducing it to 1e-15,
                            # actually worsened the agreement.
                            # I suspect that this is caused due to an error cancellation
                            # that happens with the nucmat calculation, as the same screening is 
                            # used there as well
                            continue
                        # if abs(screenfactorAB*screenfactorKL)<1.0e-10:   
                        #     #TODO: Check for optimal value for screening
                        #     continue
                          
                        Q = K#(alphakk*K + alphalk*L)/gammaQ        
                        PQ = P - Q
                        PQsq = np.sum(PQ**2)
                        rho = gammaP*gammaQ/(gammaP+gammaQ)
                                
                                
                        # QK = Q - K
                        # QL = Q - L
                                
                                
                                  
                        val += tempcoeff5*coulomb_rysNumba2(roots,weights,G,PQsq, rho, norder,n,m,la,lb,lc,ld,ma,mb,mc,md,na,nb,nc,nd,alphaik, alphajk, alphakk, alphalk,I,J,K,L)
                            
        else: # Analytical (Conventional)
            #Loop over primitives
            for ik in range(bfs_nprim[i]):   
                dik = bfs_coeffs[i,ik]
                Nik = bfs_prim_norms[i,ik]
                alphaik = bfs_expnts[i,ik]
                tempcoeff3 = tempcoeff2*dik*Nik
                    
                for jk in range(bfs_nprim[j]):
                    alphajk = bfs_expnts[j,jk]
                    gammaP = alphaik + alphajk
                    screenfactorAB = np.exp(-alphaik*alphajk/gammaP*IJsq)
                    if abs(screenfactorAB)<1.0e-8:   
                        #Although this value of screening threshold seems very large
                        # it actually gives the best consistency with PySCF. Reducing it to 1e-15,
                        # actually worsened the agreement.
                        # I suspect that this is caused due to an error cancellation
                        # that happens with the nucmat calculation, as the same screening is 
                        # used there as well
                        continue
                    djk = bfs_coeffs[j][jk] 
                    Njk = bfs_prim_norms[j][jk]      
                    P = (alphaik*I + alphajk*J)/gammaP
                    PI = P - I
                    PJ = P - J  
                    fac1 = twopisq/gammaP*screenfactorAB   
                    onefourthgammaPinv = 0.25/gammaP  
                    tempcoeff4 = tempcoeff3*djk*Njk  
                        
                    for kk in range(aux_bfs_nprim[k]):
                        dkk = aux_bfs_coeffs[k,kk]
                        Nkk = aux_bfs_prim_norms[k,kk]
                        alphakk = aux_bfs_expnts[k,kk]
                        tempcoeff5 = tempcoeff4*dkk*Nkk 
                              
                        
                        gammaQ = alphakk #+ alphalk
                        screenfactorKL = np.exp(-alphakk*alphalk/gammaQ*KLsq)
                        if abs(screenfactorKL)<1.0e-8:   
                            #Although this value of screening threshold seems very large
                            # it actually gives the best consistency with PySCF. Reducing it to 1e-15,
                            # actually worsened the agreement.
                            # I suspect that this is caused due to an error cancellation
                            # that happens with the nucmat calculation, as the same screening is 
                            # used there as well
                            continue
                        
                        Q = K#(alphakk*K + alphalk*L)/gammaQ        
                        PQ = P - Q
                                
                        QK = Q - K
                        QL = Q #- L
                        
                                  
                        fac2 = fac1/gammaQ
                                

                        omega = (fac2)*np.sqrt(pi/(gammaP + gammaQ))#*screenfactorKL
                        delta = 0.25*(1/gammaQ) + onefourthgammaPinv          
                        PQsqBy4delta = np.sum(PQ**2)/(4*delta)         
                                    
                        sum1 = innerLoop4c2eNumba2(la,lb,lc,ld,ma,mb,mc,md,na,nb,nc,nd,gammaP,gammaQ,PI,PJ,QK,QL,PQ,PQsqBy4delta,delta)
                        # sum1 = FboysNumba2(la+lb+lc+ld+ma+mb+mc+md+na+nb+nc+nd,PQsqBy4delta)
                        # sum1 = 1.0
                                    
                        val += omega*sum1*tempcoeff5

        threeC2E[j+offset,k] = val
                                    
                                   
                            
        
    return threeC2E

def rys3c2eSymmNumba1(bfs_coords, bfs_contr_prim_norms, bfs_lmn, bfs_nprim, bfs_coeffs, bfs_prim_norms, bfs_expnts,aux_bfs_coords, aux_bfs_contr_prim_norms, aux_bfs_lmn, aux_bfs_nprim, aux_bfs_coeffs, aux_bfs_prim_norms, aux_bfs_expnts,indx_startA,indx_endA,indx_startB,indx_endB,indx_startC,indx_endC):
        # Based on Rys Quadrature from https://github.com/rpmuller/MolecularIntegrals.jl
        # This function calculates the electron-electron Coulomb potential matrix for a given basis object and mol object.
        # This uses 8 fold symmetry to only calculate the unique elements and assign the rest via symmetry
        # The basis object holds the information of basis functions like: exponents, coeffs, etc.
        
        # The integrals are performed using the formulas https://pubs.acs.org/doi/full/10.1021/acs.jchemed.8b00255
        # Some useful resources:
        # https://chemistry.stackexchange.com/questions/71527/how-does-one-compute-the-number-of-unique-2-electron-integrals-for-a-given-basis
        # https://chemistry.stackexchange.com/questions/82532/practical-differences-between-storing-2-electron-integrals-and-calculating-them
        # Density matrix based screening Section (5.1.2): https://www.diva-portal.org/smash/get/diva2:740301/FULLTEXT02.pdf
        # Lots of notes on screening and efficiencies (3.3 pg 31): https://www.zora.uzh.ch/id/eprint/44716/1/Diss.123456.pdf
        # In the above resource in section 3.4 it also says that storing in 64 bit floating point precision is usually not required. 
        # So, the storage requirements can be reduced by 4 to 8 times.

       

    
    m = indx_endA - indx_startA
    n = indx_endB - indx_startB
    o = indx_endC - indx_startC
    
    threeC2E = np.zeros((m,n,o),dtype=np.float64) 
   
    
       
        
    pi = 3.141592653589793
    # pisq = 9.869604401089358  #PI^2
    twopisq = 19.739208802178716  #2*PI^2
    L = np.zeros((3))
    alphalk = 0.0
    ld, md, nd = int(0), int(0), int(0)
    #Loop pver BFs
    for i in prange(0, indx_endA): #A
        I = bfs_coords[i]
        Ni = bfs_contr_prim_norms[i]
        lmni = bfs_lmn[i]
        la, ma, na = lmni
        # nprimi = bfs_nprim[i]
        
        for j in prange(0, i+1): #B
            J = bfs_coords[j]
            IJ = I - J
            IJsq = np.sum(IJ**2)
            Nj = bfs_contr_prim_norms[j]
            lmnj = bfs_lmn[j]
            lb, mb, nb = lmnj
            tempcoeff1 = Ni*Nj
            # nprimj = bfs_nprim[j]
            # if i<j:
            #     triangle2ij = (j)*(j+1)/2+i
            # else:
            #     triangle2ij = (i)*(i+1)/2+j
            
            for k in prange(0, indx_endC): #C
                K = aux_bfs_coords[k]
                Nk = aux_bfs_contr_prim_norms[k]
                lmnk = aux_bfs_lmn[k]
                lc, mc, nc = lmnk
                tempcoeff2 = tempcoeff1*Nk
                # nprimk = bfs_nprim[k]
                
               
                  
                KL = K #- L  
                KLsq = np.sum(KL**2)
                
                # npriml = bfs_nprim[l]

                norder = int((la+ma+na+lb+mb+nb+lc+mc+nc+ld+md+nd)/2 + 1 ) 
                val = 0.0
                if norder<=5: # Use rys quadrature
                    n = int(max(la+lb,ma+mb,na+nb))
                    m = int(max(lc+ld,mc+md,nc+nd))
                    roots = np.zeros((norder))
                    weights = np.zeros((norder))
                    G = np.zeros((n+1,m+1))
                        

                    
                        
                    #Loop over primitives
                    for ik in range(bfs_nprim[i]):   
                        dik = bfs_coeffs[i][ik]
                        Nik = bfs_prim_norms[i][ik]
                        alphaik = bfs_expnts[i][ik]
                        tempcoeff3 = tempcoeff2*dik*Nik
                            
                        for jk in range(bfs_nprim[j]):
                            alphajk = bfs_expnts[j][jk]
                            gammaP = alphaik + alphajk
                            screenfactorAB = np.exp(-alphaik*alphajk/gammaP*IJsq)
                            if abs(screenfactorAB)<1.0e-8:   
                                #TODO: Check for optimal value for screening
                                continue
                            djk = bfs_coeffs[j][jk] 
                            Njk = bfs_prim_norms[j][jk]      
                            P = (alphaik*I + alphajk*J)/gammaP
                            # PI = P - I
                            # PJ = P - J  
                            # fac1 = twopisq/gammaP*screenfactorAB   
                            # onefourthgammaPinv = 0.25/gammaP  
                            tempcoeff4 = tempcoeff3*djk*Njk  
                                
                            for kk in range(aux_bfs_nprim[k]):
                                dkk = aux_bfs_coeffs[k][kk]
                                Nkk = aux_bfs_prim_norms[k][kk]
                                alphakk = aux_bfs_expnts[k][kk]
                                tempcoeff5 = tempcoeff4*dkk*Nkk 
                                      
                                    
                                gammaQ = alphakk #+ alphalk
                                screenfactorKL = np.exp(-alphakk*alphalk/gammaQ*KLsq)
                                if abs(screenfactorKL)<1.0e-8:   
                                    #TODO: Check for optimal value for screening
                                    continue
                                if abs(screenfactorAB*screenfactorKL)<1.0e-10:   
                                    #TODO: Check for optimal value for screening
                                    continue
                                  
                                Q = K#(alphakk*K + alphalk*L)/gammaQ        
                                PQ = P - Q
                                PQsq = np.sum(PQ**2)
                                rho = gammaP*gammaQ/(gammaP+gammaQ)
                                        
                                        
                                # QK = Q - K
                                # QL = Q - L
                                        
                                        
                                          
                                val += tempcoeff5*coulomb_rysNumba2(roots,weights,G,PQsq, rho, norder,n,m,la,lb,lc,ld,ma,mb,mc,md,na,nb,nc,nd,alphaik, alphajk, alphakk, alphalk,I,J,K,L)
                                    

                else: # Analytical (Conventional)
                    #Loop over primitives
                    for ik in range(bfs_nprim[i]):   
                        dik = bfs_coeffs[i][ik]
                        Nik = bfs_prim_norms[i][ik]
                        alphaik = bfs_expnts[i][ik]
                        tempcoeff3 = tempcoeff2*dik*Nik
                            
                        for jk in range(bfs_nprim[j]):
                            alphajk = bfs_expnts[j][jk]
                            gammaP = alphaik + alphajk
                            screenfactorAB = np.exp(-alphaik*alphajk/gammaP*IJsq)
                            if abs(screenfactorAB)<1.0e-8:   
                                #TODO: Check for optimal value for screening
                                continue
                            djk = bfs_coeffs[j][jk] 
                            Njk = bfs_prim_norms[j][jk]      
                            P = (alphaik*I + alphajk*J)/gammaP
                            PI = P - I
                            PJ = P - J  
                            fac1 = twopisq/gammaP*screenfactorAB   
                            onefourthgammaPinv = 0.25/gammaP  
                            tempcoeff4 = tempcoeff3*djk*Njk  
                                
                            for kk in range(aux_bfs_nprim[k]):
                                dkk = aux_bfs_coeffs[k][kk]
                                Nkk = aux_bfs_prim_norms[k][kk]
                                alphakk = aux_bfs_expnts[k][kk]
                                tempcoeff5 = tempcoeff4*dkk*Nkk 
                                      
                                
                                gammaQ = alphakk #+ alphalk
                                screenfactorKL = np.exp(-alphakk*alphalk/gammaQ*KLsq)
                                if abs(screenfactorKL)<1.0e-8:   
                                    #TODO: Check for optimal value for screening
                                    continue
                                if abs(screenfactorAB*screenfactorKL)<1.0e-10:   
                                    #TODO: Check for optimal value for screening
                                    continue
                                
                                Q = K#(alphakk*K + alphalk*L)/gammaQ        
                                PQ = P - Q
                                        
                                QK = Q - K
                                QL = Q #- L
                                
                                          
                                fac2 = fac1/gammaQ
                                        

                                omega = (fac2)*np.sqrt(pi/(gammaP + gammaQ))#*screenfactorKL
                                delta = 0.25*(1/gammaQ) + onefourthgammaPinv          
                                PQsqBy4delta = np.sum(PQ**2)/(4*delta)         
                                            
                                sum1 = innerLoop4c2eNumba2(la,lb,lc,ld,ma,mb,mc,md,na,nb,nc,nd,gammaP,gammaQ,PI,PJ,QK,QL,PQ,PQsqBy4delta,delta)
                                # sum1 = FboysNumba2(la+lb+lc+ld+ma+mb+mc+md+na+nb+nc+nd,PQsqBy4delta)
                                # sum1 = 1.0
                                            
                                val += omega*sum1*tempcoeff5

                threeC2E[i,j,k] = val
                threeC2E[j,i,k] = val
                                    
                                   
                            
        
    return threeC2E

def rys4c2eSymmNumba1(bfs_coords, bfs_contr_prim_norms, bfs_lmn, bfs_nprim, bfs_coeffs, bfs_prim_norms, bfs_expnts,indx_startA,indx_endA,indx_startB,indx_endB,indx_startC,indx_endC,indx_startD,indx_endD):
        # Based on Rys Quadrature from https://github.com/rpmuller/MolecularIntegrals.jl
        # This function calculates the electron-electron Coulomb potential matrix for a given basis object and mol object.
        # This uses 8 fold symmetry to only calculate the unique elements and assign the rest via symmetry
        # The basis object holds the information of basis functions like: exponents, coeffs, etc.
        
        # The integrals are performed using the formulas https://pubs.acs.org/doi/full/10.1021/acs.jchemed.8b00255
        # Some useful resources:
        # https://chemistry.stackexchange.com/questions/71527/how-does-one-compute-the-number-of-unique-2-electron-integrals-for-a-given-basis
        # https://chemistry.stackexchange.com/questions/82532/practical-differences-between-storing-2-electron-integrals-and-calculating-them
        # Density matrix based screening Section (5.1.2): https://www.diva-portal.org/smash/get/diva2:740301/FULLTEXT02.pdf
        # Lots of notes on screening and efficiencies (3.3 pg 31): https://www.zora.uzh.ch/id/eprint/44716/1/Diss.123456.pdf
        # In the above resource in section 3.4 it also says that storing in 64 bit floating point precision is usually not required. 
        # So, the storage requirements can be reduced by 4 to 8 times.

       

    # returns (AB|CD) 
    
    m = indx_endA - indx_startA
    n = indx_endB - indx_startB
    o = indx_endC - indx_startC
    p = indx_endD - indx_startD
    fourC2E = np.zeros((m,n,o,p),dtype=np.float64) 
    # print('Four Center Two electron ERI size in GB ',fourC2E.nbytes/1e9)

    sRys = np.ones((12,12))
    
       
        
    # pi = 3.141592653589793
    # pisq = 9.869604401089358  #PI^2
    # twopisq = 19.739208802178716  #2*PI^2

    #Loop pver BFs
    for i in prange(0, indx_endA): #A
        I = bfs_coords[i]
        Ni = bfs_contr_prim_norms[i]
        lmni = bfs_lmn[i]
        la, ma, na = lmni
        # nprimi = bfs_nprim[i]
        
        for j in prange(0, i+1): #B
            J = bfs_coords[j]
            IJ = I - J
            IJsq = np.sum(IJ**2)
            Nj = bfs_contr_prim_norms[j]
            lmnj = bfs_lmn[j]
            lb, mb, nb = lmnj
            tempcoeff1 = Ni*Nj
            # nprimj = bfs_nprim[j]
            if i<j:
                triangle2ij = (j)*(j+1)/2+i
            else:
                triangle2ij = (i)*(i+1)/2+j
            
            for k in prange(0, indx_endC): #C
                K = bfs_coords[k]
                Nk = bfs_contr_prim_norms[k]
                lmnk = bfs_lmn[k]
                lc, mc, nc = lmnk
                tempcoeff2 = tempcoeff1*Nk
                # nprimk = bfs_nprim[k]
                
                for l in prange(0, k+1): #D
                    
                    if k<l:
                        triangle2kl = (l)*(l+1)/2+k
                    else:
                        triangle2kl = (k)*(k+1)/2+l
                    if triangle2ij>triangle2kl:
                        continue
                    L = bfs_coords[l]
                    KL = K - L  
                    KLsq = np.sum(KL**2)
                    Nl = bfs_contr_prim_norms[l]
                    lmnl = bfs_lmn[l]  
                    ld, md, nd = lmnl
                    tempcoeff3 = tempcoeff2*Nl
                    # npriml = bfs_nprim[l]

                    norder = int((la+ma+na+lb+mb+nb+lc+mc+nc+ld+md+nd)/2 + 1 ) 
                    n = int(max(la+lb,ma+mb,na+nb))
                    m = int(max(lc+ld,mc+md,nc+nd))
                    roots = np.zeros((norder))
                    weights = np.zeros((norder))
                    G = np.zeros((n+1,m+1))
                    

                    # val = 0.0
                    
                    #Loop over primitives
                    for ik in range(bfs_nprim[i]):   
                        dik = bfs_coeffs[i][ik]
                        Nik = bfs_prim_norms[i][ik]
                        alphaik = bfs_expnts[i][ik]
                        tempcoeff4 = tempcoeff3*dik*Nik
                        
                        for jk in range(bfs_nprim[j]):
                            alphajk = bfs_expnts[j][jk]
                            gammaP = alphaik + alphajk
                            screenfactorAB = np.exp(-alphaik*alphajk/gammaP*IJsq)
                            if abs(screenfactorAB)<1.0e-8:   
                                #TODO: Check for optimal value for screening
                                continue
                            djk = bfs_coeffs[j][jk] 
                            Njk = bfs_prim_norms[j][jk]      
                            P = (alphaik*I + alphajk*J)/gammaP
                            # PI = P - I
                            # PJ = P - J  
                            # fac1 = twopisq/gammaP*screenfactorAB   
                            # onefourthgammaPinv = 0.25/gammaP  
                            tempcoeff5 = tempcoeff4*djk*Njk  
                            
                            for kk in range(bfs_nprim[k]):
                                dkk = bfs_coeffs[k][kk]
                                Nkk = bfs_prim_norms[k][kk]
                                alphakk = bfs_expnts[k][kk]
                                tempcoeff6 = tempcoeff5*dkk*Nkk 
                                  
                                for lk in range(bfs_nprim[l]): 
                                    alphalk = bfs_expnts[l][lk]
                                    gammaQ = alphakk + alphalk
                                    screenfactorKL = np.exp(-alphakk*alphalk/gammaQ*KLsq)
                                    if abs(screenfactorKL)<1.0e-8:   
                                        #TODO: Check for optimal value for screening
                                        continue
                                    if abs(screenfactorAB*screenfactorKL)<1.0e-10:   
                                        #TODO: Check for optimal value for screening
                                        continue
                                    dlk = bfs_coeffs[l][lk] 
                                    Nlk = bfs_prim_norms[l][lk]     
                                    Q = (alphakk*K + alphalk*L)/gammaQ        
                                    PQ = P - Q
                                    PQsq = np.sum(PQ**2)
                                    rho = gammaP*gammaQ/(gammaP+gammaQ)
                                    
                                    
                                    # QK = Q - K
                                    # QL = Q - L
                                    tempcoeff7 = tempcoeff6*dlk*Nlk
                                    
                                    if norder<6:
                                        fourC2E[i,j,k,l] += tempcoeff7*coulomb_rysNumba2(roots,weights,G,PQsq, rho, norder,n,m,la,lb,lc,ld,ma,mb,mc,md,na,nb,nc,nd,alphaik, alphajk, alphakk, alphalk,I,J,K,L)
                                    else:
                                        T = rho*PQsq
                                        sERI = 34.9868366552497256925 * (screenfactorAB+screenfactorKL) / ((gammaP*gammaQ) * np.sqrt(gammaP + gammaQ))
                                        fourC2E[i,j,k,l] += tempcoeff7*sERI*ChebGausIntNumba2(1E-8,50000, gammaP, gammaQ, la, lb, lc, ld,ma, mb, mc,  md, na, nb, nc,nd, I[0],
                                                                        J[0],  K[0], L[0],  I[1],  J[1],  K[1], L[1], I[2],  J[2],  K[2],
                                                                          L[2],  P[0], P[1], P[2], Q[0], Q[1], Q[2], T, sRys)

                    # fourC2E[i,j,k,l] = val
                    fourC2E[j,i,k,l] = fourC2E[i,j,k,l]
                    fourC2E[i,j,l,k] = fourC2E[i,j,k,l]
                    fourC2E[j,i,l,k] = fourC2E[i,j,k,l]
                    fourC2E[k,l,i,j] = fourC2E[i,j,k,l]
                    fourC2E[k,l,j,i] = fourC2E[i,j,k,l]
                    fourC2E[l,k,i,j] = fourC2E[i,j,k,l]
                    fourC2E[l,k,j,i] = fourC2E[i,j,k,l]
                                    
                                   
                            
        
    return fourC2E

"Form coulomb repulsion integral using Rys quadrature"
def coulomb_rysNumba1(roots,weights,G,rpq2, rho, norder,n,m,la,lb,lc,ld,ma,mb,mc,md,na,nb,nc,nd,alphaik, alphajk, alphakk, alphalk,I,J,K,L):
    X = rpq2*rho

    
    # roots = np.zeros((norder))
    # weights = np.zeros((norder))
    # G = np.zeros((n+1,m+1))

    roots, weights = RootsNumba2(norder,X,roots,weights)
    

    ijkl = 0.0
    for i in range(norder):
        G = RecurNumba2(G,roots[i],la,lb,lc,ld,I[0],J[0],K[0],L[0],alphaik,alphajk,alphakk,alphalk)
        
        Ix = Int1dNumba2(G,roots[i],la,lb,lc,ld,I[0],J[0],K[0],L[0],
                 alphaik,alphajk,alphakk,alphalk)
        
        G = RecurNumba2(G,roots[i],ma,mb,mc,md,I[1],J[1],K[1],L[1],alphaik,alphajk,alphakk,alphalk)
        Iy = Int1dNumba2(G,roots[i],ma,mb,mc,md,I[1],J[1],K[1],L[1],
                 alphaik,alphajk,alphakk,alphalk)
        
        G = RecurNumba2(G,roots[i],na,nb,nc,nd,I[2],J[2],K[2],L[2],alphaik,alphajk,alphakk,alphalk)
        
        Iz = Int1dNumba2(G,roots[i],na,nb,nc,nd,I[2],J[2],K[2],L[2],
                 alphaik,alphajk,alphakk,alphalk)
        ijkl += Ix*Iy*Iz*weights[i] # ABD eq 5 & 9
        
        
        
    val = 2*np.sqrt(rho/np.pi)*ijkl # ABD eq 5 & 9
    
    
    return  val


def Int1dNumba1(G,t,ix,jx,kx,lx,xi,xj,xk,xl,alphai,alphaj,alphak,alphal):
    #G = RecurNumba2(G,t,ix,jx,kx,lx,xi,xj,xk,xl,alphai,alphaj,alphak,alphal)
    return ShiftNumba2(G,ix,jx,kx,lx,xi-xj,xk-xl)

"Form G(n,m)=I(n,0,m,0) intermediate values for a Rys polynomial"
def RecurNumba1(G,t,i,j,k,l,xi,xj,xk,xl,alphai,alphaj,alphak,alphal):
    # print('RecurNumba1', G[0,0])
    # G1 = np.zeros((n1+1,m1+1))
    
    n = i+j
    m = k+l        
    A = alphai+alphaj 
    B = alphak+alphal 
    Px = (alphai*xi+alphaj*xj)/A
    Qx = (alphak*xk+alphal*xl)/B

    C,Cp,B0,B1,B1p = RecurFactorsNumba2(t,A,B,Px,Qx,xi,xk) 
    

    # ABD eq 11.
    G[0,0] = np.pi*np.exp(-alphai*alphaj*(xi-xj)**2/(alphai+alphaj)
                    -alphak*alphal*(xk-xl)**2/(alphak+alphal))/np.sqrt(A*B)

    

    if n > 0: G[1,0] = C*G[0,0]   # ABD eq 15
    if m > 0: G[0,1] = Cp*G[0,0]  # ABD eq 16

    for a in range(2,n+1):
        G[a,0] = B1*(a-1)*G[a-2,0] + C*G[a-1,0]
    
    for b in range(2,m+1):
        G[0,b] = B1p*(b-1)*G[0,b-2] + Cp*G[0,b-1]
    

    
    if m==0 or n==0: 
        return G

    for a in range(1,n+1):
        G[a,1] = a*B0*G[a-1,0] + Cp*G[a,0]
        for b in range(2,m+1):
            G[a,b] = B1p*(b-1)*G[a,b-2] + a*B0*G[a-1,b-1] + Cp*G[a,b-1]

    
    
    return G

"Compute and  output I(i,j,k,l) from I(i+j,0,k+l,0) (G)"
def ShiftNumba1(G,i,j,k,l,xij,xkl):
    ijkl = 0.0 
    for m in range(l+1) :
        ijm0 = 0.0
        for n in range(j+1):
            ijm0 += comb(j,n)*xij**(j-n)*G[n+i,m+k]
        ijkl += comb(l,m)*xkl**(l-m)*ijm0 # I(i,j,k,l)<-I(i,j,m,0)  
    return ijkl

def RecurFactorsNumba1(t,A,B,Px,Qx,xi,xk):
    # Analogous versions taken from Gamess source code
    ooopt = 1/(1+t)
    fact = t*ooopt/(A+B)
    B0 = 0.5*fact
    # B1 = 0.5*ooopt/A + 0.5*fact
    # B1p = 0.5*ooopt/B + 0.5*fact
    B1 = 0.5*ooopt/A + B0
    B1p = 0.5*ooopt/B + B0
    C = (Px-xi)*ooopt + (B*(Qx-xi)+A*(Px-xi))*fact
    Cp = (Qx-xk)*ooopt + (B*(Qx-xk)+A*(Px-xk))*fact
    return C,Cp,B0,B1,B1p

# Everything after this computes the Roots and Weights of the Rys polynomial.
# Would be nice to find a more intuitive way to do this.
"Roots(n,X,roots,weights) - Return roots and weights of nth order Rys quadrature"
def RootsNumba1(n,X,roots,weights):
    if n == 1:
        return Root1Numba2(X,n,roots,weights)
    elif n == 2:
        return Root2Numba2(X,n,roots,weights)
    elif n == 3:
        return Root3Numba2(X,n,roots,weights)
    elif n == 4:
        return Root4Numba2(X,n,roots,weights)
    elif n == 5:
        return Root5Numba2(X,n,roots,weights)

def nERIRysNumba1(t, q1, q2, q3, q4, a12, a34,
             A, B, C, D, P, Q, sRys):
    '''Calculate Rys polynomials for two-electron integrals
    '''  
    # base case
    if q1+q2+q3+q4 == 0:   return 1.0
    
    # cdef double PQ,a1234,C00
    
    #distance between the two gaussian centers   
    PQ = P - Q
    a1234 = a12 + a34

    C00 = P - A - a34 * (PQ) * t / a1234

    if  q1==1 and q2+q3+q4==0: return C00
    
    # cdef double C01, D00, B00

    if q2==1 and q1+q3+q4==0:
        C01 = C00 + (A - B)
        return C01

    D00 = Q - C + a12 * (PQ) * t / a1234
    if  q3==1 and q1+q2+q4==0: return D00


    if  q1+q2+q3==0 and q4==1:
        D01 = D00 + (C - D)
        return D01

    B00 = t / (2 * a1234)    

    if q1==1 and q3==1 and q2+q4 == 0:   return D00 * C00 + B00
    
    # cdef int n, m, i, j
    n , m = q1 + q2 , q3 + q4
 
    if n > 0: sRys[1,0] = C00
    if m > 0: sRys[0,1] = D00


    B10 = 1 /(2 * a12) + -a34 * t/ ((2 * a12) * (a1234))
    B01 = 1 /(2 * a34) +  -a12 * t / ((2 * a34) * (a1234))

    for i in range(1, n+1):
        sRys[i,0] = C00 * sRys[i-1,0] + (i - 1) * B10 * sRys[i-2,0]

    for i in range(1,m+1):
        sRys[0,i] = D00 * sRys[0,i-1] + (i - 1) * B01 * sRys[0,i-2]   
    
    if m * n > 0:
    
        for i in range(1,n+1):
            sRys[i,1] = D00 * sRys[i,0] + i * B00 * sRys[i-1,0]
        for j in range(2,m+1):
            sRys[i,j] = (j-1) * B01 * sRys[i,j-2] + i * B00 * sRys[i-1,j-1] + D00 * sRys[i,j-1]
 
    if q2 + q4 == 0: return sRys[q1,q3]  
    
    # cdef double Rys,Rys0,AB,CD,Poly1

    Rys = 0

    #Angular momentum transfer
    AB,CD = A-B, C-D

    for i in range(q4+1):
        Rys0 = 0
        for j in range(q2+1):
           
            # I(i,j,m,0)<-I(n,0,m,0)
            Poly1 = comb(q2,j) * np.power(AB,q2-j) * sRys[j+q1,i+q3]
            Rys0 += Poly1
            
        Rys0 *= comb(q4, i) * np.power(CD, q4-i)
        Rys += Rys0 # I(i,j,k,l)<-I(i,j,m,0)
        
    return Rys

def ChebGausIntNumba1(eps,M,a12,a34, qx1, qx2, qx3, qx4,
                         qy1, qy2, qy3,  qy4, qz1, qz2, qz3,  qz4, x1,
                         x2,  x3, x4,  y1,  y2,  y3, y4,
                         z1,  z2,  z3, z4,  Px, Py, Pz,
                         Qx, Qy, Qz, T, sRys):
    
    # cdef double c0,s0,c1,s1,q,p,chp,c,s,xp,t1,t2,ang1,ang2,err
    # cdef int j,n,i
    err,n,c0,s0 = 10,3,0.866025403784438646,.5
    c1 = s0
    s1 = c0
    t1,t2 = 0.7628537665044517,0.0160237616047743
    ang1 = (nERIRysNumba2(t1,qx1,qx2,qx3,qx4,a12,a34,x1,x2,x3,x4,Px,Qx,sRys)* 
            nERIRysNumba2(t1,qy1,qy2,qy3,qy4,a12,a34,y1,y2,y3,y4,Py,Qy,sRys)*
            nERIRysNumba2(t1,qz1,qz2,qz3,qz4,a12,a34,z1,z2,z3,z4,Pz,Qz,sRys))
            
    ang2 = (nERIRysNumba2(t2,qx1,qx2,qx3,qx4,a12,a34,x1,x2,x3,x4,Px,Qx,sRys)* 
            nERIRysNumba2(t2,qy1,qy2,qy3,qy4,a12,a34,y1,y2,y3,y4,Py,Qy,sRys)*
            nERIRysNumba2(t2,qz1,qz2,qz3,qz4,a12,a34,z1,z2,z3,z4,Pz,Qz,sRys))
            
    q = 0.28125 * (np.exp(-T*t1) * ang1 + np.exp(-T*t2) * ang2)
    p = (0.5*np.exp(-T*0.25)* nERIRysNumba2(0.25,qx1,qx2,qx3,qx4,a12,a34,x1,x2,x3,x4,Px,Qx,sRys)*
         nERIRysNumba2(0.25,qy1,qy2,qy3,qy4,a12,a34,y1,y2,y3,y4,Py,Qy,sRys)*
         nERIRysNumba2(0.25,qz1,qz2,qz3,qz4,a12,a34,z1,z2,z3,z4,Pz,Qz,sRys))
    chp = q + p
    j = 0
    while err > eps:
        j = 1 - j
        c1 = j * c1 + (1-j) * c0
        s1 = j * s1 + (1-j) * s0
        c0 = j * c0 + (1-j) * np.sqrt(0.5 * (1 + c0))
        s0 *= j + (1-j)/(2 * c0)
        c,s = c0,s0
        
        for i in range(1,n,2):
            xp = 1 + 0.21220659078919378 * s * c * (3 + 2*s*s) - i/n
            if np.ceil(3*(i+j+j)/3.) > (i + j):
               
                t1 = 0.25 * (-xp+1) * (-xp+1)
                t2 = 0.25 * (xp+1) * (xp+1)
               
                ang1 = (nERIRysNumba2(t1,qx1,qx2,qx3,qx4,a12,a34,x1,x2,x3,x4,Px,Qx,sRys)*
                        nERIRysNumba2(t1,qy1,qy2,qy3,qy4,a12,a34,y1,y2,y3,y4,Py,Qy,sRys)*
                        nERIRysNumba2(t1,qz1,qz2,qz3,qz4,a12,a34,z1,z2,z3,z4,Pz,Qz,sRys))
                        
                ang2 = (nERIRysNumba2(t2,qx1,qx2,qx3,qx4,a12,a34,x1,x2,x3,x4,Px,Qx,sRys)* 
                        nERIRysNumba2(t2,qy1,qy2,qy3,qy4,a12,a34,y1,y2,y3,y4,Py,Qy,sRys)*
                        nERIRysNumba2(t2,qz1,qz2,qz3,qz4,a12,a34,z1,z2,z3,z4,Pz,Qz,sRys))
                
                chp += 0.5 * (np.exp(-T*t1) * ang1 + np.exp(-T*t2) * ang2) * s ** 4
                
            xp = s 
            s = s*c1 + c*s1
            c = c*c1 - xp*s1
            
        n *= (1+j)
        p += (1-j)*(chp-q)
        err = 16 * np.abs((1-j)*(q-3*p/2)+j*(chp-2*q))/(3*n)
        q = (1 - j) * q + j * chp
        print(err)
    print('convgd')
    ssss = 1.0
    return 16*q/(3*n) 

def Root1Numba1(X,n,roots,weights):
    # roots = np.zeros((n))
    # weights = np.zeros((n))
    R12,PIE4 = 2.75255128608411E-01, 7.85398163397448E-01
    # R22,W22 =  2.72474487139158E+00, 9.17517095361369E-02
    # R13 = 1.90163509193487E-01
    # R23,W23 = 1.78449274854325E+00, 1.77231492083829E-01
    # R33,W33 = 5.52534374226326E+00, 5.11156880411248E-03
    if X < 3.0E-7:
        roots0 = 0.5E+00-X/5.0E+00
        weights0 = 1.0E+00-X/3.0E+00
    elif X < 1.0:
        F1 = ((((((((-8.36313918003957E-08*X+1.21222603512827E-06 )*X-
                1.15662609053481E-05 )*X+9.25197374512647E-05 )*X-
              6.40994113129432E-04 )*X+3.78787044215009E-03 )*X-
            1.85185172458485E-02 )*X+7.14285713298222E-02 )*X-
          1.99999999997023E-01 )*X+3.33333333333318E-01
        weights0 = (X+X)*F1+np.exp(-X)
        roots0 = F1/(weights0-F1)
    elif X < 3.0:
        Y = X-2.0E+00
        F1 = ((((((((((-1.61702782425558E-10*Y+1.96215250865776E-09 )*Y-
                  2.14234468198419E-08 )*Y+2.17216556336318E-07 )*Y-
                1.98850171329371E-06 )*Y+1.62429321438911E-05 )*Y-
              1.16740298039895E-04 )*Y+7.24888732052332E-04 )*Y-
            3.79490003707156E-03 )*Y+1.61723488664661E-02 )*Y-
          5.29428148329736E-02 )*Y+1.15702180856167E-01
        weights0 = (X+X)*F1+np.exp(-X)
        roots0 = F1/(weights0-F1)
    elif X < 5.0:
        Y = X-4.0E+00
        F1 = ((((((((((-2.62453564772299E-11*Y+3.24031041623823E-10 )*Y-
                  3.614965656163E-09)*Y+3.760256799971E-08)*Y-
                3.553558319675E-07)*Y+3.022556449731E-06)*Y-
              2.290098979647E-05)*Y+1.526537461148E-04)*Y-
            8.81947375894379E-04 )*Y+4.33207949514611E-03 )*Y-
          1.75257821619926E-02 )*Y+5.28406320615584E-02
        weights0 = (X+X)*F1+np.exp(-X)
        roots0 = F1/(weights0-F1)
    elif X < 10.0:
        E = np.exp(-X)
        weights0 = (((((( 4.6897511375022E-01/X-6.9955602298985E-01)/X +
                5.3689283271887E-01)/X-3.2883030418398E-01)/X +
              2.4645596956002E-01)/X-4.9984072848436E-01)/X -3.1501078774085E-06)*E + np.sqrt(PIE4/X)
        F1 = (weights0-E)/(X+X)
        roots0 = F1/(weights0-F1)
    elif X < 15.0:
        E = np.exp(-X)
        weights0 = (((-1.8784686463512E-01/X+2.2991849164985E-01)/X -
              4.9893752514047E-01)/X-2.1916512131607E-05)*E + np.sqrt(PIE4/X)
        F1 = (weights0-E)/(X+X)
        roots0 = F1/(weights0-F1)
    elif X < 33.0:
        E = np.exp(-X)
        weights0 = (( 1.9623264149430E-01/X-4.9695241464490E-01)/X -
             6.0156581186481E-05)*E + np.sqrt(PIE4/X)
        F1 = (weights0-E)/(X+X)
        roots0 = F1/(weights0-F1)
    else:  # X > 33
        weights0 = np.sqrt(PIE4/X)
        roots0 = 0.5E+00/(X-0.5E+00)

    roots[0]=roots0
    weights[0]=weights0
  
    return roots, weights

def Root2Numba1(X,n, roots, weights):
  # roots = np.zeros((n))
  # weights = np.zeros((n))
  R12,PIE4 = 2.75255128608411E-01, 7.85398163397448E-01
  R22,W22 =  2.72474487139158E+00, 9.17517095361369E-02
  # R13 = 1.90163509193487E-01
  # R23,W23 = 1.78449274854325E+00, 1.77231492083829E-01
  # R33,W33 = 5.52534374226326E+00, 5.11156880411248E-03
  if X < 3.E-7:
    roots0 = 1.30693606237085E-01-2.90430236082028E-02 *X
    roots1 = 2.86930639376291E+00-6.37623643058102E-01 *X
    weights0 = 6.52145154862545E-01-1.22713621927067E-01 *X
    weights1 = 3.47854845137453E-01-2.10619711404725E-01 *X
  elif X < 1.0:
    F1 = ((((((((-8.36313918003957E-08*X+1.21222603512827E-06 )*X-
                1.15662609053481E-05 )*X+9.25197374512647E-05 )*X-
              6.40994113129432E-04 )*X+3.78787044215009E-03 )*X-
            1.85185172458485E-02 )*X+7.14285713298222E-02 )*X-
          1.99999999997023E-01 )*X+3.33333333333318E-01
    weights0 = (X+X)*F1+np.exp(-X)
    roots0 = (((((((-2.35234358048491E-09*X+2.49173650389842E-08)*X-
                4.558315364581E-08)*X-2.447252174587E-06)*X+
              4.743292959463E-05)*X-5.33184749432408E-04 )*X+
            4.44654947116579E-03 )*X-2.90430236084697E-02 )*X+1.30693606237085E-01
    roots1 = (((((((-2.47404902329170E-08*X+2.36809910635906E-07)*X+
                1.835367736310E-06)*X-2.066168802076E-05)*X-
              1.345693393936E-04)*X-5.88154362858038E-05 )*X+
            5.32735082098139E-02 )*X-6.37623643056745E-01 )*X+2.86930639376289E+00
    weights1 = ((F1-weights0)*roots0+F1)*(1.0E+00+roots1)/(roots1-roots0)
    weights0 = weights0-weights1
  elif X < 3.0:
    Y = X-2.0E+00
    F1 = ((((((((((-1.61702782425558E-10*Y+1.96215250865776E-09 )*Y-
                  2.14234468198419E-08 )*Y+2.17216556336318E-07 )*Y-
                1.98850171329371E-06 )*Y+1.62429321438911E-05 )*Y-
              1.16740298039895E-04 )*Y+7.24888732052332E-04 )*Y-
            3.79490003707156E-03 )*Y+1.61723488664661E-02 )*Y-
          5.29428148329736E-02 )*Y+1.15702180856167E-01
    weights0 = (X+X)*F1+np.exp(-X)
    roots0 = (((((((((-6.36859636616415E-12*Y+8.47417064776270E-11)*Y-
                  5.152207846962E-10)*Y-3.846389873308E-10)*Y+
                8.472253388380E-08)*Y-1.85306035634293E-06 )*Y+
              2.47191693238413E-05 )*Y-2.49018321709815E-04 )*Y+
            2.19173220020161E-03 )*Y-1.63329339286794E-02 )*Y+8.68085688285261E-02
    roots1 = ((((((((( 1.45331350488343E-10*Y+2.07111465297976E-09)*Y-
                  1.878920917404E-08)*Y-1.725838516261E-07)*Y+
                2.247389642339E-06)*Y+9.76783813082564E-06 )*Y-
              1.93160765581969E-04 )*Y-1.58064140671893E-03 )*Y+
            4.85928174507904E-02 )*Y-4.30761584997596E-01 )*Y+1.80400974537950E+00
    weights1 = ((F1-weights0)*roots0+F1)*(1.0E+00+roots1)/(roots1-roots0)
    weights0 = weights0-weights1
  elif X < 5.0:
    Y = X-4.0E+00
    F1 = ((((((((((-2.62453564772299E-11*Y+3.24031041623823E-10 )*Y-
                  3.614965656163E-09)*Y+3.760256799971E-08)*Y-
                3.553558319675E-07)*Y+3.022556449731E-06)*Y-
              2.290098979647E-05)*Y+1.526537461148E-04)*Y-
            8.81947375894379E-04 )*Y+4.33207949514611E-03 )*Y-
          1.75257821619926E-02 )*Y+5.28406320615584E-02
    weights0 = (X+X)*F1+np.exp(-X)
    roots0 = ((((((((-4.11560117487296E-12*Y+7.10910223886747E-11)*Y-
                  1.73508862390291E-09 )*Y+5.93066856324744E-08 )*Y-
                9.76085576741771E-07 )*Y+1.08484384385679E-05 )*Y-
              1.12608004981982E-04 )*Y+1.16210907653515E-03 )*Y-
            9.89572595720351E-03 )*Y+6.12589701086408E-02
    roots1 = (((((((((-1.80555625241001E-10*Y+5.44072475994123E-10)*Y+
                  1.603498045240E-08)*Y-1.497986283037E-07)*Y-
                7.017002532106E-07)*Y+1.85882653064034E-05 )*Y-
              2.04685420150802E-05 )*Y-2.49327728643089E-03 )*Y+
            3.56550690684281E-02 )*Y-2.60417417692375E-01 )*Y+1.12155283108289E+00
    weights1 = ((F1-weights0)*roots0+F1)*(1.0E+00+roots1)/(roots1-roots0)
    weights0 = weights0-weights1
  elif X < 10.0:
    E = np.exp(-X)
    weights0 = (((((( 4.6897511375022E-01/X-6.9955602298985E-01)/X +
                5.3689283271887E-01)/X-3.2883030418398E-01)/X +
              2.4645596956002E-01)/X-4.9984072848436E-01)/X -
            3.1501078774085E-06)*E + np.sqrt(PIE4/X)
    F1 = (weights0-E)/(X+X)
    Y = X-7.5E+00
    roots0 = (((((((((((((-1.43632730148572E-16*Y+2.38198922570405E-16)*
                      Y+1.358319618800E-14)*Y-7.064522786879E-14)*Y-
                    7.719300212748E-13)*Y+7.802544789997E-12)*Y+
                  6.628721099436E-11)*Y-1.775564159743E-09)*Y+
                1.713828823990E-08)*Y-1.497500187053E-07)*Y+
              2.283485114279E-06)*Y-3.76953869614706E-05 )*Y+
            4.74791204651451E-04 )*Y-4.60448960876139E-03 )*Y+3.72458587837249E-02
    roots1 = (((((((((((( 2.48791622798900E-14*Y-1.36113510175724E-13)*Y-
                      2.224334349799E-12)*Y+4.190559455515E-11)*Y-
                    2.222722579924E-10)*Y-2.624183464275E-09)*Y+
                  6.128153450169E-08)*Y-4.383376014528E-07)*Y-
                2.49952200232910E-06 )*Y+1.03236647888320E-04 )*Y-
              1.44614664924989E-03 )*Y+1.35094294917224E-02 )*Y-
            9.53478510453887E-02 )*Y+5.44765245686790E-01
    weights1 = ((F1-weights0)*roots0+F1)*(1.0E+00+roots1)/(roots1-roots0)
    weights0 = weights0-weights1
  elif X < 15.0:
    E = np.exp(-X)
    weights0 = (((-1.8784686463512E-01/X+2.2991849164985E-01)/X -
            4.9893752514047E-01)/X-2.1916512131607E-05)*E + np.sqrt(PIE4/X)
    F1 = (weights0-E)/(X+X)
    roots0 = ((((-1.01041157064226E-05*X+1.19483054115173E-03)*X -
              6.73760231824074E-02)*X+1.25705571069895E+00)*X +
            (((-8.57609422987199E+03/X+5.91005939591842E+03)/X -
              1.70807677109425E+03)/X+2.64536689959503E+02)/X -
            2.38570496490846E+01)*E + R12/(X-R12)
    roots1 = ((( 3.39024225137123E-04*X-9.34976436343509E-02)*X -
            4.22216483306320E+00)*X +
            (((-2.08457050986847E+03/X -
              1.04999071905664E+03)/X+3.39891508992661E+02)/X -
            1.56184800325063E+02)/X+8.00839033297501E+00)*E + R22/(X-R22)
    weights1 = ((F1-weights0)*roots0+F1)*(1.0E+00+roots1)/(roots1-roots0)
    weights0 = weights0-weights1
  elif X < 33.0:
    E = np.exp(-X)
    weights0 = (( 1.9623264149430E-01/X-4.9695241464490E-01)/X -
            6.0156581186481E-05)*E + np.sqrt(PIE4/X)
    F1 = (weights0-E)/(X+X)
    roots0 = ((((-1.14906395546354E-06*X+1.76003409708332E-04)*X -
              1.71984023644904E-02)*X-1.37292644149838E-01)*X +
            (-4.75742064274859E+01/X+9.21005186542857E+00)/X -
            2.31080873898939E-02)*E + R12/(X-R12)
    roots1 = ((( 3.64921633404158E-04*X-9.71850973831558E-02)*X -
            4.02886174850252E+00)*X +
            (-1.35831002139173E+02/X -
            8.66891724287962E+01)/X+2.98011277766958E+00)*E + R22/(X-R22)
    weights1 = ((F1-weights0)*roots0+F1)*(1.0E+00+roots1)/(roots1-roots0)
    weights0 = weights0-weights1
  else:  # X > 33
    weights0 = np.sqrt(PIE4/X)
    if X < 40.0:
        E = np.exp(-X)
        roots0 = (-8.78947307498880E-01*X+1.09243702330261E+01)*E + R12/(X-R12)
        roots1 = (-9.28903924275977E+00*X+8.10642367843811E+01)*E + R22/(X-R22)
        weights1 = ( 4.46857389308400E+00*X-7.79250653461045E+01)*E + W22*weights0
        weights0 = weights0-weights1
    else:
        roots0 = R12/(X-R12)
        roots1 = R22/(X-R22)
        weights1 = W22*weights0
        weights0 = weights0-weights1
     
  roots[0] = roots0
  roots[1] = roots1
  weights[0]= weights0
  weights[1] = weights1
  
  return roots, weights#[RT1,roots[2]],[weights1,weights[2]]  

def Root3Numba1(X,n,roots, weights):
  # roots = np.zeros((n))
  # weights = np.zeros((n))
  R12,PIE4 = 2.75255128608411E-01, 7.85398163397448E-01
  # R22,W22 =  2.72474487139158E+00, 9.17517095361369E-02
  R13 = 1.90163509193487E-01
  R23,W23 = 1.78449274854325E+00, 1.77231492083829E-01
  R33,W33 = 5.52534374226326E+00, 5.11156880411248E-03
  if X < 3.0E-7:
    roots0 = 6.03769246832797E-02-9.28875764357368E-03 *X
    roots1 = 7.76823355931043E-01-1.19511285527878E-01 *X
    roots2 = 6.66279971938567E+00-1.02504611068957E+00 *X
    weights0 = 4.67913934572691E-01-5.64876917232519E-02 *X
    weights1 = 3.60761573048137E-01-1.49077186455208E-01 *X
    weights2 = 1.71324492379169E-01-1.27768455150979E-01 *X
  elif X < 1.0:
    roots0 = ((((((-5.10186691538870E-10*X+2.40134415703450E-08)*X-
          5.01081057744427E-07 )*X+7.58291285499256E-06 )*X-
        9.55085533670919E-05 )*X+1.02893039315878E-03 )*X-
      9.28875764374337E-03 )*X+6.03769246832810E-02
    roots1 = ((((((-1.29646524960555E-08*X+7.74602292865683E-08)*X+
                1.56022811158727E-06 )*X-1.58051990661661E-05 )*X-
              3.30447806384059E-04 )*X+9.74266885190267E-03 )*X-
            1.19511285526388E-01 )*X+7.76823355931033E-01
    roots2 = ((((((-9.28536484109606E-09*X-3.02786290067014E-07)*X-
                2.50734477064200E-06 )*X-7.32728109752881E-06 )*X+
              2.44217481700129E-04 )*X+4.94758452357327E-02 )*X-
            1.02504611065774E+00 )*X+6.66279971938553E+00
    F2 = ((((((((-7.60911486098850E-08*X+1.09552870123182E-06 )*X-
                1.03463270693454E-05 )*X+8.16324851790106E-05 )*X-
              5.55526624875562E-04 )*X+3.20512054753924E-03 )*X-
            1.51515139838540E-02 )*X+5.55555554649585E-02 )*X-
          1.42857142854412E-01 )*X+1.99999999999986E-01
    E = np.exp(-X)
    F1 = ((X+X)*F2+E)/3.0E+00
    weights0 = (X+X)*F1+E
    T1 = roots0/(roots0+1.0E+00)
    T2 = roots1/(roots1+1.0E+00)
    T3 = roots2/(roots2+1.0E+00)
    A2 = F2-T1*F1
    A1 = F1-T1*weights0
    weights2 = (A2-T2*A1)/((T3-T2)*(T3-T1))
    weights1 = (T3*A1-A2)/((T3-T2)*(T2-T1))
    weights0 = weights0-weights1-weights2
  elif X < 3.0:
    Y = X-2.0E+00
    roots0 = (((((((( 1.44687969563318E-12*Y+4.85300143926755E-12)*Y-
                  6.55098264095516E-10 )*Y+1.56592951656828E-08 )*Y-
                2.60122498274734E-07 )*Y+3.86118485517386E-06 )*Y-
              5.13430986707889E-05 )*Y+6.03194524398109E-04 )*Y-
            6.11219349825090E-03 )*Y+4.52578254679079E-02
    roots1 = ((((((( 6.95964248788138E-10*Y-5.35281831445517E-09)*Y-
                6.745205954533E-08)*Y+1.502366784525E-06)*Y+
              9.923326947376E-07)*Y-3.89147469249594E-04 )*Y+
            7.51549330892401E-03 )*Y-8.48778120363400E-02 )*Y+5.73928229597613E-01
    roots2 = ((((((((-2.81496588401439E-10*Y+3.61058041895031E-09)*Y+
                  4.53631789436255E-08 )*Y-1.40971837780847E-07 )*Y-
                6.05865557561067E-06 )*Y-5.15964042227127E-05 )*Y+
              3.34761560498171E-05 )*Y+5.04871005319119E-02 )*Y-
            8.24708946991557E-01 )*Y+4.81234667357205E+00
    F2 = ((((((((((-1.48044231072140E-10*Y+1.78157031325097E-09 )*Y-
                  1.92514145088973E-08 )*Y+1.92804632038796E-07 )*Y-
                1.73806555021045E-06 )*Y+1.39195169625425E-05 )*Y-
              9.74574633246452E-05 )*Y+5.83701488646511E-04 )*Y-
            2.89955494844975E-03 )*Y+1.13847001113810E-02 )*Y-
          3.23446977320647E-02 )*Y+5.29428148329709E-02
    E = np.exp(-X)
    F1 = ((X+X)*F2+E)/3.0E+00
    weights0 = (X+X)*F1+E
    T1 = roots0/(roots0+1.0E+00)
    T2 = roots1/(roots1+1.0E+00)
    T3 = roots2/(roots2+1.0E+00)
    A2 = F2-T1*F1
    A1 = F1-T1*weights0
    weights2 = (A2-T2*A1)/((T3-T2)*(T3-T1))
    weights1 = (T3*A1-A2)/((T3-T2)*(T2-T1))
    weights0 = weights0-weights1-weights2
  elif X < 5.0:
    Y = X-4.0E+00
    roots0 = ((((((( 1.44265709189601E-11*Y-4.66622033006074E-10)*Y+
                7.649155832025E-09)*Y-1.229940017368E-07)*Y+
              2.026002142457E-06)*Y-2.87048671521677E-05 )*Y+
            3.70326938096287E-04 )*Y-4.21006346373634E-03 )*Y+3.50898470729044E-02
    roots1 = ((((((((-2.65526039155651E-11*Y+1.97549041402552E-10)*Y+
                  2.15971131403034E-09 )*Y-7.95045680685193E-08 )*Y+
                5.15021914287057E-07 )*Y+1.11788717230514E-05 )*Y-
              3.33739312603632E-04 )*Y+5.30601428208358E-03 )*Y-
            5.93483267268959E-02 )*Y+4.31180523260239E-01
    roots2 = ((((((((-3.92833750584041E-10*Y-4.16423229782280E-09)*Y+
                  4.42413039572867E-08 )*Y+6.40574545989551E-07 )*Y-
                3.05512456576552E-06 )*Y-1.05296443527943E-04 )*Y-
              6.14120969315617E-04 )*Y+4.89665802767005E-02 )*Y-
            6.24498381002855E-01 )*Y+3.36412312243724E+00
    F2 = ((((((((((-2.36788772599074E-11*Y+2.89147476459092E-10 )*Y-
                  3.18111322308846E-09 )*Y+3.25336816562485E-08 )*Y-
                3.00873821471489E-07 )*Y+2.48749160874431E-06 )*Y-
              1.81353179793672E-05 )*Y+1.14504948737066E-04 )*Y-
            6.10614987696677E-04 )*Y+2.64584212770942E-03 )*Y-
          8.66415899015349E-03 )*Y+1.75257821619922E-02
    E = np.exp(-X)
    F1 = ((X+X)*F2+E)/3.0E+00
    weights0 = (X+X)*F1+E
    T1 = roots0/(roots0+1.0E+00)
    T2 = roots1/(roots1+1.0E+00)
    T3 = roots2/(roots2+1.0E+00)
    A2 = F2-T1*F1
    A1 = F1-T1*weights0
    weights2 = (A2-T2*A1)/((T3-T2)*(T3-T1))
    weights1 = (T3*A1-A2)/((T3-T2)*(T2-T1))
    weights0 = weights0-weights1-weights2
  elif X < 10.0:
    E = np.exp(-X)
    weights0 = (((((( 4.6897511375022E-01/X-6.9955602298985E-01)/X +
                5.3689283271887E-01)/X-3.2883030418398E-01)/X +
              2.4645596956002E-01)/X-4.9984072848436E-01)/X -
            3.1501078774085E-06)*E + np.sqrt(PIE4/X)
    F1 = (weights0-E)/(X+X)
    F2 = (F1+F1+F1-E)/(X+X)
    Y = X-7.5E+00
    roots0 = ((((((((((( 5.74429401360115E-16*Y+7.11884203790984E-16)*Y-
                    6.736701449826E-14)*Y-6.264613873998E-13)*Y+
                  1.315418927040E-11)*Y-4.23879635610964E-11 )*Y+
                1.39032379769474E-09 )*Y-4.65449552856856E-08 )*Y+
              7.34609900170759E-07 )*Y-1.08656008854077E-05 )*Y+
            1.77930381549953E-04 )*Y-2.39864911618015E-03 )*Y+2.39112249488821E-02
    roots1 = ((((((((((( 1.13464096209120E-14*Y+6.99375313934242E-15)*Y-
                    8.595618132088E-13)*Y-5.293620408757E-12)*Y-
                  2.492175211635E-11)*Y+2.73681574882729E-09 )*Y-
                1.06656985608482E-08 )*Y-4.40252529648056E-07 )*Y+
              9.68100917793911E-06 )*Y-1.68211091755327E-04 )*Y+
            2.69443611274173E-03 )*Y-3.23845035189063E-02 )*Y+2.75969447451882E-01
    roots2 = (((((((((((( 6.66339416996191E-15*Y+1.84955640200794E-13)*Y-
                      1.985141104444E-12)*Y-2.309293727603E-11)*Y+
                    3.917984522103E-10)*Y+1.663165279876E-09)*Y-
                  6.205591993923E-08)*Y+8.769581622041E-09)*Y+
                8.97224398620038E-06 )*Y-3.14232666170796E-05 )*Y-
              1.83917335649633E-03 )*Y+3.51246831672571E-02 )*Y-
            3.22335051270860E-01 )*Y+1.73582831755430E+00
    T1 = roots0/(roots0+1.0E+00)
    T2 = roots1/(roots1+1.0E+00)
    T3 = roots2/(roots2+1.0E+00)
    A2 = F2-T1*F1
    A1 = F1-T1*weights0
    weights2 = (A2-T2*A1)/((T3-T2)*(T3-T1))
    weights1 = (T3*A1-A2)/((T3-T2)*(T2-T1))
    weights0 = weights0-weights1-weights2
  elif X < 15.0:
    E = np.exp(-X)
    weights0 = (((-1.8784686463512E-01/X+2.2991849164985E-01)/X -
            4.9893752514047E-01)/X-2.1916512131607E-05)*E + np.sqrt(PIE4/X)
    F1 = (weights0-E)/(X+X)
    F2 = (F1+F1+F1-E)/(X+X)
    Y = X-12.5E+00
    roots0 = ((((((((((( 4.42133001283090E-16*Y-2.77189767070441E-15)*Y-
                    4.084026087887E-14)*Y+5.379885121517E-13)*Y+
                  1.882093066702E-12)*Y-8.67286219861085E-11 )*Y+
                7.11372337079797E-10 )*Y-3.55578027040563E-09 )*Y+
              1.29454702851936E-07 )*Y-4.14222202791434E-06 )*Y+
            8.04427643593792E-05 )*Y-1.18587782909876E-03 )*Y+1.53435577063174E-02
    roots1 = ((((((((((( 6.85146742119357E-15*Y-1.08257654410279E-14)*Y-
                    8.579165965128E-13)*Y+6.642452485783E-12)*Y+
                  4.798806828724E-11)*Y-1.13413908163831E-09 )*Y+
                7.08558457182751E-09 )*Y-5.59678576054633E-08 )*Y+
              2.51020389884249E-06 )*Y-6.63678914608681E-05 )*Y+
            1.11888323089714E-03 )*Y-1.45361636398178E-02 )*Y+1.65077877454402E-01
    roots2 = (((((((((((( 3.20622388697743E-15*Y-2.73458804864628E-14)*Y-
                      3.157134329361E-13)*Y+8.654129268056E-12)*Y-
                    5.625235879301E-11)*Y-7.718080513708E-10)*Y+
                  2.064664199164E-08)*Y-1.567725007761E-07)*Y-
                1.57938204115055E-06 )*Y+6.27436306915967E-05 )*Y-
              1.01308723606946E-03 )*Y+1.13901881430697E-02 )*Y-
            1.01449652899450E-01 )*Y+7.77203937334739E-01
    T1 = roots0/(roots0+1.0E+00)
    T2 = roots1/(roots1+1.0E+00)
    T3 = roots2/(roots2+1.0E+00)
    A2 = F2-T1*F1
    A1 = F1-T1*weights0
    weights2 = (A2-T2*A1)/((T3-T2)*(T3-T1))
    weights1 = (T3*A1-A2)/((T3-T2)*(T2-T1))
    weights0 = weights0-weights1-weights2
  elif X < 33.0:
    E = np.exp(-X)
    weights0 = (( 1.9623264149430E-01/X-4.9695241464490E-01)/X -
            6.0156581186481E-05)*E + np.sqrt(PIE4/X)
    F1 = (weights0-E)/(X+X)
    F2 = (F1+F1+F1-E)/(X+X)
    if X < 20:
        roots0 = ((((((-2.43270989903742E-06*X+3.57901398988359E-04)*X -
                    2.34112415981143E-02)*X+7.81425144913975E-01)*X -
                  1.73209218219175E+01)*X+2.43517435690398E+02)*X +
                (-1.97611541576986E+04/X+9.82441363463929E+03)/X -
                2.07970687843258E+03)*E + R13/(X-R13)
        roots1 = (((((-2.62627010965435E-04*X+3.49187925428138E-02)*X -
                  3.09337618731880E+00)*X+1.07037141010778E+02)*X -
                2.36659637247087E+03)*X +
                ((-2.91669113681020E+06/X +
                  1.41129505262758E+06)/X-2.91532335433779E+05)/X +
                3.35202872835409E+04)*E + R23/(X-R23)
        roots2 = ((((( 9.31856404738601E-05*X-2.87029400759565E-02)*X -
                  7.83503697918455E-01)*X-1.84338896480695E+01)*X +
                4.04996712650414E+02)*X +
                (-1.89829509315154E+05/X +
                5.11498390849158E+04)/X-6.88145821789955E+03)*E + R33/(X-R33)
    else:
        roots0 = ((((-4.97561537069643E-04*X-5.00929599665316E-02)*X +
                  1.31099142238996E+00)*X-1.88336409225481E+01)*X -
                6.60344754467191E+02 /X+1.64931462413877E+02)*E + R13/(X-R13)
        roots1 = ((((-4.48218898474906E-03*X-5.17373211334924E-01)*X +
                  1.13691058739678E+01)*X-1.65426392885291E+02)*X -
                6.30909125686731E+03 /X+1.52231757709236E+03)*E + R23/(X-R23)
        roots2 = ((((-1.38368602394293E-02*X-1.77293428863008E+00)*X +
                  1.73639054044562E+01)*X-3.57615122086961E+02)*X -
                1.45734701095912E+04 /X+2.69831813951849E+03)*E + R33/(X-R33)
    
    T1 = roots0/(roots0+1.0E+00)
    T2 = roots1/(roots1+1.0E+00)
    T3 = roots2/(roots2+1.0E+00)
    A2 = F2-T1*F1
    A1 = F1-T1*weights0
    weights2 = (A2-T2*A1)/((T3-T2)*(T3-T1))
    weights1 = (T3*A1-A2)/((T3-T2)*(T2-T1))
    weights0 = weights0-weights1-weights2
  else  :# X > 33
    weights0 = np.sqrt(PIE4/X)
    if X < 47:
        E = np.exp(-X)
        roots0 = ((-7.39058467995275E+00*X+3.21318352526305E+02)*X -
                3.99433696473658E+03)*E + R13/(X-R13)
        roots1 = ((-7.38726243906513E+01*X+3.13569966333873E+03)*X -
                3.86862867311321E+04)*E + R23/(X-R23)
        roots2 = ((-2.63750565461336E+02*X+1.04412168692352E+04)*X -
                1.28094577915394E+05)*E + R33/(X-R33)
        weights2 = ((( 1.52258947224714E-01*X-8.30661900042651E+00)*X +
                1.92977367967984E+02)*X-1.67787926005344E+03)*E + W33*weights0
        weights1 = (( 6.15072615497811E+01*X-2.91980647450269E+03)*X +
                3.80794303087338E+04)*E + W23*weights0
        weights0 = weights0-weights1-weights2
    else:
        roots0 = R13/(X-R13)
        roots1 = R23/(X-R23)
        roots2 = R33/(X-R33)
        weights1 = W23*weights0
        weights2 = W33*weights0
        weights0 = weights0-weights1-weights2
  
  roots[0] = roots0
  roots[1] = roots1
  roots[2] = roots2
  weights[0] = weights0
  weights[1] = weights1
  weights[2] = weights2
    
  return roots, weights#[roots1,roots2,roots[3]],[weights1,weights2,weights[3]]

def Root4Numba1(X,n,roots, weights):
    # roots = np.zeros((n))
    # weights = np.zeros((n))
    R14,PIE4 = 1.45303521503316E-01, 7.85398163397448E-01
    R24,W24 = 1.33909728812636E+00, 2.34479815323517E-01
    R34,W34 = 3.92696350135829E+00, 1.92704402415764E-02
    R44,W44 = 8.58863568901199E+00, 2.25229076750736E-04

    if X <= 3.0E-7:
        roots0 = 3.48198973061471E-02 -4.09645850660395E-03 *X
        roots1 = 3.81567185080042E-01 -4.48902570656719E-02 *X
        roots2 = 1.73730726945891E+00 -2.04389090547327E-01 *X
        roots3 = 1.18463056481549E+01 -1.39368301742312E+00 *X
        weights0 = 3.62683783378362E-01 -3.13844305713928E-02 *X
        weights1 = 3.13706645877886E-01 -8.98046242557724E-02 *X
        weights2 = 2.22381034453372E-01 -1.29314370958973E-01 *X
        weights3 = 1.01228536290376E-01 -8.28299075414321E-02 *X
    elif X <= 1.0:
        roots0 = ((((((-1.95309614628539E-10*X+5.19765728707592E-09)*X-
                   1.01756452250573E-07 )*X+1.72365935872131E-06 )*X-
                 2.61203523522184E-05 )*X+3.52921308769880E-04 )*X-
               4.09645850658433E-03 )*X+3.48198973061469E-02
        roots1 = (((((-1.89554881382342E-08*X+3.07583114342365E-07)*X+
                  1.270981734393E-06)*X-1.417298563884E-04)*X+
                3.226979163176E-03)*X-4.48902570678178E-02 )*X+3.81567185080039E-01
        roots2 = (((((( 1.77280535300416E-09*X+3.36524958870615E-08)*X-
                   2.58341529013893E-07 )*X-1.13644895662320E-05 )*X-
                 7.91549618884063E-05 )*X+1.03825827346828E-02 )*X-
               2.04389090525137E-01 )*X+1.73730726945889E+00
        roots3 = (((((-5.61188882415248E-08*X-2.49480733072460E-07)*X+
                  3.428685057114E-06)*X+1.679007454539E-04)*X+
                4.722855585715E-02)*X-1.39368301737828E+00 )*X+1.18463056481543E+01
        weights0 = ((((((-1.14649303201279E-08*X+1.88015570196787E-07)*X-
                   2.33305875372323E-06 )*X+2.68880044371597E-05 )*X-
                 2.94268428977387E-04 )*X+3.06548909776613E-03 )*X-
               3.13844305680096E-02 )*X+3.62683783378335E-01
        weights1 = ((((((((-4.11720483772634E-09*X+6.54963481852134E-08)*X-
                     7.20045285129626E-07 )*X+6.93779646721723E-06 )*X-
                   6.05367572016373E-05 )*X+4.74241566251899E-04 )*X-
                 3.26956188125316E-03 )*X+1.91883866626681E-02 )*X-
               8.98046242565811E-02 )*X+3.13706645877886E-01
        weights2 = ((((((((-3.41688436990215E-08*X+5.07238960340773E-07)*X-
                     5.01675628408220E-06 )*X+4.20363420922845E-05 )*X-
                   3.08040221166823E-04 )*X+1.94431864731239E-03 )*X-
                 1.02477820460278E-02 )*X+4.28670143840073E-02 )*X-
               1.29314370962569E-01 )*X+2.22381034453369E-01
        weights3 = ((((((((( 4.99660550769508E-09*X-7.94585963310120E-08)*X+
                      8.359072409485E-07)*X-7.422369210610E-06)*X+
                    5.763374308160E-05)*X-3.86645606718233E-04 )*X+
                  2.18417516259781E-03 )*X-9.99791027771119E-03 )*X+
                3.48791097377370E-02 )*X-8.28299075413889E-02 )*X+1.01228536290376E-01
    elif X <= 5.0:
        Y = X-3.0E+00
        roots0 = (((((((((-1.48570633747284E-15*Y-1.33273068108777E-13)*Y+
                      4.068543696670E-12)*Y-9.163164161821E-11)*Y+
                    2.046819017845E-09)*Y-4.03076426299031E-08 )*Y+
                  7.29407420660149E-07 )*Y-1.23118059980833E-05 )*Y+
                1.88796581246938E-04 )*Y-2.53262912046853E-03 )*Y+2.51198234505021E-02
        roots1 = ((((((((( 1.35830583483312E-13*Y-2.29772605964836E-12)*Y-
                      3.821500128045E-12)*Y+6.844424214735E-10)*Y-
                    1.048063352259E-08)*Y+1.50083186233363E-08 )*Y+
                  3.48848942324454E-06 )*Y-1.08694174399193E-04 )*Y+
                2.08048885251999E-03 )*Y-2.91205805373793E-02 )*Y+2.72276489515713E-01
        roots2 = ((((((((( 5.02799392850289E-13*Y+1.07461812944084E-11)*Y-
                      1.482277886411E-10)*Y-2.153585661215E-09)*Y+
                    3.654087802817E-08)*Y+5.15929575830120E-07 )*Y-
                  9.52388379435709E-06 )*Y-2.16552440036426E-04 )*Y+
                9.03551469568320E-03 )*Y-1.45505469175613E-01 )*Y+1.21449092319186E+00
        roots3 = (((((((((-1.08510370291979E-12*Y+6.41492397277798E-11)*Y+
                      7.542387436125E-10)*Y-2.213111836647E-09)*Y-
                    1.448228963549E-07)*Y-1.95670833237101E-06 )*Y-
                  1.07481314670844E-05 )*Y+1.49335941252765E-04 )*Y+
                4.87791531990593E-02 )*Y-1.10559909038653E+00 )*Y+8.09502028611780E+00
        weights0 = ((((((((((-4.65801912689961E-14*Y+7.58669507106800E-13)*Y-
                       1.186387548048E-11)*Y+1.862334710665E-10)*Y-
                     2.799399389539E-09)*Y+4.148972684255E-08)*Y-
                   5.933568079600E-07)*Y+8.168349266115E-06)*Y-
                 1.08989176177409E-04 )*Y+1.41357961729531E-03 )*Y-
               1.87588361833659E-02 )*Y+2.89898651436026E-01
        weights1 = ((((((((((((-1.46345073267549E-14*Y+2.25644205432182E-13)*Y-
                         3.116258693847E-12)*Y+4.321908756610E-11)*Y-
                       5.673270062669E-10)*Y+7.006295962960E-09)*Y-
                     8.120186517000E-08)*Y+8.775294645770E-07)*Y-
                   8.77829235749024E-06 )*Y+8.04372147732379E-05 )*Y-
                 6.64149238804153E-04 )*Y+4.81181506827225E-03 )*Y-
               2.88982669486183E-02 )*Y+1.56247249979288E-01
        weights2 = ((((((((((((( 9.06812118895365E-15*Y-1.40541322766087E-13)*
                          Y+1.919270015269E-12)*Y-2.605135739010E-11)*Y+
                        3.299685839012E-10)*Y-3.86354139348735E-09 )*Y+
                      4.16265847927498E-08 )*Y-4.09462835471470E-07 )*Y+
                    3.64018881086111E-06 )*Y-2.88665153269386E-05 )*Y+
                  2.00515819789028E-04 )*Y-1.18791896897934E-03 )*Y+
                5.75223633388589E-03 )*Y-2.09400418772687E-02 )*Y+4.85368861938873E-02
        weights3 = ((((((((((((((-9.74835552342257E-16*Y+1.57857099317175E-14)*
                           Y-2.249993780112E-13)*Y+3.173422008953E-12)*Y-
                         4.161159459680E-11)*Y+5.021343560166E-10)*Y-
                       5.545047534808E-09)*Y+5.554146993491E-08)*Y-
                     4.99048696190133E-07 )*Y+3.96650392371311E-06 )*Y-
                   2.73816413291214E-05 )*Y+1.60106988333186E-04 )*Y-
                 7.64560567879592E-04 )*Y+2.81330044426892E-03 )*Y-
               7.16227030134947E-03 )*Y+9.66077262223353E-03
    elif X <= 10.0:
        Y = X-7.5E+00
        roots0 = ((((((((( 4.64217329776215E-15*Y-6.27892383644164E-15)*Y+
                      3.462236347446E-13)*Y-2.927229355350E-11)*Y+
                    5.090355371676E-10)*Y-9.97272656345253E-09 )*Y+
                  2.37835295639281E-07 )*Y-4.60301761310921E-06 )*Y+
                8.42824204233222E-05 )*Y-1.37983082233081E-03 )*Y+1.66630865869375E-02
        roots1 = ((((((((( 2.93981127919047E-14*Y+8.47635639065744E-13)*Y-
                      1.446314544774E-11)*Y-6.149155555753E-12)*Y+
                    8.484275604612E-10)*Y-6.10898827887652E-08 )*Y+
                  2.39156093611106E-06 )*Y-5.35837089462592E-05 )*Y+
                1.00967602595557E-03 )*Y-1.57769317127372E-02 )*Y+1.74853819464285E-01
        roots2 = (((((((((( 2.93523563363000E-14*Y-6.40041776667020E-14)*Y-
                       2.695740446312E-12)*Y+1.027082960169E-10)*Y-
                     5.822038656780E-10)*Y-3.159991002539E-08)*Y+
                   4.327249251331E-07)*Y+4.856768455119E-06)*Y-
                 2.54617989427762E-04 )*Y+5.54843378106589E-03 )*Y-
               7.95013029486684E-02 )*Y+7.20206142703162E-01
        roots3 = (((((((((((-1.62212382394553E-14*Y+7.68943641360593E-13)*Y+
                        5.764015756615E-12)*Y-1.380635298784E-10)*Y-
                      1.476849808675E-09)*Y+1.84347052385605E-08 )*Y+
                    3.34382940759405E-07 )*Y-1.39428366421645E-06 )*Y-
                  7.50249313713996E-05 )*Y-6.26495899187507E-04 )*Y+
                4.69716410901162E-02 )*Y-6.66871297428209E-01 )*Y+4.11207530217806E+00
        weights0 = ((((((((((-1.65995045235997E-15*Y+6.91838935879598E-14)*Y-
                       9.131223418888E-13)*Y+1.403341829454E-11)*Y-
                     3.672235069444E-10)*Y+6.366962546990E-09)*Y-
                   1.039220021671E-07)*Y+1.959098751715E-06)*Y-
                 3.33474893152939E-05 )*Y+5.72164211151013E-04 )*Y-
               1.05583210553392E-02 )*Y+2.26696066029591E-01
        weights1 = ((((((((((((-3.57248951192047E-16*Y+6.25708409149331E-15)*Y-
                         9.657033089714E-14)*Y+1.507864898748E-12)*Y-
                       2.332522256110E-11)*Y+3.428545616603E-10)*Y-
                     4.698730937661E-09)*Y+6.219977635130E-08)*Y-
                   7.83008889613661E-07 )*Y+9.08621687041567E-06 )*Y-
                 9.86368311253873E-05 )*Y+9.69632496710088E-04 )*Y-
               8.14594214284187E-03 )*Y+8.50218447733457E-02
        weights2 = ((((((((((((( 1.64742458534277E-16*Y-2.68512265928410E-15)*
                          Y+3.788890667676E-14)*Y-5.508918529823E-13)*Y+
                        7.555896810069E-12)*Y-9.69039768312637E-11 )*Y+
                      1.16034263529672E-09 )*Y-1.28771698573873E-08 )*Y+
                    1.31949431805798E-07 )*Y-1.23673915616005E-06 )*Y+
                  1.04189803544936E-05 )*Y-7.79566003744742E-05 )*Y+
                5.03162624754434E-04 )*Y-2.55138844587555E-03 )*Y+1.13250730954014E-02
        weights3 = ((((((((((((((-1.55714130075679E-17*Y+2.57193722698891E-16)*
                           Y-3.626606654097E-15)*Y+5.234734676175E-14)*Y-
                         7.067105402134E-13)*Y+8.793512664890E-12)*Y-
                       1.006088923498E-10)*Y+1.050565098393E-09)*Y-
                     9.91517881772662E-09 )*Y+8.35835975882941E-08 )*Y-
                   6.19785782240693E-07 )*Y+3.95841149373135E-06 )*Y-
                 2.11366761402403E-05 )*Y+9.00474771229507E-05 )*Y-
               2.78777909813289E-04 )*Y+5.26543779837487E-04
    elif X <= 15.0:
        Y = X-12.5E+00
        roots0 = ((((((((((( 4.94869622744119E-17*Y+8.03568805739160E-16)*Y-
                        5.599125915431E-15)*Y-1.378685560217E-13)*Y+
                      7.006511663249E-13)*Y+1.30391406991118E-11 )*Y+
                    8.06987313467541E-11 )*Y-5.20644072732933E-09 )*Y+
                  7.72794187755457E-08 )*Y-1.61512612564194E-06 )*Y+
                4.15083811185831E-05 )*Y-7.87855975560199E-04 )*Y+1.14189319050009E-02
        roots1 = ((((((((((( 4.89224285522336E-16*Y+1.06390248099712E-14)*Y-
                        5.446260182933E-14)*Y-1.613630106295E-12)*Y+
                      3.910179118937E-12)*Y+1.90712434258806E-10 )*Y+
                    8.78470199094761E-10 )*Y-5.97332993206797E-08 )*Y+
                  9.25750831481589E-07 )*Y-2.02362185197088E-05 )*Y+
                4.92341968336776E-04 )*Y-8.68438439874703E-03 )*Y+1.15825965127958E-01
        roots2 = (((((((((( 6.12419396208408E-14*Y+1.12328861406073E-13)*Y-
                       9.051094103059E-12)*Y-4.781797525341E-11)*Y+
                     1.660828868694E-09)*Y+4.499058798868E-10)*Y-
                   2.519549641933E-07)*Y+4.977444040180E-06)*Y-
                 1.25858350034589E-04 )*Y+2.70279176970044E-03 )*Y-
               3.99327850801083E-02 )*Y+4.33467200855434E-01
        roots3 = ((((((((((( 4.63414725924048E-14*Y-4.72757262693062E-14)*Y-
                        1.001926833832E-11)*Y+6.074107718414E-11)*Y+
                      1.576976911942E-09)*Y-2.01186401974027E-08 )*Y-
                    1.84530195217118E-07 )*Y+5.02333087806827E-06 )*Y+
                  9.66961790843006E-06 )*Y-1.58522208889528E-03 )*Y+
                2.80539673938339E-02 )*Y-2.78953904330072E-01 )*Y+1.82835655238235E+00
        weights3 = ((((((((((((( 2.90401781000996E-18*Y-4.63389683098251E-17)*
                          Y+6.274018198326E-16)*Y-8.936002188168E-15)*Y+
                        1.194719074934E-13)*Y-1.45501321259466E-12 )*Y+
                      1.64090830181013E-11 )*Y-1.71987745310181E-10 )*Y+
                    1.63738403295718E-09 )*Y-1.39237504892842E-08 )*Y+
                  1.06527318142151E-07 )*Y-7.27634957230524E-07 )*Y+
                4.12159381310339E-06 )*Y-1.74648169719173E-05 )*Y+8.50290130067818E-05
        weights2 = ((((((((((((-4.19569145459480E-17*Y+5.94344180261644E-16)*Y-
                         1.148797566469E-14)*Y+1.881303962576E-13)*Y-
                       2.413554618391E-12)*Y+3.372127423047E-11)*Y-
                     4.933988617784E-10)*Y+6.116545396281E-09)*Y-
                   6.69965691739299E-08 )*Y+7.52380085447161E-07 )*Y-
                 8.08708393262321E-06 )*Y+6.88603417296672E-05 )*Y-
               4.67067112993427E-04 )*Y+5.42313365864597E-03
        weights1 = ((((((((((-6.22272689880615E-15*Y+1.04126809657554E-13)*Y-
                       6.842418230913E-13)*Y+1.576841731919E-11)*Y-
                     4.203948834175E-10)*Y+6.287255934781E-09)*Y-
                   8.307159819228E-08)*Y+1.356478091922E-06)*Y-
                 2.08065576105639E-05 )*Y+2.52396730332340E-04 )*Y-
               2.94484050194539E-03 )*Y+6.01396183129168E-02
        weights0 = (((-1.8784686463512E-01/X+2.2991849164985E-01)/X -
                4.9893752514047E-01)/X-2.1916512131607E-05)*np.exp(-X) +np.sqrt(PIE4/X)-weights3-weights2-weights1
    elif X <= 20.0:
        weights0 = np.sqrt(PIE4/X)
        Y = X-17.5E+00
        roots0 = ((((((((((( 4.36701759531398E-17*Y-1.12860600219889E-16)*Y-
                        6.149849164164E-15)*Y+5.820231579541E-14)*Y+
                      4.396602872143E-13)*Y-1.24330365320172E-11 )*Y+
                    6.71083474044549E-11 )*Y+2.43865205376067E-10 )*Y+
                  1.67559587099969E-08 )*Y-9.32738632357572E-07 )*Y+
                2.39030487004977E-05 )*Y-4.68648206591515E-04 )*Y+8.34977776583956E-03
        roots1 = ((((((((((( 4.98913142288158E-16*Y-2.60732537093612E-16)*Y-
                        7.775156445127E-14)*Y+5.766105220086E-13)*Y+
                      6.432696729600E-12)*Y-1.39571683725792E-10 )*Y+
                    5.95451479522191E-10 )*Y+2.42471442836205E-09 )*Y+
                  2.47485710143120E-07 )*Y-1.14710398652091E-05 )*Y+
                2.71252453754519E-04 )*Y-4.96812745851408E-03 )*Y+8.26020602026780E-02
        roots2 = ((((((((((( 1.91498302509009E-15*Y+1.48840394311115E-14)*Y-
                        4.316925145767E-13)*Y+1.186495793471E-12)*Y+
                      4.615806713055E-11)*Y-5.54336148667141E-10 )*Y+
                    3.48789978951367E-10 )*Y-2.79188977451042E-09 )*Y+
                  2.09563208958551E-06 )*Y-6.76512715080324E-05 )*Y+
                1.32129867629062E-03 )*Y-2.05062147771513E-02 )*Y+2.88068671894324E-01
        roots3 = (((((((((((-5.43697691672942E-15*Y-1.12483395714468E-13)*Y+
                        2.826607936174E-12)*Y-1.266734493280E-11)*Y-
                      4.258722866437E-10)*Y+9.45486578503261E-09 )*Y-
                    5.86635622821309E-08 )*Y-1.28835028104639E-06 )*Y+
                  4.41413815691885E-05 )*Y-7.61738385590776E-04 )*Y+
                9.66090902985550E-03 )*Y-1.01410568057649E-01 )*Y+9.54714798156712E-01
        weights3 = ((((((((((((-7.56882223582704E-19*Y+7.53541779268175E-18)*Y-
                         1.157318032236E-16)*Y+2.411195002314E-15)*Y-
                       3.601794386996E-14)*Y+4.082150659615E-13)*Y-
                     4.289542980767E-12)*Y+5.086829642731E-11)*Y-
                   6.35435561050807E-10 )*Y+6.82309323251123E-09 )*Y-
                 5.63374555753167E-08 )*Y+3.57005361100431E-07 )*Y-
               2.40050045173721E-06 )*Y+4.94171300536397E-05
        weights2 = (((((((((((-5.54451040921657E-17*Y+2.68748367250999E-16)*Y+
                        1.349020069254E-14)*Y-2.507452792892E-13)*Y+
                      1.944339743818E-12)*Y-1.29816917658823E-11 )*Y+
                    3.49977768819641E-10 )*Y-8.67270669346398E-09 )*Y+
                  1.31381116840118E-07 )*Y-1.36790720600822E-06 )*Y+
                1.19210697673160E-05 )*Y-1.42181943986587E-04 )*Y+4.12615396191829E-03
        weights1 = (((((((((((-1.86506057729700E-16*Y+1.16661114435809E-15)*Y+
                        2.563712856363E-14)*Y-4.498350984631E-13)*Y+
                      1.765194089338E-12)*Y+9.04483676345625E-12 )*Y+
                    4.98930345609785E-10 )*Y-2.11964170928181E-08 )*Y+
                  3.98295476005614E-07 )*Y-5.49390160829409E-06 )*Y+
                7.74065155353262E-05 )*Y-1.48201933009105E-03 )*Y+4.97836392625268E-02
        weights0 = (( 1.9623264149430E-01/X-4.9695241464490E-01)/X -
               6.0156581186481E-05)*np.exp(-X)+weights0-weights1-weights2-weights3
    elif X <= 35.0:
        weights0 = np.sqrt(PIE4/X)
        E = np.exp(-X)
        roots0 = ((((((-4.45711399441838E-05*X+1.27267770241379E-03)*X -
                   2.36954961381262E-01)*X+1.54330657903756E+01)*X -
                 5.22799159267808E+02)*X+1.05951216669313E+04)*X +
               (-2.51177235556236E+06/X+8.72975373557709E+05)/X -
               1.29194382386499E+05)*E + R14/(X-R14)
        roots1 = (((((-7.85617372254488E-02*X+6.35653573484868E+00)*X -
                  3.38296938763990E+02)*X+1.25120495802096E+04)*X -
                3.16847570511637E+05)*X +
               ((-1.02427466127427E+09/X +
                 3.70104713293016E+08)/X-5.87119005093822E+07)/X +
               5.38614211391604E+06)*E + R24/(X-R24)
        roots2 = (((((-2.37900485051067E-01*X+1.84122184400896E+01)*X -
                  1.00200731304146E+03)*X+3.75151841595736E+04)*X -
                9.50626663390130E+05)*X +
               ((-2.88139014651985E+09/X +
                 1.06625915044526E+09)/X-1.72465289687396E+08)/X +
               1.60419390230055E+07)*E + R34/(X-R34)
        roots3 = ((((((-6.00691586407385E-04*X-3.64479545338439E-01)*X +
                   1.57496131755179E+01)*X-6.54944248734901E+02)*X +
                 1.70830039597097E+04)*X-2.90517939780207E+05)*X +
               (3.49059698304732E+07/X-1.64944522586065E+07)/X +
               2.96817940164703E+06)*E + R44/(X-R44)
        if X <= 25.0:
            weights3 = ((((((( 2.33766206773151E-07*X-
                          3.81542906607063E-05)*X +3.51416601267000E-03)*X-
                       1.66538571864728E-01)*X +4.80006136831847E+00)*X-
                     8.73165934223603E+01)*X +9.77683627474638E+02)*X +
                   1.66000945117640E+04/X -6.14479071209961E+03)*E + W44*weights0
        else:
            weights3 = (((((( 5.74245945342286E-06*X-
                         7.58735928102351E-05)*X +2.35072857922892E-04)*X-
                      3.78812134013125E-03)*X +3.09871652785805E-01)*X-
                    7.11108633061306E+00)*X +5.55297573149528E+01)*E + W44*weights0
        
        weights2 = (((((( 2.36392855180768E-04*X-9.16785337967013E-03)*X +
          4.62186525041313E-01)*X-1.96943786006540E+01)*X +
          4.99169195295559E+02)*X-6.21419845845090E+03)*X +
          ((+5.21445053212414E+07/X-1.34113464389309E+07)/X +
          1.13673298305631E+06)/X-2.81501182042707E+03)*E + W34*weights0
        weights1 = (((((( 7.29841848989391E-04*X-3.53899555749875E-02)*X +
                2.07797425718513E+00)*X-1.00464709786287E+02)*X +
                3.15206108877819E+03)*X-6.27054715090012E+04)*X +
            (+1.54721246264919E+07/X-5.26074391316381E+06)/X +
            7.67135400969617E+05)*E + W24*weights0
        weights0 = (( 1.9623264149430E-01/X-4.9695241464490E-01)/X -
            6.0156581186481E-05)*E + weights0-weights1-weights2-weights3
    elif X <= 53.0:
        weights0 = np.sqrt(PIE4/X)
        E = np.exp(-X)*(X*X)**2
        roots3 = ((-2.19135070169653E-03*X-1.19108256987623E-01)*X -
               7.50238795695573E-01)*E + R44/(X-R44)
        roots2 = ((-9.65842534508637E-04*X-4.49822013469279E-02)*X +
               6.08784033347757E-01)*E + R34/(X-R34)
        roots1 = ((-3.62569791162153E-04*X-9.09231717268466E-03)*X +
               1.84336760556262E-01)*E + R24/(X-R24)
        roots0 = ((-4.07557525914600E-05*X-6.88846864931685E-04)*X +
               1.74725309199384E-02)*E + R14/(X-R14)
        weights3 = (( 5.76631982000990E-06*X-7.89187283804890E-05)*X +
               3.28297971853126E-04)*E + W44*weights0
        weights2 = (( 2.08294969857230E-04*X-3.77489954837361E-03)*X +
               2.09857151617436E-02)*E + W34*weights0
        weights1 = (( 6.16374517326469E-04*X-1.26711744680092E-02)*X +
               8.14504890732155E-02)*E + W24*weights0
        weights0 = weights0-weights1-weights2-weights3
    else:
        weights0 = np.sqrt(PIE4/X)
        roots0 = R14/(X-R14)
        roots1 = R24/(X-R24)
        roots2 = R34/(X-R34)
        roots3 = R44/(X-R44)
        weights3 = W44*weights0
        weights2 = W34*weights0
        weights1 = W24*weights0
        weights0 = weights0-weights1-weights2-weights3
    
    roots[0] = roots0
    roots[1] = roots1
    roots[2] = roots2
    roots[3] = roots3
    weights[0] = weights0
    weights[1] = weights1
    weights[2] = weights2
    weights[3] = weights3
    
    return roots, weights#[roots1,roots2,roots3,roots[4]],[weights1,weights2,weights3,weights[4]]

def Root5Numba1(X,n, roots, weights):
    # roots = np.zeros((n))
    # weights = np.zeros((n))
    R15,PIE4 = 1.17581320211778E-01, 7.85398163397448E-01
    R25,W25 = 1.07456201243690E+00, 2.70967405960535E-01
    R35,W35 = 3.08593744371754E+00, 3.82231610015404E-02
    R45,W45 = 6.41472973366203E+00, 1.51614186862443E-03
    R55,W55 = 1.18071894899717E+01, 8.62130526143657E-06


    if X < 3.0E-7:
        roots0 = 2.26659266316985E-02 -2.15865967920897E-03 *X
        roots1 = 2.31271692140903E-01 -2.20258754389745E-02 *X
        roots2 = 8.57346024118836E-01 -8.16520023025515E-02 *X
        roots3 = 2.97353038120346E+00 -2.83193369647137E-01 *X
        roots4 = 1.84151859759051E+01 -1.75382723579439E+00 *X
        weights0 = 2.95524224714752E-01 -1.96867576909777E-02 *X
        weights1 = 2.69266719309995E-01 -5.61737590184721E-02 *X
        weights2 = 2.19086362515981E-01 -9.71152726793658E-02 *X
        weights3 = 1.49451349150580E-01 -1.02979262193565E-01 *X
        weights4 = 6.66713443086877E-02 -5.73782817488315E-02 *X
    elif X < 1.0:
        roots0 = ((((((-4.46679165328413E-11*X+1.21879111988031E-09)*X-
                   2.62975022612104E-08 )*X+5.15106194905897E-07 )*X-
                 9.27933625824749E-06 )*X+1.51794097682482E-04 )*X-
               2.15865967920301E-03 )*X+2.26659266316985E-02
        roots1 = (((((( 1.93117331714174E-10*X-4.57267589660699E-09)*X+
                   2.48339908218932E-08 )*X+1.50716729438474E-06 )*X-
                 6.07268757707381E-05 )*X+1.37506939145643E-03 )*X-
               2.20258754419939E-02 )*X+2.31271692140905E-01
        roots2 = ((((( 4.84989776180094E-09*X+1.31538893944284E-07)*X-
                  2.766753852879E-06)*X-7.651163510626E-05)*X+
                4.033058545972E-03)*X-8.16520022916145E-02 )*X+8.57346024118779E-01
        roots3 = ((((-2.48581772214623E-07*X-4.34482635782585E-06)*X-
                 7.46018257987630E-07 )*X+1.01210776517279E-02 )*X-
               2.83193369640005E-01 )*X+2.97353038120345E+00
        roots4 = (((((-8.92432153868554E-09*X+1.77288899268988E-08)*X+
                  3.040754680666E-06)*X+1.058229325071E-04)*X+
                4.596379534985E-02)*X-1.75382723579114E+00 )*X+1.84151859759049E+01
        weights0 = ((((((-2.03822632771791E-09*X+3.89110229133810E-08)*X-
                   5.84914787904823E-07 )*X+8.30316168666696E-06 )*X-
                 1.13218402310546E-04 )*X+1.49128888586790E-03 )*X-
               1.96867576904816E-02 )*X+2.95524224714749E-01
        weights1 = ((((((( 8.62848118397570E-09*X-1.38975551148989E-07)*X+
                    1.602894068228E-06)*X-1.646364300836E-05)*X+
                  1.538445806778E-04)*X-1.28848868034502E-03 )*X+
                9.38866933338584E-03 )*X-5.61737590178812E-02 )*X+2.69266719309991E-01
        weights2 = ((((((((-9.41953204205665E-09*X+1.47452251067755E-07)*X-
                     1.57456991199322E-06 )*X+1.45098401798393E-05 )*X-
                   1.18858834181513E-04 )*X+8.53697675984210E-04 )*X-
                 5.22877807397165E-03 )*X+2.60854524809786E-02 )*X-
               9.71152726809059E-02 )*X+2.19086362515979E-01
        weights3 = ((((((((-3.84961617022042E-08*X+5.66595396544470E-07)*X-
                     5.52351805403748E-06 )*X+4.53160377546073E-05 )*X-
                   3.22542784865557E-04 )*X+1.95682017370967E-03 )*X-
                 9.77232537679229E-03 )*X+3.79455945268632E-02 )*X-
               1.02979262192227E-01 )*X+1.49451349150573E-01
        weights4 = ((((((((( 4.09594812521430E-09*X-6.47097874264417E-08)*X+
                      6.743541482689E-07)*X-5.917993920224E-06)*X+
                    4.531969237381E-05)*X-2.99102856679638E-04 )*X+
                  1.65695765202643E-03 )*X-7.40671222520653E-03 )*X+
                2.50889946832192E-02 )*X-5.73782817487958E-02 )*X+6.66713443086877E-02
    elif X < 5.0:
        Y = X-3.0E+00
        roots0 = ((((((((-2.58163897135138E-14*Y+8.14127461488273E-13)*Y-
                     2.11414838976129E-11 )*Y+5.09822003260014E-10 )*Y-
                   1.16002134438663E-08 )*Y+2.46810694414540E-07 )*Y-
                 4.92556826124502E-06 )*Y+9.02580687971053E-05 )*Y-
               1.45190025120726E-03 )*Y+1.73416786387475E-02
        roots1 = ((((((((( 1.04525287289788E-14*Y+5.44611782010773E-14)*Y-
                      4.831059411392E-12)*Y+1.136643908832E-10)*Y-
                    1.104373076913E-09)*Y-2.35346740649916E-08 )*Y+
                  1.43772622028764E-06 )*Y-4.23405023015273E-05 )*Y+
                9.12034574793379E-04 )*Y-1.52479441718739E-02 )*Y+1.76055265928744E-01
        roots2 = (((((((((-6.89693150857911E-14*Y+5.92064260918861E-13)*Y+
                      1.847170956043E-11)*Y-3.390752744265E-10)*Y-
                    2.995532064116E-09)*Y+1.57456141058535E-07 )*Y-
                  3.95859409711346E-07 )*Y-9.58924580919747E-05 )*Y+
                3.23551502557785E-03 )*Y-5.97587007636479E-02 )*Y+6.46432853383057E-01
        roots3 = ((((((((-3.61293809667763E-12*Y-2.70803518291085E-11)*Y+
                     8.83758848468769E-10 )*Y+1.59166632851267E-08 )*Y-
                   1.32581997983422E-07 )*Y-7.60223407443995E-06 )*Y-
                 7.41019244900952E-05 )*Y+9.81432631743423E-03 )*Y-
               2.23055570487771E-01 )*Y+2.21460798080643E+00
        roots4 = ((((((((( 7.12332088345321E-13*Y+3.16578501501894E-12)*Y-
                      8.776668218053E-11)*Y-2.342817613343E-09)*Y-
                    3.496962018025E-08)*Y-3.03172870136802E-07 )*Y+
                  1.50511293969805E-06 )*Y+1.37704919387696E-04 )*Y+
                4.70723869619745E-02 )*Y-1.47486623003693E+00 )*Y+1.35704792175847E+01
        weights0 = ((((((((( 1.04348658616398E-13*Y-1.94147461891055E-12)*Y+
                      3.485512360993E-11)*Y-6.277497362235E-10)*Y+
                    1.100758247388E-08)*Y-1.88329804969573E-07 )*Y+
                  3.12338120839468E-06 )*Y-5.04404167403568E-05 )*Y+
                8.00338056610995E-04 )*Y-1.30892406559521E-02 )*Y+2.47383140241103E-01
        weights1 = ((((((((((( 3.23496149760478E-14*Y-5.24314473469311E-13)*Y+
                        7.743219385056E-12)*Y-1.146022750992E-10)*Y+
                      1.615238462197E-09)*Y-2.15479017572233E-08 )*Y+
                    2.70933462557631E-07 )*Y-3.18750295288531E-06 )*Y+
                  3.47425221210099E-05 )*Y-3.45558237388223E-04 )*Y+
                3.05779768191621E-03 )*Y-2.29118251223003E-02 )*Y+1.59834227924213E-01
        weights2 = ((((((((((((-3.42790561802876E-14*Y+5.26475736681542E-13)*Y-
                         7.184330797139E-12)*Y+9.763932908544E-11)*Y-
                       1.244014559219E-09)*Y+1.472744068942E-08)*Y-
                     1.611749975234E-07)*Y+1.616487851917E-06)*Y-
                   1.46852359124154E-05 )*Y+1.18900349101069E-04 )*Y-
                 8.37562373221756E-04 )*Y+4.93752683045845E-03 )*Y-
               2.25514728915673E-02 )*Y+6.95211812453929E-02
        weights3 = ((((((((((((( 1.04072340345039E-14*Y-1.60808044529211E-13)*
                          Y+2.183534866798E-12)*Y-2.939403008391E-11)*Y+
                        3.679254029085E-10)*Y-4.23775673047899E-09 )*Y+
                      4.46559231067006E-08 )*Y-4.26488836563267E-07 )*Y+
                    3.64721335274973E-06 )*Y-2.74868382777722E-05 )*Y+
                  1.78586118867488E-04 )*Y-9.68428981886534E-04 )*Y+
                4.16002324339929E-03 )*Y-1.28290192663141E-02 )*Y+2.22353727685016E-02
        weights4 = ((((((((((((((-8.16770412525963E-16*Y+1.31376515047977E-14)*
                           Y-1.856950818865E-13)*Y+2.596836515749E-12)*Y-
                         3.372639523006E-11)*Y+4.025371849467E-10)*Y-
                       4.389453269417E-09)*Y+4.332753856271E-08)*Y-
                     3.82673275931962E-07 )*Y+2.98006900751543E-06 )*Y-
                   2.00718990300052E-05 )*Y+1.13876001386361E-04 )*Y-
                 5.23627942443563E-04 )*Y+1.83524565118203E-03 )*Y-
               4.37785737450783E-03 )*Y+5.36963805223095E-03
    elif X < 10.0:
        Y = X-7.5E+00
        roots0 = ((((((((-1.13825201010775E-14*Y+1.89737681670375E-13)*Y-
                     4.81561201185876E-12 )*Y+1.56666512163407E-10 )*Y-
                   3.73782213255083E-09 )*Y+9.15858355075147E-08 )*Y-
                 2.13775073585629E-06 )*Y+4.56547356365536E-05 )*Y-
               8.68003909323740E-04 )*Y+1.22703754069176E-02
        roots1 = (((((((((-3.67160504428358E-15*Y+1.27876280158297E-14)*Y-
                      1.296476623788E-12)*Y+1.477175434354E-11)*Y+
                    5.464102147892E-10)*Y-2.42538340602723E-08 )*Y+
                  8.20460740637617E-07 )*Y-2.20379304598661E-05 )*Y+
                4.90295372978785E-04 )*Y-9.14294111576119E-03 )*Y+1.22590403403690E-01
        roots2 = ((((((((( 1.39017367502123E-14*Y-6.96391385426890E-13)*Y+
                      1.176946020731E-12)*Y+1.725627235645E-10)*Y-
                    3.686383856300E-09)*Y+2.87495324207095E-08 )*Y+
                  1.71307311000282E-06 )*Y-7.94273603184629E-05 )*Y+
                2.00938064965897E-03 )*Y-3.63329491677178E-02 )*Y+4.34393683888443E-01
        roots3 = ((((((((((-1.27815158195209E-14*Y+1.99910415869821E-14)*Y+
                       3.753542914426E-12)*Y-2.708018219579E-11)*Y-
                     1.190574776587E-09)*Y+1.106696436509E-08)*Y+
                   3.954955671326E-07)*Y-4.398596059588E-06)*Y-
                 2.01087998907735E-04 )*Y+7.89092425542937E-03 )*Y-
               1.42056749162695E-01 )*Y+1.39964149420683E+00
        roots4 = ((((((((((-1.19442341030461E-13*Y-2.34074833275956E-12)*Y+
                       6.861649627426E-12)*Y+6.082671496226E-10)*Y+
                     5.381160105420E-09)*Y-6.253297138700E-08)*Y-
                   2.135966835050E-06)*Y-2.373394341886E-05)*Y+
                 2.88711171412814E-06 )*Y+4.85221195290753E-02 )*Y-
               1.04346091985269E+00 )*Y+7.89901551676692E+00
        weights0 = ((((((((( 7.95526040108997E-15*Y-2.48593096128045E-13)*Y+
                      4.761246208720E-12)*Y-9.535763686605E-11)*Y+
                    2.225273630974E-09)*Y-4.49796778054865E-08 )*Y+
                  9.17812870287386E-07 )*Y-1.86764236490502E-05 )*Y+
                3.76807779068053E-04 )*Y-8.10456360143408E-03 )*Y+2.01097936411496E-01
        weights1 = ((((((((((( 1.25678686624734E-15*Y-2.34266248891173E-14)*Y+
                        3.973252415832E-13)*Y-6.830539401049E-12)*Y+
                      1.140771033372E-10)*Y-1.82546185762009E-09 )*Y+
                    2.77209637550134E-08 )*Y-4.01726946190383E-07 )*Y+
                  5.48227244014763E-06 )*Y-6.95676245982121E-05 )*Y+
                8.05193921815776E-04 )*Y-8.15528438784469E-03 )*Y+9.71769901268114E-02
        weights2 = ((((((((((((-8.20929494859896E-16*Y+1.37356038393016E-14)*Y-
                         2.022863065220E-13)*Y+3.058055403795E-12)*Y-
                       4.387890955243E-11)*Y+5.923946274445E-10)*Y-
                     7.503659964159E-09)*Y+8.851599803902E-08)*Y-
                   9.65561998415038E-07 )*Y+9.60884622778092E-06 )*Y-
                 8.56551787594404E-05 )*Y+6.66057194311179E-04 )*Y-
               4.17753183902198E-03 )*Y+2.25443826852447E-02
        weights3 = ((((((((((((((-1.08764612488790E-17*Y+1.85299909689937E-16)*
                           Y-2.730195628655E-15)*Y+4.127368817265E-14)*Y-
                         5.881379088074E-13)*Y+7.805245193391E-12)*Y-
                       9.632707991704E-11)*Y+1.099047050624E-09)*Y-
                     1.15042731790748E-08 )*Y+1.09415155268932E-07 )*Y-
                   9.33687124875935E-07 )*Y+7.02338477986218E-06 )*Y-
                 4.53759748787756E-05 )*Y+2.41722511389146E-04 )*Y-
               9.75935943447037E-04 )*Y+2.57520532789644E-03
        weights4 = ((((((((((((((( 7.28996979748849E-19*Y-1.26518146195173E-17)
                            *Y+1.886145834486E-16)*Y-2.876728287383E-15)*Y+
                          4.114588668138E-14)*Y-5.44436631413933E-13 )*Y+
                        6.64976446790959E-12 )*Y-7.44560069974940E-11 )*Y+
                      7.57553198166848E-10 )*Y-6.92956101109829E-09 )*Y+
                    5.62222859033624E-08 )*Y-3.97500114084351E-07 )*Y+
                  2.39039126138140E-06 )*Y-1.18023950002105E-05 )*Y+
                4.52254031046244E-05 )*Y-1.21113782150370E-04 )*Y+1.75013126731224E-04
    elif X < 15.0:
        Y = X-12.5E+00
        roots0 = ((((((((((-4.16387977337393E-17*Y+7.20872997373860E-16)*Y+
                       1.395993802064E-14)*Y+3.660484641252E-14)*Y-
                     4.154857548139E-12)*Y+2.301379846544E-11)*Y-
                   1.033307012866E-09)*Y+3.997777641049E-08)*Y-
                 9.35118186333939E-07 )*Y+2.38589932752937E-05 )*Y-
               5.35185183652937E-04 )*Y+8.85218988709735E-03
        roots1 = ((((((((((-4.56279214732217E-16*Y+6.24941647247927E-15)*Y+
                       1.737896339191E-13)*Y+8.964205979517E-14)*Y-
                     3.538906780633E-11)*Y+9.561341254948E-11)*Y-
                   9.772831891310E-09)*Y+4.240340194620E-07)*Y-
                 1.02384302866534E-05 )*Y+2.57987709704822E-04 )*Y-
               5.54735977651677E-03 )*Y+8.68245143991948E-02
        roots2 = ((((((((((-2.52879337929239E-15*Y+2.13925810087833E-14)*Y+
                       7.884307667104E-13)*Y-9.023398159510E-13)*Y-
                     5.814101544957E-11)*Y-1.333480437968E-09)*Y-
                   2.217064940373E-08)*Y+1.643290788086E-06)*Y-
                 4.39602147345028E-05 )*Y+1.08648982748911E-03 )*Y-
               2.13014521653498E-02 )*Y+2.94150684465425E-01
        roots3 = ((((((((((-6.42391438038888E-15*Y+5.37848223438815E-15)*Y+
                       8.960828117859E-13)*Y+5.214153461337E-11)*Y-
                     1.106601744067E-10)*Y-2.007890743962E-08)*Y+
                   1.543764346501E-07)*Y+4.520749076914E-06)*Y-
                 1.88893338587047E-04 )*Y+4.73264487389288E-03 )*Y-
               7.91197893350253E-02 )*Y+8.60057928514554E-01
        roots4 = (((((((((((-2.24366166957225E-14*Y+4.87224967526081E-14)*Y+
                        5.587369053655E-12)*Y-3.045253104617E-12)*Y-
                      1.223983883080E-09)*Y-2.05603889396319E-09 )*Y+
                    2.58604071603561E-07 )*Y+1.34240904266268E-06 )*Y-
                  5.72877569731162E-05 )*Y-9.56275105032191E-04 )*Y+
                4.23367010370921E-02 )*Y-5.76800927133412E-01 )*Y+3.87328263873381E+00
        weights0 = ((((((((( 8.98007931950169E-15*Y+7.25673623859497E-14)*Y+
                      5.851494250405E-14)*Y-4.234204823846E-11)*Y+
                    3.911507312679E-10)*Y-9.65094802088511E-09 )*Y+
                  3.42197444235714E-07 )*Y-7.51821178144509E-06 )*Y+
                1.94218051498662E-04 )*Y-5.38533819142287E-03 )*Y+1.68122596736809E-01
        weights1 = ((((((((((-1.05490525395105E-15*Y+1.96855386549388E-14)*Y-
                       5.500330153548E-13)*Y+1.003849567976E-11)*Y-
                     1.720997242621E-10)*Y+3.533277061402E-09)*Y-
                   6.389171736029E-08)*Y+1.046236652393E-06)*Y-
                 1.73148206795827E-05 )*Y+2.57820531617185E-04 )*Y-
               3.46188265338350E-03 )*Y+7.03302497508176E-02
        weights2 = ((((((((((( 3.60020423754545E-16*Y-6.24245825017148E-15)*Y+
                        9.945311467434E-14)*Y-1.749051512721E-12)*Y+
                      2.768503957853E-11)*Y-4.08688551136506E-10 )*Y+
                    6.04189063303610E-09 )*Y-8.23540111024147E-08 )*Y+
                  1.01503783870262E-06 )*Y-1.20490761741576E-05 )*Y+
                1.26928442448148E-04 )*Y-1.05539461930597E-03 )*Y+1.15543698537013E-02
        weights3 = ((((((((((((( 2.51163533058925E-18*Y-4.31723745510697E-17)*
                          Y+6.557620865832E-16)*Y-1.016528519495E-14)*Y+
                        1.491302084832E-13)*Y-2.06638666222265E-12 )*Y+
                      2.67958697789258E-11 )*Y-3.23322654638336E-10 )*Y+
                    3.63722952167779E-09 )*Y-3.75484943783021E-08 )*Y+
                  3.49164261987184E-07 )*Y-2.92658670674908E-06 )*Y+
                2.12937256719543E-05 )*Y-1.19434130620929E-04 )*Y+6.45524336158384E-04
        weights4 = ((((((((((((((-1.29043630202811E-19*Y+2.16234952241296E-18)*
                           Y-3.107631557965E-17)*Y+4.570804313173E-16)*Y-
                         6.301348858104E-15)*Y+8.031304476153E-14)*Y-
                       9.446196472547E-13)*Y+1.018245804339E-11)*Y-
                     9.96995451348129E-11 )*Y+8.77489010276305E-10 )*Y-
                   6.84655877575364E-09 )*Y+4.64460857084983E-08 )*Y-
                 2.66924538268397E-07 )*Y+1.24621276265907E-06 )*Y-
               4.30868944351523E-06 )*Y+9.94307982432868E-06
    elif X < 20.0:
        Y = X-17.5E+00
        roots0 = (((((((((( 1.91875764545740E-16*Y+7.8357401095707E-16)*Y-
                       3.260875931644E-14)*Y-1.186752035569E-13)*Y+
                     4.275180095653E-12)*Y+3.357056136731E-11)*Y-
                   1.123776903884E-09)*Y+1.231203269887E-08)*Y-
                 3.99851421361031E-07 )*Y+1.45418822817771E-05 )*Y-
               3.49912254976317E-04 )*Y+6.67768703938812E-03
        roots1 = (((((((((( 2.02778478673555E-15*Y+1.01640716785099E-14)*Y-
                       3.385363492036E-13)*Y-1.615655871159E-12)*Y+
                     4.527419140333E-11)*Y+3.853670706486E-10)*Y-
                   1.184607130107E-08)*Y+1.347873288827E-07)*Y-
                 4.47788241748377E-06 )*Y+1.54942754358273E-04 )*Y-
               3.55524254280266E-03 )*Y+6.44912219301603E-02
        roots2 = (((((((((( 7.79850771456444E-15*Y+6.00464406395001E-14)*Y-
                       1.249779730869E-12)*Y-1.020720636353E-11)*Y+
                     1.814709816693E-10)*Y+1.766397336977E-09)*Y-
                   4.603559449010E-08)*Y+5.863956443581E-07)*Y-
                 2.03797212506691E-05 )*Y+6.31405161185185E-04 )*Y-
               1.30102750145071E-02 )*Y+2.10244289044705E-01
        roots3 = (((((((((((-2.92397030777912E-15*Y+1.94152129078465E-14)*Y+
                        4.859447665850E-13)*Y-3.217227223463E-12)*Y-
                      7.484522135512E-11)*Y+7.19101516047753E-10 )*Y+
                    6.88409355245582E-09 )*Y-1.44374545515769E-07 )*Y+
                  2.74941013315834E-06 )*Y-1.02790452049013E-04 )*Y+
                2.59924221372643E-03 )*Y-4.35712368303551E-02 )*Y+5.62170709585029E-01
        roots4 = ((((((((((( 1.17976126840060E-14*Y+1.24156229350669E-13)*Y-
                        3.892741622280E-12)*Y-7.755793199043E-12)*Y+
                      9.492190032313E-10)*Y-4.98680128123353E-09 )*Y-
                    1.81502268782664E-07 )*Y+2.69463269394888E-06 )*Y+
                  2.50032154421640E-05 )*Y-1.33684303917681E-03 )*Y+
                2.29121951862538E-02 )*Y-2.45653725061323E-01 )*Y+1.89999883453047E+00
        weights0 = (((((((((( 1.74841995087592E-15*Y-6.95671892641256E-16)*Y-
                       3.000659497257E-13)*Y+2.021279817961E-13)*Y+
                     3.853596935400E-11)*Y+1.461418533652E-10)*Y-
                   1.014517563435E-08)*Y+1.132736008979E-07)*Y-
                 2.86605475073259E-06 )*Y+1.21958354908768E-04 )*Y-
               3.86293751153466E-03 )*Y+1.45298342081522E-01
        weights1 = ((((((((((-1.11199320525573E-15*Y+1.85007587796671E-15)*Y+
                       1.220613939709E-13)*Y+1.275068098526E-12)*Y-
                     5.341838883262E-11)*Y+6.161037256669E-10)*Y-
                   1.009147879750E-08)*Y+2.907862965346E-07)*Y-
                 6.12300038720919E-06 )*Y+1.00104454489518E-04 )*Y-
               1.80677298502757E-03 )*Y+5.78009914536630E-02
        weights2 = ((((((((((-9.49816486853687E-16*Y+6.67922080354234E-15)*Y+
                       2.606163540537E-15)*Y+1.983799950150E-12)*Y-
                     5.400548574357E-11)*Y+6.638043374114E-10)*Y-
                   8.799518866802E-09)*Y+1.791418482685E-07)*Y-
                 2.96075397351101E-06 )*Y+3.38028206156144E-05 )*Y-
               3.58426847857878E-04 )*Y+8.39213709428516E-03
        weights3 = ((((((((((( 1.33829971060180E-17*Y-3.44841877844140E-16)*Y+
                        4.745009557656E-15)*Y-6.033814209875E-14)*Y+
                      1.049256040808E-12)*Y-1.70859789556117E-11 )*Y+
                    2.15219425727959E-10 )*Y-2.52746574206884E-09 )*Y+
                  3.27761714422960E-08 )*Y-3.90387662925193E-07 )*Y+
                3.46340204593870E-06 )*Y-2.43236345136782E-05 )*Y+3.54846978585226E-04
        weights4 = ((((((((((((( 2.69412277020887E-20*Y-4.24837886165685E-19)*
                          Y+6.030500065438E-18)*Y-9.069722758289E-17)*Y+
                        1.246599177672E-15)*Y-1.56872999797549E-14 )*Y+
                      1.87305099552692E-13 )*Y-2.09498886675861E-12 )*Y+
                    2.11630022068394E-11 )*Y-1.92566242323525E-10 )*Y+
                  1.62012436344069E-09 )*Y-1.23621614171556E-08 )*Y+
                7.72165684563049E-08 )*Y-3.59858901591047E-07 )*Y+2.43682618601000E-06
    elif X < 25.0:
        Y = X-22.5E+00
        roots0 = (((((((((-1.13927848238726E-15*Y+7.39404133595713E-15)*Y+
                      1.445982921243E-13)*Y-2.676703245252E-12)*Y+
                    5.823521627177E-12)*Y+2.17264723874381E-10 )*Y+
                  3.56242145897468E-09 )*Y-3.03763737404491E-07 )*Y+
                9.46859114120901E-06 )*Y-2.30896753853196E-04 )*Y+5.24663913001114E-03
        roots1 = (((((((((( 2.89872355524581E-16*Y-1.22296292045864E-14)*Y+
                       6.184065097200E-14)*Y+1.649846591230E-12)*Y-
                     2.729713905266E-11)*Y+3.709913790650E-11)*Y+
                   2.216486288382E-09)*Y+4.616160236414E-08)*Y-
                 3.32380270861364E-06 )*Y+9.84635072633776E-05 )*Y-
               2.30092118015697E-03 )*Y+5.00845183695073E-02
        roots2 = (((((((((( 1.97068646590923E-15*Y-4.89419270626800E-14)*Y+
                       1.136466605916E-13)*Y+7.546203883874E-12)*Y-
                     9.635646767455E-11)*Y-8.295965491209E-11)*Y+
                   7.534109114453E-09)*Y+2.699970652707E-07)*Y-
                 1.42982334217081E-05 )*Y+3.78290946669264E-04 )*Y-
               8.03133015084373E-03 )*Y+1.58689469640791E-01
        roots3 = (((((((((( 1.33642069941389E-14*Y-1.55850612605745E-13)*Y-
                       7.522712577474E-13)*Y+3.209520801187E-11)*Y-
                     2.075594313618E-10)*Y-2.070575894402E-09)*Y+
                   7.323046997451E-09)*Y+1.851491550417E-06)*Y-
                 6.37524802411383E-05 )*Y+1.36795464918785E-03 )*Y-
               2.42051126993146E-02 )*Y+3.97847167557815E-01
        roots4 = ((((((((((-6.07053986130526E-14*Y+1.04447493138843E-12)*Y-
                       4.286617818951E-13)*Y-2.632066100073E-10)*Y+
                     4.804518986559E-09)*Y-1.835675889421E-08)*Y-
                   1.068175391334E-06)*Y+3.292234974141E-05)*Y-
                 5.94805357558251E-04 )*Y+8.29382168612791E-03 )*Y-
               9.93122509049447E-02 )*Y+1.09857804755042E+00
        weights0 = (((((((((-9.10338640266542E-15*Y+1.00438927627833E-13)*Y+
                      7.817349237071E-13)*Y-2.547619474232E-11)*Y+
                    1.479321506529E-10)*Y+1.52314028857627E-09 )*Y+
                  9.20072040917242E-09 )*Y-2.19427111221848E-06 )*Y+
                8.65797782880311E-05 )*Y-2.82718629312875E-03 )*Y+1.28718310443295E-01
        weights1 = ((((((((( 5.52380927618760E-15*Y-6.43424400204124E-14)*Y-
                      2.358734508092E-13)*Y+8.261326648131E-12)*Y+
                    9.229645304956E-11)*Y-5.68108973828949E-09 )*Y+
                  1.22477891136278E-07 )*Y-2.11919643127927E-06 )*Y+
                4.23605032368922E-05 )*Y-1.14423444576221E-03 )*Y+5.06607252890186E-02
        weights2 = ((((((((( 3.99457454087556E-15*Y-5.11826702824182E-14)*Y-
                      4.157593182747E-14)*Y+4.214670817758E-12)*Y+
                    6.705582751532E-11)*Y-3.36086411698418E-09 )*Y+
                  6.07453633298986E-08 )*Y-7.40736211041247E-07 )*Y+
                8.84176371665149E-06 )*Y-1.72559275066834E-04 )*Y+7.16639814253567E-03
        weights3 = (((((((((((-2.14649508112234E-18*Y-2.45525846412281E-18)*Y+
                        6.126212599772E-16)*Y-8.526651626939E-15)*Y+
                      4.826636065733E-14)*Y-3.39554163649740E-13 )*Y+
                    1.67070784862985E-11 )*Y-4.42671979311163E-10 )*Y+
                  6.77368055908400E-09 )*Y-7.03520999708859E-08 )*Y+
                6.04993294708874E-07 )*Y-7.80555094280483E-06 )*Y+2.85954806605017E-04
        weights4 = ((((((((((((-5.63938733073804E-21*Y+6.92182516324628E-20)*Y-
                         1.586937691507E-18)*Y+3.357639744582E-17)*Y-
                       4.810285046442E-16)*Y+5.386312669975E-15)*Y-
                     6.117895297439E-14)*Y+8.441808227634E-13)*Y-
                   1.18527596836592E-11 )*Y+1.36296870441445E-10 )*Y-
                 1.17842611094141E-09 )*Y+7.80430641995926E-09 )*Y-
               5.97767417400540E-08 )*Y+1.65186146094969E-06
    elif X < 40.0:
        weights0 = np.sqrt(PIE4/X)
        E = np.exp(-X)
        roots0 = ((((((((-1.73363958895356E-06*X+1.19921331441483E-04)*X -
                     1.59437614121125E-02)*X+1.13467897349442E+00)*X -
                   4.47216460864586E+01)*X+1.06251216612604E+03)*X -
                 1.52073917378512E+04)*X+1.20662887111273E+05)*X -
               4.07186366852475E+05)*E + R15/(X-R15)
        roots1 = ((((((((-1.60102542621710E-05*X+1.10331262112395E-03)*X -
                     1.50043662589017E-01)*X+1.05563640866077E+01)*X -
                   4.10468817024806E+02)*X+9.62604416506819E+03)*X -
                 1.35888069838270E+05)*X+1.06107577038340E+06)*X -
               3.51190792816119E+06)*E + R25/(X-R25)
        roots2 = ((((((((-4.48880032128422E-05*X+2.69025112122177E-03)*X -
                     4.01048115525954E-01)*X+2.78360021977405E+01)*X -
                   1.04891729356965E+03)*X+2.36985942687423E+04)*X -
                 3.19504627257548E+05)*X+2.34879693563358E+06)*X -
               7.16341568174085E+06)*E + R35/(X-R35)
        roots3 = ((((((((-6.38526371092582E-05*X-2.29263585792626E-03)*X -
                     7.65735935499627E-02)*X+9.12692349152792E+00)*X -
                   2.32077034386717E+02)*X+2.81839578728845E+02)*X +
                 9.59529683876419E+04)*X-1.77638956809518E+06)*X +
               1.02489759645410E+07)*E + R45/(X-R45)
        roots4 = ((((((((-3.59049364231569E-05*X-2.25963977930044E-02)*X +
                     1.12594870794668E+00)*X-4.56752462103909E+01)*X +
                   1.05804526830637E+03)*X-1.16003199605875E+04)*X -
                 4.07297627297272E+04)*X+2.22215528319857E+06)*X -
               1.61196455032613E+07)*E + R55/(X-R55)
        weights4 = (((((((((-4.61100906133970E-10*X+1.43069932644286E-07)*X -
                      1.63960915431080E-05)*X+1.15791154612838E-03)*X -
                    5.30573476742071E-02)*X+1.61156533367153E+00)*X -
                  3.23248143316007E+01)*X+4.12007318109157E+02)*X -
                3.02260070158372E+03)*X+9.71575094154768E+03)*E + W55*weights0
        weights3 = (((((((((-2.40799435809950E-08*X+8.12621667601546E-06)*X -
                      9.04491430884113E-04)*X+6.37686375770059E-02)*X -
                    2.96135703135647E+00)*X+9.15142356996330E+01)*X -
                  1.86971865249111E+03)*X+2.42945528916947E+04)*X -
                1.81852473229081E+05)*X+5.96854758661427E+05)*E + W45*weights0
        weights2 = (((((((( 1.83574464457207E-05*X-1.54837969489927E-03)*X +
                     1.18520453711586E-01)*X-6.69649981309161E+00)*X +
                   2.44789386487321E+02)*X-5.68832664556359E+03)*X +
                 8.14507604229357E+04)*X-6.55181056671474E+05)*X +
               2.26410896607237E+06)*E + W35*weights0
        weights1 = (((((((( 2.77778345870650E-05*X-2.22835017655890E-03)*X +
                     1.61077633475573E-01)*X-8.96743743396132E+00)*X +
                   3.28062687293374E+02)*X-7.65722701219557E+03)*X +
                 1.10255055017664E+05)*X-8.92528122219324E+05)*X +
               3.10638627744347E+06)*E + W25*weights0
        weights0 = weights0-0.01962E+00*E-weights1-weights2-weights3-weights4
    elif X < 59.0:
        weights0 = np.sqrt(PIE4/X)
        XXX = X**3
        E = XXX*np.exp(-X)
        roots0 = (((-2.43758528330205E-02*X+2.07301567989771E+00)*X -
                6.45964225381113E+01)*X+7.14160088655470E+02)*E + R15/(X-R15)
        roots1 = (((-2.28861955413636E-01*X+1.93190784733691E+01)*X -
                5.99774730340912E+02)*X+6.61844165304871E+03)*E + R25/(X-R25)
        roots2 = (((-6.95053039285586E-01*X+5.76874090316016E+01)*X -
                1.77704143225520E+03)*X+1.95366082947811E+04)*E + R35/(X-R35)
        roots3 = (((-1.58072809087018E+00*X+1.27050801091948E+02)*X -
                3.86687350914280E+03)*X+4.23024828121420E+04)*E + R45/(X-R45)
        roots4 = (((-3.33963830405396E+00*X+2.51830424600204E+02)*X -
                7.57728527654961E+03)*X+8.21966816595690E+04)*E + R55/(X-R55)
        E = XXX*E
        weights4 = (( 1.35482430510942E-08*X-3.27722199212781E-07)*X +
               2.41522703684296E-06)*E + W55*weights0
        weights3 = (( 1.23464092261605E-06*X-3.55224564275590E-05)*X +
               3.03274662192286E-04)*E + W45*weights0
        weights2 = (( 1.34547929260279E-05*X-4.19389884772726E-04)*X +
               3.87706687610809E-03)*E + W35*weights0
        weights1 = (( 2.09539509123135E-05*X-6.87646614786982E-04)*X +
               6.68743788585688E-03)*E + W25*weights0
        weights0 = weights0-weights1-weights2-weights3-weights4
    else:
        weights0 = np.sqrt(PIE4/X)
        roots0 = R15/(X-R15)
        roots1 = R25/(X-R25)
        roots2 = R35/(X-R35)
        roots3 = R45/(X-R45)
        roots4 = R55/(X-R55)
        weights1 = W25*weights0
        weights2 = W35*weights0
        weights3 = W45*weights0
        weights4 = W55*weights0
        weights0 = weights0-weights1-weights2-weights3-weights4

    roots[0] = roots0
    roots[1] = roots1
    roots[2] = roots2
    roots[3] = roots3
    roots[4] = roots4
    weights[0] = weights0
    weights[1] = weights1
    weights[2] = weights2
    weights[3] = weights3
    weights[4] = weights4
    
    return  roots, weights#[roots1,roots2,roots3,roots4,roots[5]],[weights1,weights2,weights3,weights4,weights[5]]

def eri_4c2e_diagonal_numba1(bfs_coords, bfs_contr_prim_norms, bfs_lmn, bfs_nprim, bfs_coeffs, bfs_prim_norms, bfs_expnts):
    # This function calculates the "diagonal" elements of the 4c2e ERI array
    # Used to implement Schwarz screening
    # http://vergil.chemistry.gatech.edu/notes/df.pdf

       

    # returns a 2D array whose elements are given as A[i,j] = (ij|ij) 
    nao = bfs_coords.shape[0]
    fourC2E_diag = np.zeros((nao, nao),dtype=np.float64) 
        
    pi = 3.141592653589793
    pisq = 9.869604401089358  #PI^2
    twopisq = 19.739208802178716  #2*PI^2

    #Loop pver BFs
    for i in prange(0, nao): #A
        I = bfs_coords[i]
        Ni = bfs_contr_prim_norms[i]
        lmni = bfs_lmn[i]
        la, ma, na = lmni
        nprimi = bfs_nprim[i]

        K = I
        Nk = Ni
        lc, mc, nc = lmni
        
        nprimk = nprimi
        
        for j in range(0, i+1): #B
            J = bfs_coords[j]
            IJ = I - J
            IJsq = np.sum(IJ**2)
            Nj = bfs_contr_prim_norms[j]
            lmnj = bfs_lmn[j]
            lb, mb, nb = lmnj
            tempcoeff1 = Ni*Nj
            nprimj = bfs_nprim[j]

            tempcoeff2 = tempcoeff1*Nk

            L = J
            KL = IJ
            KLsq = IJsq
            Nl = Nj
            lmnl = lmnj  
            ld, md, nd = lmnj
            tempcoeff3 = tempcoeff2*Nl
            npriml = nprimj
            
            

            val = 0.0
            
            #Loop over primitives
            for ik in range(bfs_nprim[i]):   
                dik = bfs_coeffs[i][ik]
                Nik = bfs_prim_norms[i][ik]
                alphaik = bfs_expnts[i][ik]
                tempcoeff4 = tempcoeff3*dik*Nik
                
                for jk in range(bfs_nprim[j]):
                    alphajk = bfs_expnts[j][jk]
                    gammaP = alphaik + alphajk
                    screenfactorAB = np.exp(-alphaik*alphajk/gammaP*IJsq)
                    # if abs(screenfactorAB)<1.0e-8:   
                    #     #TODO: Check for optimal value for screening
                    #     continue
                    djk = bfs_coeffs[j][jk] 
                    Njk = bfs_prim_norms[j][jk]      
                    P = (alphaik*I + alphajk*J)/gammaP
                    PI = P - I
                    PJ = P - J  
                    fac1 = twopisq/gammaP*screenfactorAB   
                    onefourthgammaPinv = 0.25/gammaP  
                    tempcoeff5 = tempcoeff4*djk*Njk  

                    dkk = dik
                    Nkk = Nik
                    alphakk = alphaik
                    tempcoeff6 = tempcoeff5*dkk*Nkk 

                    alphalk = alphajk
                    gammaQ = alphakk + alphalk
                    screenfactorKL = np.exp(-alphakk*alphalk/gammaQ*KLsq)
                    # if abs(screenfactorKL)<1.0e-8:   
                    #     #TODO: Check for optimal value for screening
                    #     continue
                    # if abs(screenfactorAB*screenfactorKL)<1.0e-10:   
                    #     #TODO: Check for optimal value for screening
                    #     continue
                    dlk = djk
                    Nlk = Njk     
                    Q = (alphakk*K + alphalk*L)/gammaQ        
                    PQ = P - Q
                    
                    QK = Q - K
                    QL = Q - L
                    tempcoeff7 = tempcoeff6*dlk*Nlk
                    
                  
                              
                    fac2 = fac1/gammaQ
                    

                    omega = (fac2)*np.sqrt(pi/(gammaP + gammaQ))*screenfactorKL
                    delta = 0.25*(1/gammaQ) + onefourthgammaPinv          
                    PQsqBy4delta = np.sum(PQ**2)/(4*delta)         
                        
                    sum1 = innerLoop4c2eNumba2(la,lb,lc,ld,ma,mb,mc,md,na,nb,nc,nd,gammaP,gammaQ,PI,PJ,QK,QL,PQ,PQsqBy4delta,delta)
                    # sum1 = FboysNumba2(la+lb+lc+ld+ma+mb+mc+md+na+nb+nc+nd,PQsqBy4delta)
                    # sum1 = 1.0
                        
                    val += omega*sum1*tempcoeff7
                            
            # BIG NOTE: This actually fixed an issue that I was getting where the energies of Crysx and PySCF only matched upto 10^-4 au and after
            # this change the energies match upto 10^-5 to 10^-6 au
            # But somehow this seems slower than before
            fourC2E_diag[i,j] = val
            fourC2E_diag[j,i] = val
        
    return fourC2E_diag

  
def fourCenterTwoElecSymmNumba1(bfs_coords, bfs_contr_prim_norms, bfs_lmn, bfs_nprim, bfs_coeffs, bfs_prim_norms, bfs_expnts,indx_startA,indx_endA,indx_startB,indx_endB,indx_startC,indx_endC,indx_startD,indx_endD):
        # This function calculates the electron-electron Coulomb potential matrix for a given basis object and mol object.
        # This uses 8 fold symmetry to only calculate the unique elements and assign the rest via symmetry
        # The basis object holds the information of basis functions like: exponents, coeffs, etc.
        
        # The integrals are performed using the formulas https://pubs.acs.org/doi/full/10.1021/acs.jchemed.8b00255
        # Some useful resources:
        # https://chemistry.stackexchange.com/questions/71527/how-does-one-compute-the-number-of-unique-2-electron-integrals-for-a-given-basis
        # https://chemistry.stackexchange.com/questions/82532/practical-differences-between-storing-2-electron-integrals-and-calculating-them
        # Density matrix based screening Section (5.1.2): https://www.diva-portal.org/smash/get/diva2:740301/FULLTEXT02.pdf
        # Lots of notes on screening and efficiencies (3.3 pg 31): https://www.zora.uzh.ch/id/eprint/44716/1/Diss.123456.pdf
        # In the above resource in section 3.4 it also says that storing in 64 bit floating point precision is usually not required. 
        # So, the storage requirements can be reduced by 4 to 8 times.

       

    # returns (AB|CD) 
    
    m = indx_endA - indx_startA
    n = indx_endB - indx_startB
    o = indx_endC - indx_startC
    p = indx_endD - indx_startD
    fourC2E = np.zeros((m,n,o,p),dtype=np.float64) #The difference in syntax is due to Numba
    # print('Four Center Two electron ERI size in GB ',fourC2E.nbytes/1e9)
        
       
        
    pi = 3.141592653589793
    pisq = 9.869604401089358  #PI^2
    twopisq = 19.739208802178716  #2*PI^2

    #Loop pver BFs
    for i in prange(0, indx_endA): #A
        I = bfs_coords[i]
        Ni = bfs_contr_prim_norms[i]
        lmni = bfs_lmn[i]
        la, ma, na = lmni
        nprimi = bfs_nprim[i]
        
        for j in prange(0, i+1): #B
            J = bfs_coords[j]
            IJ = I - J
            IJsq = np.sum(IJ**2)
            Nj = bfs_contr_prim_norms[j]
            lmnj = bfs_lmn[j]
            lb, mb, nb = lmnj
            tempcoeff1 = Ni*Nj
            nprimj = bfs_nprim[j]
            
            for k in prange(0, indx_endC): #C
                K = bfs_coords[k]
                Nk = bfs_contr_prim_norms[k]
                lmnk = bfs_lmn[k]
                lc, mc, nc = lmnk
                tempcoeff2 = tempcoeff1*Nk
                nprimk = bfs_nprim[k]
                
                for l in prange(0, k+1): #D
                    if i<j:
                        triangle2ij = (j)*(j+1)/2+i
                    else:
                        triangle2ij = (i)*(i+1)/2+j
                    if k<l:
                        triangle2kl = (l)*(l+1)/2+k
                    else:
                        triangle2kl = (k)*(k+1)/2+l
                    if triangle2ij>triangle2kl:
                        continue
                    L = bfs_coords[l]
                    KL = K - L  
                    KLsq = np.sum(KL**2)
                    Nl = bfs_contr_prim_norms[l]
                    lmnl = bfs_lmn[l]  
                    ld, md, nd = lmnl
                    tempcoeff3 = tempcoeff2*Nl
                    npriml = bfs_nprim[l]

                    val = 0.0
                    
                    #Loop over primitives
                    for ik in range(bfs_nprim[i]):   
                        dik = bfs_coeffs[i][ik]
                        Nik = bfs_prim_norms[i][ik]
                        alphaik = bfs_expnts[i][ik]
                        tempcoeff4 = tempcoeff3*dik*Nik
                        
                        for jk in range(bfs_nprim[j]):
                            alphajk = bfs_expnts[j][jk]
                            gammaP = alphaik + alphajk
                            screenfactorAB = np.exp(-alphaik*alphajk/gammaP*IJsq)
                            if abs(screenfactorAB)<1.0e-8:   
                                #TODO: Check for optimal value for screening
                                continue
                            djk = bfs_coeffs[j][jk] 
                            Njk = bfs_prim_norms[j][jk]      
                            P = (alphaik*I + alphajk*J)/gammaP
                            PI = P - I
                            PJ = P - J  
                            fac1 = twopisq/gammaP*screenfactorAB   
                            onefourthgammaPinv = 0.25/gammaP  
                            tempcoeff5 = tempcoeff4*djk*Njk  
                            
                            for kk in range(bfs_nprim[k]):
                                dkk = bfs_coeffs[k][kk]
                                Nkk = bfs_prim_norms[k][kk]
                                alphakk = bfs_expnts[k][kk]
                                tempcoeff6 = tempcoeff5*dkk*Nkk 
                                  
                                for lk in range(bfs_nprim[l]): 
                                    alphalk = bfs_expnts[l][lk]
                                    gammaQ = alphakk + alphalk
                                    screenfactorKL = np.exp(-alphakk*alphalk/gammaQ*KLsq)
                                    if abs(screenfactorKL)<1.0e-8:   
                                        #TODO: Check for optimal value for screening
                                        continue
                                    if abs(screenfactorAB*screenfactorKL)<1.0e-10:   
                                        #TODO: Check for optimal value for screening
                                        continue
                                    dlk = bfs_coeffs[l][lk] 
                                    Nlk = bfs_prim_norms[l][lk]     
                                    Q = (alphakk*K + alphalk*L)/gammaQ        
                                    PQ = P - Q
                                    
                                    QK = Q - K
                                    QL = Q - L
                                    tempcoeff7 = tempcoeff6*dlk*Nlk
                                      
                                    fac2 = fac1/gammaQ
                                    

                                    omega = (fac2)*np.sqrt(pi/(gammaP + gammaQ))*screenfactorKL
                                    delta = 0.25*(1/gammaQ) + onefourthgammaPinv          
                                    PQsqBy4delta = np.sum(PQ**2)/(4*delta)         
                                        
                                    sum1 = innerLoop4c2eNumba2(la,lb,lc,ld,ma,mb,mc,md,na,nb,nc,nd,gammaP,gammaQ,PI,PJ,QK,QL,PQ,PQsqBy4delta,delta)
                                    # sum1 = FboysNumba2(la+lb+lc+ld+ma+mb+mc+md+na+nb+nc+nd,PQsqBy4delta)
                                    # sum1 = 1.0
                                        
                                    val += omega*sum1*tempcoeff7
                                    
                    # BIG NOTE: This actually fixed an issue that I was getting where the energies of Crysx and PySCF only matched upto 10^-4 au and after
                    # this change the energies match upto 10^-5 to 10^-6 au
                    # But somehow this seems slower than before
                    fourC2E[i,j,k,l] = val
                    fourC2E[j,i,k,l] = fourC2E[i,j,k,l]
                    fourC2E[i,j,l,k] = fourC2E[i,j,k,l]
                    fourC2E[j,i,l,k] = fourC2E[i,j,k,l]
                    fourC2E[k,l,i,j] = fourC2E[i,j,k,l]
                    fourC2E[k,l,j,i] = fourC2E[i,j,k,l]
                    fourC2E[l,k,i,j] = fourC2E[i,j,k,l]
                    fourC2E[l,k,j,i] = fourC2E[i,j,k,l]
                                    
                                   
                            
        
    return fourC2E

def twoCenterTwoElecSymmNumba1(bfs_coords, bfs_contr_prim_norms, bfs_lmn, bfs_nprim, bfs_coeffs, bfs_prim_norms, bfs_expnts,a,b,c,d):
        # This function calculates the two centered two electron integral matrix for a given basis object and mol object.
        # This uses 8 fold symmetry to only calculate the unique elements and assign the rest via symmetry
        # The basis object holds the information of basis functions like: exponents, coeffs, etc.
        
        # The integrals are performed using the formulas https://pubs.acs.org/doi/full/10.1021/acs.jchemed.8b00255
        # Some useful resources:
        # https://chemistry.stackexchange.com/questions/71527/how-does-one-compute-the-number-of-unique-2-electron-integrals-for-a-given-basis
        # https://chemistry.stackexchange.com/questions/82532/practical-differences-between-storing-2-electron-integrals-and-calculating-them
        # Density matrix based screening Section (5.1.2): https://www.diva-portal.org/smash/get/diva2:740301/FULLTEXT02.pdf
        # Lots of notes on screening and efficiencies (3.3 pg 31): https://www.zora.uzh.ch/id/eprint/44716/1/Diss.123456.pdf
        # In the above resource in section 3.4 it also says that storing in 64 bit floating point precision is usually not required. 
        # So, the storage requirements can be reduced by 4 to 8 times.

       

    # returns (AB|CD) 
    
    m = b-a
    n = d-c
    twoC2E = np.zeros((m,n)) #The difference in syntax is due to Numba
    # print('Four Center Two electron ERI size in GB ',fourC2E.nbytes/1e9)
    # print(comb(0,0))
    # print(c2kNumba(1,2,0,1.0,0.0))
       
        
    pi = 3.141592653589793
    pisq = 9.869604401089358  #PI^2
    twopisq = 19.739208802178716  #2*PI^2

    J = L = np.zeros((3))
    Nj = Nl = 1
    lnmj = lmnl = np.zeros((3),dtype=np.int32)
    lb, mb, nb = int(0), int(0), int(0)
    ld, md, nd = int(0), int(0), int(0)
    alphajk = alphalk = 0.0
    djk, dlk = 1.0, 1.0
    Njk, Nlk = 1.0, 1.0
    #Loop pver BFs
    for i in prange(a, b): #A
        I = bfs_coords[i]
        IJ = I #- J
        # IJsq = np.sum(IJ**2)
        Ni = bfs_contr_prim_norms[i]
        lmni = bfs_lmn[i]
        la, ma, na = lmni
        
           
            
        for k in prange(c, i+1): #C
            K = bfs_coords[k]
            Nk = bfs_contr_prim_norms[k]
            lmnk = bfs_lmn[k]
            lc, mc, nc = lmnk
            KL = K #- L  
            # KLsq = np.sum(KL**2)
            
            tempcoeff1 = Ni*Nk
                    

            
                    
            #Loop over primitives
            for ik in range(bfs_nprim[i]):   
                dik = bfs_coeffs[i][ik]
                Nik = bfs_prim_norms[i][ik]
                alphaik = bfs_expnts[i][ik]
                tempcoeff4 = tempcoeff1*dik*Nik
                gammaP = alphaik #+ alphajk
                screenfactorAB = 1.0#np.exp(-alphaik*alphajk/gammaP*IJsq)
                            
                P = I
                PI = np.zeros((3))
                PJ = IJ#P - J  
                fac1 = twopisq/gammaP*screenfactorAB   
                onefourthgammaPinv = 0.25/gammaP  
                
                            
                for kk in range(bfs_nprim[k]):
                    dkk = bfs_coeffs[k][kk]
                    Nkk = bfs_prim_norms[k][kk]
                    alphakk = bfs_expnts[k][kk]
                    tempcoeff5 = tempcoeff4*dkk*Nkk 
                                  
                    gammaQ = alphakk #+ alphalk
                    screenfactorKL = 1.0#np.exp(-alphakk*alphalk/gammaQ*KLsq)
                    Q = K#(alphakk*K + alphalk*L)/gammaQ        
                    PQ = P - Q
                    QK = Q - K #np.ones((3))
                    QL = KL#Q - L
                    
                                      
                    fac2 = fac1/gammaQ
                                    

                    omega = (fac2)*np.sqrt(pi/(gammaP + gammaQ))*screenfactorKL
                    delta = 0.25*(1/gammaQ) + onefourthgammaPinv          
                    PQsqBy4delta = np.sum(PQ**2)/(4*delta)         
                    sum1 = innerLoop4c2eNumba2(la,lb,lc,ld,ma,mb,mc,md,na,nb,nc,nd,gammaP,gammaQ,PI,PJ,QK,QL,PQ,PQsqBy4delta,delta)
                    # sum1 = FboysNumba2(la+lb+lc+ld+ma+mb+mc+md+na+nb+nc+nd,PQsqBy4delta)
                    # sum1 = 1.0
                                        
                    twoC2E[i,k] += omega*sum1*tempcoeff5
                                    
                    
                             
            twoC2E[k,i] = twoC2E[i,k]
                                    
                                   
                            
        
    return twoC2E

def threeCenterTwoElecSymmNumba1(bfs_coords, bfs_contr_prim_norms, bfs_lmn, bfs_nprim, bfs_coeffs, bfs_prim_norms, bfs_expnts,aux_bfs_coords, aux_bfs_contr_prim_norms, aux_bfs_lmn, aux_bfs_nprim, aux_bfs_coeffs, aux_bfs_prim_norms, aux_bfs_expnts,indx_startA,indx_endA,indx_startB,indx_endB,indx_startC,indx_endC):
        # This function calculates the electron-electron Coulomb potential matrix for a given basis object and mol object.
        # This uses 8 fold symmetry to only calculate the unique elements and assign the rest via symmetry
        # The basis object holds the information of basis functions like: exponents, coeffs, etc.
        
        # The integrals are performed using the formulas https://pubs.acs.org/doi/full/10.1021/acs.jchemed.8b00255
        # Some useful resources:
        # https://chemistry.stackexchange.com/questions/71527/how-does-one-compute-the-number-of-unique-2-electron-integrals-for-a-given-basis
        # https://chemistry.stackexchange.com/questions/82532/practical-differences-between-storing-2-electron-integrals-and-calculating-them
        # Density matrix based screening Section (5.1.2): https://www.diva-portal.org/smash/get/diva2:740301/FULLTEXT02.pdf
        # Lots of notes on screening and efficiencies (3.3 pg 31): https://www.zora.uzh.ch/id/eprint/44716/1/Diss.123456.pdf
        # In the above resource in section 3.4 it also says that storing in 64 bit floating point precision is usually not required. 
        # So, the storage requirements can be reduced by 4 to 8 times.

       

    # returns (AB|CD) 
    
    m = indx_endA - indx_startA
    n = indx_endB - indx_startB
    o = indx_endC - indx_startC
    
    threeC2E = np.zeros((m,n,o),dtype=np.float64) #The difference in syntax is due to Numba
    # print('Four Center Two electron ERI size in GB ',fourC2E.nbytes/1e9)
        
       
        
    pi = 3.141592653589793
    pisq = 9.869604401089358  #PI^2
    twopisq = 19.739208802178716  #2*PI^2

    L = np.zeros((3))
    ld, md, nd = int(0), int(0), int(0)
    alphalk = 0.0

    #Loop pver BFs
    for i in prange(0, indx_endA): #A
        I = bfs_coords[i]
        Ni = bfs_contr_prim_norms[i]
        lmni = bfs_lmn[i]
        la, ma, na = lmni
        # nprimi = bfs_nprim[i]
        
        for j in prange(0, i+1): #B
            J = bfs_coords[j]
            IJ = I - J
            IJsq = np.sum(IJ**2)
            Nj = bfs_contr_prim_norms[j]
            lmnj = bfs_lmn[j]
            lb, mb, nb = lmnj
            tempcoeff1 = Ni*Nj
            # nprimj = bfs_nprim[j]
            
            for k in prange(0, indx_endC): #C
                K = aux_bfs_coords[k]
                Nk = aux_bfs_contr_prim_norms[k]
                lmnk = aux_bfs_lmn[k]
                lc, mc, nc = lmnk
                tempcoeff2 = tempcoeff1*Nk
                # nprimk = aux_bfs_nprim[k]
                
                
                    
                KL = K #- L  
                KLsq = np.sum(KL**2)
                
                val = 0.0
                    
                #Loop over primitives
                for ik in range(bfs_nprim[i]):   
                    dik = bfs_coeffs[i][ik]
                    Nik = bfs_prim_norms[i][ik]
                    alphaik = bfs_expnts[i][ik]
                    tempcoeff3 = tempcoeff2*dik*Nik
                        
                    for jk in range(bfs_nprim[j]):
                        alphajk = bfs_expnts[j][jk]
                        gammaP = alphaik + alphajk
                        screenfactorAB = np.exp(-alphaik*alphajk/gammaP*IJsq)
                        if abs(screenfactorAB)<1.0e-8:   
                            #TODO: Check for optimal value for screening
                            continue
                        djk = bfs_coeffs[j][jk] 
                        Njk = bfs_prim_norms[j][jk]      
                        P = (alphaik*I + alphajk*J)/gammaP
                        PI = P - I
                        PJ = P - J  
                        fac1 = twopisq/gammaP*screenfactorAB   
                        onefourthgammaPinv = 0.25/gammaP  
                        tempcoeff4 = tempcoeff3*djk*Njk  
                            
                        for kk in range(aux_bfs_nprim[k]):
                            dkk = aux_bfs_coeffs[k][kk]
                            Nkk = aux_bfs_prim_norms[k][kk]
                            alphakk = aux_bfs_expnts[k][kk]
                            tempcoeff5 = tempcoeff4*dkk*Nkk 
                                  
                             
                            gammaQ = alphakk #+ alphalk
                            screenfactorKL = np.exp(-alphakk*alphalk/gammaQ*KLsq)
                            if abs(screenfactorKL)<1.0e-8:   
                                #TODO: Check for optimal value for screening
                                continue
                            if abs(screenfactorAB*screenfactorKL)<1.0e-10:   
                                #TODO: Check for optimal value for screening
                                continue
                            
                            Q = K#(alphakk*K + alphalk*L)/gammaQ        
                            PQ = P - Q
                                    
                            QK = Q - K
                            QL = Q #- L
                            
                                      
                            fac2 = fac1/gammaQ
                                    

                            omega = (fac2)*np.sqrt(pi/(gammaP + gammaQ))#*screenfactorKL
                            delta = 0.25*(1/gammaQ) + onefourthgammaPinv          
                            PQsqBy4delta = np.sum(PQ**2)/(4*delta)         
                                        
                            sum1 = innerLoop4c2eNumba2(la,lb,lc,ld,ma,mb,mc,md,na,nb,nc,nd,gammaP,gammaQ,PI,PJ,QK,QL,PQ,PQsqBy4delta,delta)
                            # sum1 = FboysNumba2(la+lb+lc+ld+ma+mb+mc+md+na+nb+nc+nd,PQsqBy4delta)
                            # sum1 = 1.0
                                        
                            val += omega*sum1*tempcoeff5
                                    
                    # BIG NOTE: This actually fixed an issue that I was getting where the energies of Crysx and PySCF only matched upto 10^-4 au and after
                    # this change the energies match upto 10^-5 to 10^-6 au
                    # But somehow this seems slower than before
                    threeC2E[i,j,k] = val
                    threeC2E[j,i,k] = val
                    
                                    
                                   
                            
        
    return threeC2E

def fourCenterTwoElecFastNumba1(quadruplets, bfs_coords, bfs_contr_prim_norms, bfs_lmn, bfs_nprim, bfs_coeffs, bfs_prim_norms, bfs_expnts,indx_startA,indx_endA,indx_startB,indx_endB,indx_startC,indx_endC,indx_startD,indx_endD):
        # Here quadruplets is the list of indices (i,j,k,l) that have been generated using np.stack to circumvent having four different loops over i,j,k,l.
        # This function calculates the electron-electron Coulomb potential matrix for a given basis object and mol object.
        # The basis object holds the information of basis functions like: exponents, coeffs, etc.
        
        # It is possible to only calculate a slice (block) of the complete matrix.
        # slice is a 4 element list whose first and second elements give the range of the rows to be calculated.
        # the third and fourth element give the range of columns to be calculated.
        # The integrals are performed using the formulas https://pubs.acs.org/doi/full/10.1021/acs.jchemed.8b00255
        # Some useful resources:
        # https://chemistry.stackexchange.com/questions/71527/how-does-one-compute-the-number-of-unique-2-electron-integrals-for-a-given-basis
        # https://chemistry.stackexchange.com/questions/82532/practical-differences-between-storing-2-electron-integrals-and-calculating-them
        # Density matrix based screening Section (5.1.2): https://www.diva-portal.org/smash/get/diva2:740301/FULLTEXT02.pdf
        # Lots of notes on screening and efficiencies (3.3 pg 31): https://www.zora.uzh.ch/id/eprint/44716/1/Diss.123456.pdf
        # In the above resource in section 3.4 it also says that storing in 64 bit floating point precision is usually not required. 
        # So, the storage requirements can be reduced by 4 to 8 times.
        # https://d-nb.info/1140164724/34

       

    # returns (AB|CD) 
    
    m = indx_endA - indx_startA
    n = indx_endB - indx_startB
    o = indx_endC - indx_startC
    p = indx_endD - indx_startD
    fourC2E = np.zeros((m,n,o,p)) #The difference in syntax is due to Numba

    nquads = quadruplets.shape[0]
        
    pi = 3.141592653589793
    pisq = 9.869604401089358  #PI^2
    twopisq = 19.739208802178716  #2*PI^2

    #Loop pver BFs
    for indx_quad in prange(0,nquads):
        # i = quadruplets[indx_quad, 0]
        # j = quadruplets[indx_quad, 1]
        # k = quadruplets[indx_quad, 2]
        # l = quadruplets[indx_quad, 3]

        i,j,k,l = quadruplets[indx_quad]

        I = bfs_coords[i]
        Ni = bfs_contr_prim_norms[i]
        lmni = bfs_lmn[i]
        la, ma, na = lmni
        nprimi = bfs_nprim[i]
        
        
        J = bfs_coords[j]
        IJ = I - J
        IJsq = np.sum(IJ**2)
        Nj = bfs_contr_prim_norms[j]
        lmnj = bfs_lmn[j]
        lb, mb, nb = lmnj
        tempcoeff1 = Ni*Nj
        nprimj = bfs_nprim[j]
            
        K = bfs_coords[k]
        Nk = bfs_contr_prim_norms[k]
        lmnk = bfs_lmn[k]
        lc, mc, nc = lmnk
        tempcoeff2 = tempcoeff1*Nk
        nprimk = bfs_nprim[k]
                
        L = bfs_coords[l]
        KL = K - L  
        KLsq = np.sum(KL**2)
        Nl = bfs_contr_prim_norms[l]
        lmnl = bfs_lmn[l]  
        ld, md, nd = lmnl
        tempcoeff3 = tempcoeff2*Nl
        npriml = bfs_nprim[l]
                    
        #Loop over primitives
        for ik in range(bfs_nprim[i]):   
                        dik = bfs_coeffs[i,ik]
                        Nik = bfs_prim_norms[i,ik]
                        alphaik = bfs_expnts[i,ik]
                        tempcoeff4 = tempcoeff3*dik*Nik
                        
                        for jk in range(bfs_nprim[j]):
                            alphajk = bfs_expnts[j,jk]
                            gammaP = alphaik + alphajk
                            screenfactorAB = np.exp(-alphaik*alphajk/gammaP*IJsq)
                            if abs(screenfactorAB)<1.0e-8:   
                                #TODO: Check for optimal value for screening
                                continue
                            djk = bfs_coeffs[j][jk] 
                            Njk = bfs_prim_norms[j,jk]      
                            P = (alphaik*I + alphajk*J)/gammaP
                            PI = P - I
                            PJ = P - J  
                            fac1 = twopisq/gammaP*screenfactorAB   
                            onefourthgammaPinv = 0.25/gammaP  
                            tempcoeff5 = tempcoeff4*djk*Njk  
                            
                            for kk in prange(bfs_nprim[k]):
                                dkk = bfs_coeffs[k,kk]
                                Nkk = bfs_prim_norms[k,kk]
                                alphakk = bfs_expnts[k,kk]
                                tempcoeff6 = tempcoeff5*dkk*Nkk 
                                  
                                for lk in prange(bfs_nprim[l]): 
                                    alphalk = bfs_expnts[l,lk]
                                    gammaQ = alphakk + alphalk
                                    screenfactorKL = np.exp(-alphakk*alphalk/gammaQ*KLsq)
                                    if abs(screenfactorKL)<1.0e-8:   
                                        #TODO: Check for optimal value for screening
                                        continue
                                    if abs(screenfactorAB*screenfactorKL)<1.0e-10:   
                                        #TODO: Check for optimal value for screening
                                        continue
                                    dlk = bfs_coeffs[l,lk] 
                                    Nlk = bfs_prim_norms[l,lk]     
                                    Q = (alphakk*K + alphalk*L)/gammaQ        
                                    PQ = P - Q
                                    
                                    QK = Q - K
                                    QL = Q - L
                                    tempcoeff7 = tempcoeff6*dlk*Nlk
                                      
                                    fac2 = fac1/gammaQ
                                    

                                    omega = (fac2)*np.sqrt(pi/(gammaP + gammaQ))*screenfactorKL
                                    delta = 0.25*(1/gammaQ) + onefourthgammaPinv          
                                    PQsqBy4delta = np.sum(PQ**2)/(4*delta)         
                                        
                                    sum1 = innerLoop4c2eNumba2(la,lb,lc,ld,ma,mb,mc,md,na,nb,nc,nd,gammaP,gammaQ,PI,PJ,QK,QL,PQ,PQsqBy4delta,delta)
                                    # sum1 = FboysNumba2(la+lb+lc+ld+ma+mb+mc+md+na+nb+nc+nd,PQsqBy4delta)
                                    # sum1 = 1.0
                                        
                                    fourC2E[i,j,k,l] = fourC2E[i,j,k,l] + omega*sum1*tempcoeff7
                                    
                                   
                            
        
    return fourC2E

def genQuadrupletsNumba1(n,quadruplets):
    indx = 0 # Index to loop over quadruplets
    for i in range(0,n):
        for j in range(0,i+1):
            for k in range(0,n):
                for l in range(0,k+1):
                    if i<j:
                        triangle2ij = (j)*(j+1)/2+i
                    else:
                        triangle2ij = (i)*(i+1)/2+j
                    if k<l:
                        triangle2kl = (l)*(l+1)/2+k
                    else:
                        triangle2kl = (k)*(k+1)/2+l
                    if triangle2ij<=triangle2kl:
                        # quadruplets[i] = int(i), int(j), int(k), int(l)
                        quadruplets[indx] = i,j,k,l
                        indx = indx + 1
                        # print(i,j,k,l)
                        # count = count+1
    # print(quadruplets)
    return quadruplets

def fourCenterTwoElecFastSymmNumba1(quadruplets, bfs_coords, bfs_contr_prim_norms, bfs_lmn, bfs_nprim, bfs_coeffs, bfs_prim_norms, bfs_expnts,indx_startA,indx_endA,indx_startB,indx_endB,indx_startC,indx_endC,indx_startD,indx_endD):
        # This implementation only takes in unique quadruplets based on symmetry (8-fold) and then generates the remaining elements of the N^4
        # array using the symmetry operations.
        # Here quadruplets is the list of indices (i,j,k,l) that have been generated using np.stack to circumvent having four different loops over i,j,k,l.
        # This function calculates the electron-electron Coulomb potential matrix for a given basis object and mol object.
        # The basis object holds the information of basis functions like: exponents, coeffs, etc.
        
        # It is possible to only calculate a slice (block) of the complete matrix.
        # slice is a 4 element list whose first and second elements give the range of the rows to be calculated.
        # the third and fourth element give the range of columns to be calculated.
        # The integrals are performed using the formulas https://pubs.acs.org/doi/full/10.1021/acs.jchemed.8b00255
        # Some useful resources:
        # https://chemistry.stackexchange.com/questions/71527/how-does-one-compute-the-number-of-unique-2-electron-integrals-for-a-given-basis
        # https://chemistry.stackexchange.com/questions/82532/practical-differences-between-storing-2-electron-integrals-and-calculating-them
        # Density matrix based screening Section (5.1.2): https://www.diva-portal.org/smash/get/diva2:740301/FULLTEXT02.pdf
        # Lots of notes on screening and efficiencies (3.3 pg 31): https://www.zora.uzh.ch/id/eprint/44716/1/Diss.123456.pdf
        # In the above resource in section 3.4 it also says that storing in 64 bit floating point precision is usually not required. 
        # So, the storage requirements can be reduced by 4 to 8 times.

       

    # returns (AB|CD) 
    
    m = indx_endA - indx_startA
    n = indx_endB - indx_startB
    o = indx_endC - indx_startC
    p = indx_endD - indx_startD
    fourC2E = np.zeros((m,n,o,p)) #The difference in syntax is due to Numba

    nquads = quadruplets.shape[0]
        
    pi = 3.141592653589793
    pisq = 9.869604401089358  #PI^2
    twopisq = 19.739208802178716  #2*PI^2

    #Loop pver BFs
    for indx_quad in prange(0,nquads):
        # i = quadruplets[indx_quad, 0]
        # j = quadruplets[indx_quad, 1]
        # k = quadruplets[indx_quad, 2]
        # l = quadruplets[indx_quad, 3]

        i,j,k,l = quadruplets[indx_quad]

        I = bfs_coords[i]
        Ni = bfs_contr_prim_norms[i]
        lmni = bfs_lmn[i]
        la, ma, na = lmni
        nprimi = bfs_nprim[i]
        
        
        J = bfs_coords[j]
        IJ = I - J
        IJsq = np.sum(IJ**2)
        Nj = bfs_contr_prim_norms[j]
        lmnj = bfs_lmn[j]
        lb, mb, nb = lmnj
        tempcoeff1 = Ni*Nj
        nprimj = bfs_nprim[j]
            
        K = bfs_coords[k]
        Nk = bfs_contr_prim_norms[k]
        lmnk = bfs_lmn[k]
        lc, mc, nc = lmnk
        tempcoeff2 = tempcoeff1*Nk
        nprimk = bfs_nprim[k]
                
        L = bfs_coords[l]
        KL = K - L  
        KLsq = np.sum(KL**2)
        Nl = bfs_contr_prim_norms[l]
        lmnl = bfs_lmn[l]  
        ld, md, nd = lmnl
        tempcoeff3 = tempcoeff2*Nl
        npriml = bfs_nprim[l]
                    
        #Loop over primitives
        for ik in range(bfs_nprim[i]):   
                        dik = bfs_coeffs[i,ik]
                        Nik = bfs_prim_norms[i,ik]
                        alphaik = bfs_expnts[i,ik]
                        tempcoeff4 = tempcoeff3*dik*Nik
                        
                        for jk in range(bfs_nprim[j]):
                            alphajk = bfs_expnts[j,jk]
                            gammaP = alphaik + alphajk
                            screenfactorAB = np.exp(-alphaik*alphajk/gammaP*IJsq)
                            if abs(screenfactorAB)<1.0e-8:   
                                #TODO: Check for optimal value for screening
                                continue
                            djk = bfs_coeffs[j][jk] 
                            Njk = bfs_prim_norms[j,jk]      
                            P = (alphaik*I + alphajk*J)/gammaP
                            PI = P - I
                            PJ = P - J  
                            fac1 = twopisq/gammaP*screenfactorAB   
                            onefourthgammaPinv = 0.25/gammaP  
                            tempcoeff5 = tempcoeff4*djk*Njk  
                            
                            for kk in range(bfs_nprim[k]):
                                dkk = bfs_coeffs[k,kk]
                                Nkk = bfs_prim_norms[k,kk]
                                alphakk = bfs_expnts[k,kk]
                                tempcoeff6 = tempcoeff5*dkk*Nkk 
                                  
                                for lk in range(bfs_nprim[l]): 
                                    alphalk = bfs_expnts[l,lk]
                                    gammaQ = alphakk + alphalk
                                    screenfactorKL = np.exp(-alphakk*alphalk/gammaQ*KLsq)
                                    if abs(screenfactorKL)<1.0e-8:   
                                        #TODO: Check for optimal value for screening
                                        continue
                                    if abs(screenfactorAB*screenfactorKL)<1.0e-10:   
                                        #TODO: Check for optimal value for screening
                                        continue
                                    dlk = bfs_coeffs[l,lk] 
                                    Nlk = bfs_prim_norms[l,lk]     
                                    Q = (alphakk*K + alphalk*L)/gammaQ        
                                    PQ = P - Q
                                    
                                    QK = Q - K
                                    QL = Q - L
                                    tempcoeff7 = tempcoeff6*dlk*Nlk
                                      
                                    fac2 = fac1/gammaQ
                                    

                                    omega = (fac2)*np.sqrt(pi/(gammaP + gammaQ))*screenfactorKL
                                    delta = 0.25*(1/gammaQ) + onefourthgammaPinv          
                                    PQsqBy4delta = np.sum(PQ**2)/(4*delta)         
                                        
                                    sum1 = innerLoop4c2eNumba2(la,lb,lc,ld,ma,mb,mc,md,na,nb,nc,nd,gammaP,gammaQ,PI,PJ,QK,QL,PQ,PQsqBy4delta,delta)
                                    # sum1 = FboysNumba2(la+lb+lc+ld+ma+mb+mc+md+na+nb+nc+nd,PQsqBy4delta)
                                    # sum1 = 1.0
                                        
                                    fourC2E[i,j,k,l] = fourC2E[i,j,k,l] + omega*sum1*tempcoeff7
                                    fourC2E[j,i,k,l] = fourC2E[i,j,k,l]
                                    fourC2E[i,j,l,k] = fourC2E[i,j,k,l]
                                    fourC2E[j,i,l,k] = fourC2E[i,j,k,l]
                                    fourC2E[k,l,i,j] = fourC2E[i,j,k,l]
                                    fourC2E[k,l,j,i] = fourC2E[i,j,k,l]
                                    fourC2E[l,k,i,j] = fourC2E[i,j,k,l]
                                    fourC2E[l,k,j,i] = fourC2E[i,j,k,l]
                                    # if not (i==j==k==l):
                                    #     if not(i==j):
                                    #         fourC2E[j,i,k,l] = fourC2E[i,j,k,l]
                                    #     if not(k==l):
                                    #         fourC2E[i,j,l,k] = fourC2E[i,j,k,l]
                                    #     if i!=j and k!=l:
                                    #         fourC2E[j,i,l,k] = fourC2E[i,j,k,l]
                                    #     if i!=k or j!=l:
                                    #         fourC2E[k,l,i,j] = fourC2E[i,j,k,l]
                                    #     if not(i==j):
                                    #         fourC2E[k,l,j,i] = fourC2E[i,j,k,l]
                                    #     if not(k==l):
                                    #         fourC2E[l,k,i,j] = fourC2E[i,j,k,l]
                                    #     if i!=j and k!=l:
                                    #         fourC2E[l,k,j,i] = fourC2E[i,j,k,l]
                                            
    return fourC2E

def FboysRecursiveNumba1(v, x, Fold):
    return (np.exp(-x) + 2 * x * Fold) / (2 * v - 1)


# define wrapper call function with optimizer
if NUMBA_EXISTS:
    print('Numba found! Using Numba')
    calcSNumba = njit(cache=True)(calcSNumba1)
    c2kNumba = njit(cache=True)(c2kNumba1)
    doublefactorial = njit(cache=True)(doublefactorial1)
    comb = njit(cache=True, fastmath=True)(comb1)
    fac = njit(cache=True)(fac1)
    overlapMatNumba2 = njit(parallel=True, cache=True)(overlapMatNumba1)
    overlapMatSymmNumba2 = njit(parallel=True, cache=True)(overlapMatSymmNumba1)
    kinMatNumba2 = njit(parallel=True, cache=True)(kinMatNumba1)
    kinMatSymmNumba2 = njit(parallel=True, cache=True)(kinMatSymmNumba1)
    evalGTONumba = njit(cache=True, nogil=True, error_model="numpy", fastmath=True)(evalGTONumba1)
    evalGTOandGradNumba = njit(cache=True, nogil=True, error_model="numpy", fastmath=True)(evalGTOandGradNumba1)
    evalGTOgradxNumba2 = njit(cache=True, nogil=True, error_model="numpy", fastmath=True)(evalGTOgradxNumba1)
    evalGTOgradyNumba2 = njit(cache=True, nogil=True, error_model="numpy", fastmath=True)(evalGTOgradyNumba1)
    evalGTOgradzNumba2 = njit(cache=True, nogil=True, error_model="numpy", fastmath=True)(evalGTOgradzNumba1)
    evalBFsNumba2 = njit(parallel=True, cache=True, nogil=True)(evalBFsNumba1)
    evalBFsSparseNumba2 = njit(parallel=True, cache=True, nogil=True)(evalBFsSparseNumba1)
    evalBFs_serialNumba2 = njit(parallel=False, cache=True, nogil=True)(evalBFsNumba1)
    evalBFsSparse_serialNumba2 = njit(parallel=False, cache=True, nogil=True)(evalBFsSparseNumba1)
    nonZeroBFIndicesNumba2 = njit(parallel=True, cache=True, nogil=True)(nonZeroBFIndicesNumba1)
    evalBFsandRhoNumba2 = njit(parallel=True, cache=True, nogil=True)(evalBFsandRhoNumba1)
    evalBFsandRho_serialNumba2 = njit(parallel=False, cache=True, nogil=True)(evalBFsandRho_serialNumba1)
    evalBFsgradNumba2 = njit(parallel=True, cache=True, nogil=True)(evalBFsgradNumba1)
    evalBFsandgradNumba2 = njit(parallel=True, cache=True, nogil=True, error_model="numpy", fastmath=True)(evalBFsandgradNumba1)
    evalRhoNumba2 = njit(parallel=True, cache=True, nogil=True, error_model="numpy", fastmath=True)(evalRhoNumba1)
    evalRho_serialNumba2 = njit(parallel=False, cache=True, nogil=True, error_model="numpy", fastmath=True)(evalRhoNumba1)
    calcZ2 = njit(parallel=True, cache=True)(calcZ1)
    overlapPrimitives = njit(parallel=True, cache=True)(overlapPrimitives1)
    vlriNumbaPartial = njit(cache=True)(vlriNumbaPartial1)
    nucMatNumba2 = njit(parallel=True, cache=True)(nucMatNumba1)
    nucMatSymmNumba2 = njit(parallel=True, cache=True)(nucMatSymmNumba1)
    FboysNumba2 = njit(cache=True)(FboysNumba1)
    FboysNumba2_jjgoings = njit()(FboysNumba1_jjgoings)
    gammaNumba = njit(cache=True)(gammaNumba1)
    incGammaNumba = njit(cache=True)(incGammaNumba1)
    calcCgammincNumba = njit(cache=True)(calcCgammincNumba1)
    overlapPrimitivesOSNumba2 = njit(cache=True)(overlapPrimitivesOSNumba1)
    overlapPrimitivesOS_2Numba2 = njit(cache=True)(overlapPrimitivesOS_2Numba1)
    overlapMatOSNumba2 = njit(parallel=True, cache=True)(overlapMatOSNumba1)
    fourCenterTwoElecNumba2 = njit(parallel=True, cache=True)(fourCenterTwoElecNumba1)
    fourCenterTwoElecSymmNumba2 = njit(parallel=True, cache=True)(fourCenterTwoElecSymmNumba1)
    fourCenterTwoElecFastNumba2 = njit(parallel=True, cache=True)(fourCenterTwoElecFastNumba1)
    fourCenterTwoElecFastSymmNumba2 = njit(parallel=True, cache=True)(fourCenterTwoElecFastSymmNumba1)
    eri_4c2e_diagonal_numba2 = njit(parallel=True, fastmath=True, cache=True, error_model="numpy")(eri_4c2e_diagonal_numba1)
    genQuadrupletsNumba2 = njit(cache=True)(genQuadrupletsNumba1)
    gNumba2 = njit(cache=True)(gNumba1)
    thetaNumba2 = njit(cache=True)(thetaNumba1)
    innerLoop4c2eNumba2 = njit(cache=True, fastmath=True)(innerLoop4c2eNumba1)
    fastFactorial2 = njit(cache=True, fastmath=True)(fastFactorial1)
    FboysRecursiveNumba2 = njit(cache=True)(FboysRecursiveNumba1)
    Root1Numba2 = njit(cache=True,fastmath=True, error_model='numpy')(Root1Numba1)
    Root2Numba2 = njit(cache=True,fastmath=True, error_model='numpy')(Root2Numba1)
    Root3Numba2 = njit(cache=True,fastmath=True, error_model='numpy')(Root3Numba1)
    Root4Numba2 = njit(cache=True,fastmath=True, error_model='numpy')(Root4Numba1)
    Root5Numba2 = njit(cache=True,fastmath=True, error_model='numpy')(Root5Numba1)
    RecurFactorsNumba2 = njit(cache=True, fastmath=True, error_model='numpy')(RecurFactorsNumba1)
    rys4c2eSymmNumba2 = njit(parallel=True, cache=True, error_model='numpy')(rys4c2eSymmNumba1)
    coulomb_rysNumba2 = njit(cache=True, error_model='numpy')(coulomb_rysNumba1)
    # ShiftNumba2 = njit("float64(float64[:,:],int32, int32, int32, int32, float64, float64)",cache=True)(ShiftNumba1)
    ShiftNumba2 = njit(cache=True, error_model='numpy')(ShiftNumba1)
    # import numba 
    # RecurNumba2 = njit("numba.types.Array(numba.float64, 0, 'C')(numba.float64,numba.float64,numba.int32,numba.int32,numba.int32,numba.int32,numba.float64,numba.float64,numba.float64,numba.float64,numba.float64,numba.float64,numba.float64,numba.float64)",cache=True)(RecurNumba1)
    RecurNumba2 = njit(cache=True, error_model='numpy')(RecurNumba1)
    Int1dNumba2 = njit(cache=True, error_model='numpy')(Int1dNumba1)
    RootsNumba2 = njit(cache=True, error_model='numpy')(RootsNumba1)
    ChebGausIntNumba2 = njit(cache=True, error_model='numpy')(ChebGausIntNumba1)
    nERIRysNumba2 = njit(cache=True, error_model='numpy')(nERIRysNumba1)
    rys2c2eSymmNumba2 = njit(parallel=True, cache=True, error_model='numpy')(rys2c2eSymmNumba1)
    rys3c2eSymmNumba2 = njit(parallel=True, cache=True, error_model='numpy')(rys3c2eSymmNumba1)
    rys3c2eSymmNumba_tri2 = njit(parallel=True, cache=True, error_model='numpy')(rys3c2eSymmNumba_tri1)
    rys3c2eSymmNumba_tri_schwarz2 = njit(parallel=True, fastmath=True, cache=True, error_model='numpy')(rys3c2eSymmNumba_tri_schwarz1)
    twoCenterTwoElecSymmNumba2 = njit(parallel=True, cache=True, error_model='numpy')(twoCenterTwoElecSymmNumba1)
    threeCenterTwoElecSymmNumba2 = njit(parallel=True, cache=True, error_model='numpy')(threeCenterTwoElecSymmNumba1)
    matMulNumba2 = njit(parallel=True, fastmath=True, cache=True, error_model='numpy')(matMulNumba1)
    calcVtempNumba2 = njit(parallel=True, fastmath=True, cache=True, error_model='numpy')(calcVtempNumba1)
else:
    print('Falling back to python')
    calcSNumba = calcSNumba1
    c2kNumba = c2kNumba1
    doublefactorial = doublefactorial1
    comb = comb1
    fac = fac1
    overlapMatNumba2 = overlapMatNumba1
    overlapMatSymmNumba2 = overlapMatSymmNumba1
    kinMatNumba2 = kinMatNumba1
    kinMatSymmNumba2 = kinMatSymmNumba1
    nucMatNumba2 = nucMatNumba1
    nucMatSymmNumba2 = nucMatSymmNumba1
    evalGTONumba = evalGTONumba1
    evalGTOandGradNumba = evalGTOandGradNumba1
    evalBFsNumba2 = evalBFsNumba1
    evalGTOgradxNumba2 = evalGTOgradxNumba1
    evalGTOgradyNumba2 = evalGTOgradyNumba1
    evalGTOgradzNumba2 = evalGTOgradzNumba1
    evalBFsgradNumba2 = evalBFsgradNumba1
    evalBFsandgradNumba2 = evalBFsandgradNumba1
    evalRhoNumba2 = evalRhoNumba1
    calcZ2 = calcZ1
    FboysNumba2 = FboysNumba1
    FboysNumba2_jjgoings = FboysNumba1_jjgoings
    overlapPrimitives = overlapPrimitives1
    vlriNumbaPartial = vlriNumbaPartial1
    gammaNumba = gammaNumba1
    incGammaNumba = incGammaNumba1
    calcCgammincNumba = calcCgammincNumba1
    overlapPrimitivesOSNumba2 = overlapPrimitivesOSNumba1
    overlapPrimitivesOS_2Numba2 = overlapPrimitivesOS_2Numba1
    overlapMatOSNumba2 = overlapMatOSNumba1
    fourCenterTwoElecNumba2 = fourCenterTwoElecNumba1
    fourCenterTwoElecSymmNumba2 = fourCenterTwoElecSymmNumba1
    fourCenterTwoElecFastNumba2 = fourCenterTwoElecFastNumba1
    fourCenterTwoElecFastSymmNumba2 = fourCenterTwoElecFastSymmNumba1
    genQuadrupletsNumba2 = genQuadrupletsNumba1 
    gNumba2 = gNumba1
    thetaNumba2 = thetaNumba1
    innerLoop4c2eNumba2 = innerLoop4c2eNumba1
    fastFactorial2 = fastFactorial1
    FboysRecursiveNumba2 = FboysRecursiveNumba1
    prange = range
    Root1Numba2 = Root1Numba1
    Root2Numba2 = Root2Numba1 
    Root3Numba2 = Root3Numba1
    Root4Numba2 = Root4Numba1
    Root5Numba2 = Root5Numba1
    RecurFactorsNumba2 = RecurFactorsNumba1
    rys4c2eSymmNumba2 = rys4c2eSymmNumba1
    coulomb_rysNumba2 = coulomb_rysNumba1
    ShiftNumba2 = ShiftNumba1
    RecurNumba2 = RecurNumba1
    Int1dNumba2 = Int1dNumba1
    RootsNumba2 = RootsNumba1
    ChebGausIntNumba2 = ChebGausIntNumba1
    nERIRysNumba2 = nERIRysNumba1
    rys2c2eSymmNumba2 = rys2c2eSymmNumba1
    rys3c2eSymmNumba2 = rys3c2eSymmNumba1
    rys3c2eSymmNumba_tri2 = rys3c2eSymmNumba_tri1
    twoCenterTwoElecSymmNumba2 = twoCenterTwoElecSymmNumba1
    threeCenterTwoElecSymmNumba2 = threeCenterTwoElecSymmNumba1
    matMulNumba2 = matMulNumba1
    calcVtempNumba2 = calcVtempNumba1

class Integrals:
    #Class to calculate integrals

    # The following links are useful to get information and knowledge about the calculation of various integrals
    # A standard grid for density functional calculations https://www.sciencedirect.com/science/article/pii/0009261493801259
    # Molecular integrals Over Gaussian Basis Functions https://www.sciencedirect.com/science/article/pii/S0065327608600192
    # Normalization factors and integral formulae: http://www.chem.unifr.ch/cd/lectures/files/module5.pdf
    # A good PPT overviewing the various algorithms and useful notes: http://www.esqc.org/lectures/WK4.pdf
    # Transformation from Cartesian to Spherical Gaussians https://onlinelibrary.wiley.com/doi/pdf/10.1002/qua.560540202
    # Transformation matrices for Cartesian to Spherical transformation: https://theochem.github.io/horton/2.0.1/tech_ref_gaussian_basis.html
    # Formulae for integration are taken from: https://pubs.acs.org/doi/10.1021/acs.jchemed.8b00255
    # The above formulae are from Taketa, Huzinaga and Ohata https://doi.org/10.1143/JPSJ.21.2313
    # An interesting MS Excel program for HF: https://pubs.acs.org/doi/abs/10.1021/ed085p159
    # Interesting read on numerical methods for integration: https://www.pamoc.it/tpc_num_int.html
    # Calculation of XC integrals, energies and densities (TURBOMOLE) : https://pubs.acs.org/doi/full/10.1021/ct200412r
    # https://iopscience.iop.org/article/10.1088/0143-0807/31/1/004/pdf
    # Thesis on QOL program : https://kups.ub.uni-koeln.de/7772/1/dissertation_joseph-held_09-10-1981.pdf
    # Good PPT with various formulae using Obara Saika recurrence relations: http://www.chem.helsinki.fi/~manninen/aqc2012/Session180412.pdf
    # Very good resource on Quantum Chemistry: http://www.chem.helsinki.fi/~manninen/aqc2012/
    # A good PPT on integrals and stuff: http://www.esqc.org/lectures/WK4.pdf
    # Thesis on OFDFT: https://www.diva-portal.org/smash/get/diva2:864857/FULLTEXT01.pdf

    # For Boys function and 4c2e integrals:
    # https://onlinelibrary.wiley.com/doi/full/10.1002/wcms.78
    # https://www.duo.uio.no/bitstream/handle/10852/12863/Reine_PUBL2.pdf?sequence=5&isAllowed=y
    # https://chemistry.stackexchange.com/questions/97522/two-electron-integral-algorithm
    # https://pubs.acs.org/doi/10.1021/acs.jctc.9b01296


    #Some variables related to benchmarking and profiling
    time_Smat = 0.0
    time_Vmat = 0.0
    time_Tmat = 0.0
    time_Jmat = 0.0
    time_XCmat = 0.0


    def c2k(k,la,lb,PA,PB):
        temp = 0.0
        for i in range(la+1):
            for j in range(lb+1):
                if (i+j)==k :
                    temp = temp + binom(la,i)*binom(lb,j)*PA**(la-i)*PB**(lb-j)
        return temp


    def calcS(la,lb,gamma,PA,PB):
        temp = 0.0
        for k in range(0, int(np.floor((la+lb)/2))+1):
            temp = temp + Integrals.c2k(2*k,la,lb,PA,PB)*np.sqrt(np.pi/gamma)*factorial2(2*k-1)/(2*gamma)**k
        return temp

    def vlri(la,lb,Ai,Bi,Ci,gamma,l,r,i):
        epsilon = 1/(4*gamma)
        return (-1)**l*Integrals.c2k(l,la,lb,Ai,Bi)*((-1)**i*math.factorial(l)*Ci**(l-2*r-2*i)*epsilon**(r+i)/(math.factorial(r)*math.factorial(i)*math.factorial(l-2*r-2*i)))
        
    def sum_vlri(la,lb,Ai,Bi,Ci,gamma):
        sum_v = 0.0
        for l in range(0,la+lb+1):
            for r in range(0, int(np.floor(l/2))+1):
                for i in range(0, int(np.floor((l-2*r)/2))+1):
                    sum_v = sum_v + Integrals.vlri(la,lb,Ai,Bi,Ci,gamma)
        return sum_v
    
    def Fboys(v, x):
    #Calculates the Boys function
    # Fv(x) = \int_0^1 t^2v \exp{-xt^2}dt
    #This is very complicated and as a starting point there are multiple implementation options:
    #https://joshuagoings.com/assets/integrals.pdf Relies on scipy
    #https://github.com/erikkjellgren/SlowQuant/blob/69025884086d5c2a11129cb02766ee5e6aa9e373/MolecularIntegrals.py#L173 
    #https://smartech.gatech.edu/bitstream/handle/1853/55395/BRZYCKI-UNDERGRADUATERESEARCHOPTIONTHESIS-2016.pdf

        #From https://pubs.acs.org/doi/full/10.1021/acs.jchemed.8b00255 supporting information pdf handout 4
        if x >= 0 and x < 0.0000001:      
            F = 1/(2*v+1) - x/(2*v+3)
        else:
            F = 0.5*x**(-(v+0.5))*gammainc(v+0.5,x)*gamma(v+0.5)
        # From jj goings
        # F = hyp1f1(v+0.5,v+1.5,-x)/(2.0*v+1.0)
        return F

    def nucMat(basis, mol, slice=None):
        # This function calculates the nuclear potential matrix for a given basis object and mol object.
        # The basis object holds the information of basis functions like: exponents, coeffs, etc.
        # The mol object is used to get information about the charges of nuclei and their positions.
        # It is here, that we see the advantage of having the mol and basis objects be supplied separately.
        # This allows to calculate the nuclear matrix of one molecule in the basis of another.
        # It is possible to only calculate a slice (block) of the complete matrix.
        # slice is a 4 element list whose first and second elements give the range of the rows to be calculated.
        # the third and fourth element give the range of columns to be calculated.
        # The integrals are performed using the formulas https://pubs.acs.org/doi/full/10.1021/acs.jchemed.8b00255

        # (A|-Z_C/r_{iC}|B) = 

        #If the user doesn't provide a slice then calculate the complete overlap matrix for all the BFs
        if slice==None:
            slice = [0,basis.bfs_nao,0,basis.bfs_nao]
        #V = np.zeros([basis.bfs_nao,basis.bfs_nao])
        V = np.zeros([slice[1]-slice[0],slice[3]-slice[2]])
        #Get the nuclear charges of the molecule
        Z = mol.Zcharges
        #Get the coordinates of molecule's nuclei
        coordsMol = mol.coordsBohrs
        #Get the total number of nuclei
        natoms = mol.natoms
        
        #durationFboys = 0.0
        #Loop pver BFs
        for i in range(slice[0],slice[1]):
            for j in range(slice[2],slice[3]):
                I = basis.bfs_coords[i]
                J = basis.bfs_coords[j]
                IJ = I - J  
                Ni = basis.bfs_contr_prim_norms[i]
                Nj = basis.bfs_contr_prim_norms[j]
                lmni = basis.bfs_lmn[i]
                lmnj = basis.bfs_lmn[j]
                la, ma, na = lmni
                lb, mb, nb = lmnj
                #Loop over primitives
                for ik in range(basis.bfs_nprim[i]):
                    for jk in range(basis.bfs_nprim[j]):
                        dik = basis.bfs_coeffs[i][ik]
                        djk = basis.bfs_coeffs[j][jk] 
                        Nik = basis.bfs_prim_norms[i][ik]
                        Njk = basis.bfs_prim_norms[j][jk]
                        alphaik = basis.bfs_expnts[i][ik]
                        alphajk = basis.bfs_expnts[j][jk]
                        gamma = alphaik + alphajk
                        P = (alphaik*I + alphajk*J)/gamma
                        PI = P - I
                        PJ = P - J

                        Vc = 0.0
                        #Loop over nuclei
                        for iatom in range(natoms):
                            Rc = coordsMol[iatom]
                            Zc = Z[iatom]
                            PC = P - Rc

                            fac1 = -Zc*(2*np.pi/gamma)*np.exp(-alphaik*alphajk/gamma*np.sum(IJ**2))
                            #print(fac1)
                            sum_Vl = 0.0
                            
                            for l in range(0,la+lb+1):
                                for r in range(0, int(l/2)+1):
                                    for i1 in range(0, int((l-2*r)/2)+1):
                                        v_lri = Integrals.vlri(la,lb,PI[0],PJ[0],PC[0],gamma,l,r,i1)#*math.factorial(l)
                                        sum_Vm = 0.0
                                        for m in range(0,ma+mb+1):
                                            for s in range(0, int(m/2)+1):
                                                for j1 in range(0, int((m-2*s)/2)+1):
                                                    v_msj = Integrals.vlri(ma,mb,PI[1],PJ[1],PC[1],gamma,m,s,j1)#*math.factorial(l)
                                                    sum_Vn = 0.0
                                                    for n in range(0,na+nb+1):
                                                        for t in range(0, int(n/2)+1):
                                                            for k in range(0, int((n-2*t)/2)+1):
                                                                v_ntk = Integrals.vlri(na,nb,PI[2],PJ[2],PC[2],gamma,n,t,k)#*math.factorial(l)
                                                                #startFboys = timer()
                                                                F = Integrals.Fboys(l+m+n-2*(r+s+t)-(i1+j1+k),gamma*np.sum(PC**2))
                                                                #durationFboys = durationFboys + timer() - startFboys 
                                                                sum_Vn = sum_Vn + v_ntk*F
                                                    sum_Vm = sum_Vm + v_msj*sum_Vn
                                        sum_Vl = sum_Vl + v_lri*sum_Vm
                            
                            Vc = Vc + sum_Vl*fac1
                        #print(Vc)
                        V[i,j] = V[i,j] + Vc*dik*djk*Nik*Njk*Ni*Nj
                        #print(dik*djk*Nik*Njk*Ni*Nj*Vc)       
                        #print(i,j)                 
                            
        #print(durationFboys)
        return V

    def theta(l,la,lb,PA,PB,gamma,r):
        return Integrals.c2k(l,la,lb,PA,PB)*math.factorial(l)*(gamma**(r-l))/math.factorial(r)/math.factorial(l-2*r)

    def g(lp,lq,rp,rq,i,la,lb,lc,ld,gammaP,gammaQ,PA,PB,QC,QD,PQ,delta):
        temp = ((-1)**lp)*Integrals.theta(lp,la,lb,PA,PB,gammaP,rp)*Integrals.theta(lq,lc,ld,QC,QD,gammaQ,rq)
        numerator = temp*((-1)**i)*((2*delta)**(2*(rp+rq)))*math.factorial(lp+lq-2*rp-2*rq)*(delta**i)*(PQ**(lp+lq-2*(rp+rq+i)))
        denominator = ((4*delta)**(lp+lq))*math.factorial(i)*math.factorial(lp+lq-2*(rp+rq+i))
        # print(numerator/temp)
        return (numerator/denominator)

    def fourCenterTwoElec(basis, slice=None):
        # This function calculates the electron-electron Coulomb potential matrix for a given basis object and mol object.
        # The basis object holds the information of basis functions like: exponents, coeffs, etc.
        
        # It is possible to only calculate a slice (block) of the complete matrix.
        # slice is a 4 element list whose first and second elements give the range of the rows to be calculated.
        # the third and fourth element give the range of columns to be calculated.
        # The integrals are performed using the formulas https://pubs.acs.org/doi/full/10.1021/acs.jchemed.8b00255
        # Some useful resources:
        # https://chemistry.stackexchange.com/questions/71527/how-does-one-compute-the-number-of-unique-2-electron-integrals-for-a-given-basis
        # https://chemistry.stackexchange.com/questions/82532/practical-differences-between-storing-2-electron-integrals-and-calculating-them
        # Density matrix based screening Section (5.1.2): https://www.diva-portal.org/smash/get/diva2:740301/FULLTEXT02.pdf
        # Lots of notes on screening and efficiencies (3.3 pg 31): https://www.zora.uzh.ch/id/eprint/44716/1/Diss.123456.pdf
        # In the above resource in section 3.4 it also says that storing in 64 bit floating point precision is usually not required. 
        # So, the storage requirements can be reduced by 4 to 8 times.

       

        # returns (AB|CD) 
        import itertools

        #If the user doesn't provide a slice then calculate the complete overlap matrix for all the BFs
        if slice==None:
            slice = [0,basis.bfs_nao,0,basis.bfs_nao,0,basis.bfs_nao,0,basis.bfs_nao]
        
        fourC2E = np.zeros([slice[1]-slice[0],slice[3]-slice[2],slice[5]-slice[4],slice[7]-slice[6]])
        durationFboys = 0.0
        
        
        #Loop pver BFs
        for i,j,k,l in itertools.product(range(slice[0],slice[1]), range(slice[2],slice[3]), range(slice[4],slice[5]), range(slice[6],slice[7])):
            I = basis.bfs_coords[i]
            J = basis.bfs_coords[j]
            K = basis.bfs_coords[k]
            L = basis.bfs_coords[l]
            IJ = I - J
            KL = K - L  
            Ni = basis.bfs_contr_prim_norms[i]
            Nj = basis.bfs_contr_prim_norms[j]
            Nk = basis.bfs_contr_prim_norms[k]
            Nl = basis.bfs_contr_prim_norms[l]
            lmni = basis.bfs_lmn[i]
            lmnj = basis.bfs_lmn[j]
            lmnk = basis.bfs_lmn[k]
            lmnl = basis.bfs_lmn[l]
            la, ma, na = lmni
            lb, mb, nb = lmnj
            lc, mc, nc = lmnk
            ld, md, nd = lmnl
            #Loop over primitives
            for ik in range(basis.bfs_nprim[i]):
                for jk in range(basis.bfs_nprim[j]):
                    for kk in range(basis.bfs_nprim[k]):
                        for lk in range(basis.bfs_nprim[l]):
                            dik = basis.bfs_coeffs[i][ik]
                            djk = basis.bfs_coeffs[j][jk] 
                            dkk = basis.bfs_coeffs[k][kk]
                            dlk = basis.bfs_coeffs[l][lk] 
                            Nik = basis.bfs_prim_norms[i][ik]
                            Njk = basis.bfs_prim_norms[j][jk]
                            Nkk = basis.bfs_prim_norms[k][kk]
                            Nlk = basis.bfs_prim_norms[l][lk]
                            alphaik = basis.bfs_expnts[i][ik]
                            alphajk = basis.bfs_expnts[j][jk]
                            alphakk = basis.bfs_expnts[k][kk]
                            alphalk = basis.bfs_expnts[l][lk]
                            gammaP = alphaik + alphajk
                            gammaQ = alphakk + alphalk
                            P = (alphaik*I + alphajk*J)/gammaP
                            Q = (alphakk*K + alphalk*L)/gammaQ
                            PI = P - I
                            PJ = P - J
                            PQ = P - Q
                            QK = Q - K
                            QL = Q - L

                            omega = (2*np.pi**2/gammaP/gammaQ)*np.sqrt(np.pi/(gammaP + gammaQ))*np.exp(-alphaik*alphajk/gammaP*np.sum(IJ**2))*np.exp(-alphakk*alphalk/gammaQ*np.sum(KL**2))
                            delta = 0.25*(1/gammaP + 1/gammaQ) 
                            # print(omega)                
                                        
                            sum1 = 0.0
                            for lp in range(0,la+lb+1):
                                for rp in range(0, int(lp/2)+1):
                                    for lq in range(0, lc+ld+1):
                                        for rq in range(0, int(lq/2)+1):
                                            for i1 in range(0,int((lp+lq-2*rp-2*rq)/2)+1):
                                                gx = Integrals.g(lp,lq,rp,rq,i1,la,lb,lc,ld,gammaP,gammaQ,PI[0],PJ[0],QK[0],QL[0],PQ[0],delta)
                                                sum2 = 0.0

                                                for mp in range(0,ma+mb+1):
                                                    for sp in range(0, int(mp/2)+1):
                                                        for mq in range(0, mc+md+1):
                                                            for sq in range(0, int(mq/2)+1):
                                                                for j1 in range(0,int((mp+mq-2*sp-2*sq)/2)+1):
                                                                    gy = Integrals.g(mp,mq,sp,sq,j1,ma,mb,mc,md,gammaP,gammaQ,PI[1],PJ[1],QK[1],QL[1],PQ[1],delta)
                                                                    sum3 = 0.0
                                                                    
                                                                    for np1 in range(0,na+nb+1):
                                                                        for tp in range(0, int(np1/2)+1):
                                                                            for nq in range(0, nc+nd+1):
                                                                                for tq in range(0, int(nq/2)+1):
                                                                                    for k1 in range(0,int((np1+nq-2*tp-2*tq)/2)+1):
                                                                                        gz = Integrals.g(np1,nq,tp,tq,k1,na,nb,nc,nd,gammaP,gammaQ,PI[2],PJ[2],QK[2],QL[2],PQ[2],delta)
                                                                                        v = lp+lq+mp+mq+np1+nq-2*(rp+rq+sp+sq+tp+tq)-(i1+j1+k1)
                                                                                        # start  = timer()
                                                                                        F = Integrals.Fboys(v,np.sum(PQ**2)/(4*delta))
                                                                                        # durationFboys = durationFboys + timer() - start
                                                                                        sum3 = sum3 + gz*F
                                                                                        #sum1 = sum1 + gx*gy*gz*F
                                                                                        
                                                                    sum2 = sum2 + gy*sum3
                                                sum1 = sum1 + gx*sum2
                            PQsqBy4delta = np.sum(PQ**2)/(4*delta) 
                            # sum12 = innerLoop4c2eNumba1(la,lb,lc,ld,ma,mb,mc,md,na,nb,nc,nd,gammaP,gammaQ,PI,PJ,QK,QL,PQ,PQsqBy4delta,delta) 
                            # if(abs(sum1-sum12)>1.0e-10): 
                            #     print('yes')
                            #     print(sum1)
                            #     print(sum12)
                            fourC2E[i,j,k,l] = fourC2E[i,j,k,l] + dik*djk*dkk*dlk*Nik*Njk*Nkk*Nlk*Ni*Nj*Nk*Nl*omega*sum1
        # for i in range(slice[0],slice[1]):
        #     for j in range(slice[2],slice[3]):
        #         for k in range(slice[4],slice[5]):
        #             for l in range(slice[6],slice[7]):
        #                 I = basis.bfs_coords[i]
        #                 J = basis.bfs_coords[j]
        #                 K = basis.bfs_coords[k]
        #                 L = basis.bfs_coords[l]
        #                 IJ = I - J
        #                 KL = K - L  
        #                 Ni = basis.bfs_contr_prim_norms[i]
        #                 Nj = basis.bfs_contr_prim_norms[j]
        #                 Nk = basis.bfs_contr_prim_norms[k]
        #                 Nl = basis.bfs_contr_prim_norms[l]
        #                 lmni = basis.bfs_lmn[i]
        #                 lmnj = basis.bfs_lmn[j]
        #                 lmnk = basis.bfs_lmn[k]
        #                 lmnl = basis.bfs_lmn[l]
        #                 la, ma, na = lmni
        #                 lb, mb, nb = lmnj
        #                 lc, mc, nc = lmnk
        #                 ld, md, nd = lmnl
        #                 #Loop over primitives
        #                 for ik in range(basis.bfs_nprim[i]):
        #                     for jk in range(basis.bfs_nprim[j]):
        #                         for kk in range(basis.bfs_nprim[k]):
        #                             for lk in range(basis.bfs_nprim[l]):
        #                                 dik = basis.bfs_coeffs[i][ik]
        #                                 djk = basis.bfs_coeffs[j][jk] 
        #                                 dkk = basis.bfs_coeffs[k][kk]
        #                                 dlk = basis.bfs_coeffs[l][lk] 
        #                                 Nik = basis.bfs_prim_norms[i][ik]
        #                                 Njk = basis.bfs_prim_norms[j][jk]
        #                                 Nkk = basis.bfs_prim_norms[k][kk]
        #                                 Nlk = basis.bfs_prim_norms[l][lk]
        #                                 alphaik = basis.bfs_expnts[i][ik]
        #                                 alphajk = basis.bfs_expnts[j][jk]
        #                                 alphakk = basis.bfs_expnts[k][kk]
        #                                 alphalk = basis.bfs_expnts[l][lk]
        #                                 gammaP = alphaik + alphajk
        #                                 gammaQ = alphakk + alphalk
        #                                 P = (alphaik*I + alphajk*J)/gammaP
        #                                 Q = (alphakk*K + alphalk*L)/gammaQ
        #                                 PI = P - I
        #                                 PJ = P - J
        #                                 PQ = P - Q
        #                                 QK = Q - K
        #                                 QL = Q - L

        #                                 omega = (2*np.pi**2/gammaP/gammaQ)*np.sqrt(np.pi/(gammaP + gammaQ))*np.exp(-alphaik*alphajk/gammaP*IJ**2)*np.exp(-alphakk*alphalk/gammaQ*KL**2)
        #                                 delta = 0.25*(1/gammaP + 1/gammaQ)                    
                                        
        #                                 sum1 = 0.0
        #                                 for lp in range(0,la+lb+1):
        #                                     for rp in range(0, int(l/2)+1):
        #                                         for lq in range(0, int(l/2)+1):
        #                                             for rq in range(0, int(l/2)+1):
        #                                                 for i1 in range():
        #                                                     gx = g(lp,lq,rp,rq,i1,la,lb,lc,ld,gammaP,gammaQ,PI[0],PJ[0],QK[0],QL[0],PQ[0],delta)
        #                                                     sum2 = 0.0
        #                                                     for mp in range(0,la+lb+1):
        #                                                         for sp in range(0, int(l/2)+1):
        #                                                             for mq in range(0, int(l/2)+1):
        #                                                                 for sq in range(0, int(l/2)+1):
        #                                                                     for j1 in range():
        #                                                                         gy = g(mp,mq,sp,sq,j1,ma,mb,mc,md,gammaP,gammaQ,PI[1],PJ[1],QK[1],QL[1],PQ[1],delta)
        #                                                                         sum3 = 0.0
        #                                                                         for np in range(0,la+lb+1):
        #                                                                             for tp in range(0, int(l/2)+1):
        #                                                                                 for nq in range(0, int(l/2)+1):
        #                                                                                     for tq in range(0, int(l/2)+1):
        #                                                                                         for k1 in range():
        #                                                                                             gz = g(mp,mq,tp,tq,k1,na,nb,nc,nd,gammaP,gammaQ,PI[2],PJ[2],QK[2],QL[2],PQ[2],delta)
        #                                                                                             v = lp+lq+mp+mq+np+nq-2*(rp+rq+sp+sq+tp+tq)-(i1+j1+k1)
        #                                                                                             F = Integrals.Fboys(v,np.sum(PQ**2)/4/delta)
        #                                                                                             sum3 = sum3 + gz*F
        #                                                                         sum2 = sum2 + gy*sum3
        #                                                     sum1 = sum1 + gx*sum2
                                        
                                        
        #                                 fourC2E[i,j,k,l] = fourC2E[i,j,k,l] + dik*djk*dkk*dlk*Nik*Njk*Nkk*Nlk*Ni*Nj*Nk*Nl*omega*sum1
                                       
                            
        # print(durationFboys)
        return fourC2E
    
    def fourCenterTwoElecNumbawrap(basis, slice1=None):
        # This is robust Numba implementation
        # Here the lists are converted to numpy arrays for better use with Numba.
        # Once these conversions are done we pass these to a Numba decorated
        # function that uses prange, etc. to calculate the 4c2e integrals efficiently.
        # This function calculates the electron-electron Coulomb potential matrix for a given basis object and mol object.
        # The basis object holds the information of basis functions like: exponents, coeffs, etc.
        
        # It is possible to only calculate a slice (block) of the complete matrix.
        # slice is a 4 element list whose first and second elements give the range of the rows to be calculated.
        # the third and fourth element give the range of columns to be calculated.
        # The integrals are performed using the formulas https://pubs.acs.org/doi/full/10.1021/acs.jchemed.8b00255
        # Some useful resources:
        # https://chemistry.stackexchange.com/questions/71527/how-does-one-compute-the-number-of-unique-2-electron-integrals-for-a-given-basis
        # https://chemistry.stackexchange.com/questions/82532/practical-differences-between-storing-2-electron-integrals-and-calculating-them
        # Density matrix based screening Section (5.1.2): https://www.diva-portal.org/smash/get/diva2:740301/FULLTEXT02.pdf
        # Lots of notes on screening and efficiencies (3.3 pg 31): https://www.zora.uzh.ch/id/eprint/44716/1/Diss.123456.pdf
        # In the above resource in section 3.4 it also says that storing in 64 bit floating point precision is usually not required. 
        # So, the storage requirements can be reduced by 4 to 8 times.
        

        # This function calculates the 4c2e integrals for a given basis object.
        # The basis object holds the information of basis functions like: exponents, coeffs, etc.
        # It is possible to only calculate a slice (block) of the complete set of integrals.
        # slice is an 8 element list whose first and second elements give the range of the A functions to be calculated.
        # and so on.

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
            


        #If the user doesn't provide a slice then calculate the complete set of 4c2e integrals for all the BFs
        if slice1==None:
            slice1 = [0,basis.bfs_nao,0,basis.bfs_nao,0,basis.bfs_nao,0,basis.bfs_nao]
            
        #Limits for the calculation of 4c2e integrals
        indx_startA = int(slice1[0])
        indx_endA = int(slice1[1])
        indx_startB = int(slice1[2])
        indx_endB = int(slice1[3])
        indx_startC = int(slice1[4])
        indx_endC = int(slice1[5])
        indx_startD = int(slice1[6])
        indx_endD = int(slice1[7])

        ints4c2e = fourCenterTwoElecNumba2(bfs_coords[0], bfs_contr_prim_norms[0], bfs_lmn[0], bfs_nprim[0], bfs_coeffs, bfs_prim_norms, bfs_expnts,indx_startA,indx_endA,indx_startB,indx_endB,indx_startC,indx_endC,indx_startD,indx_endD)

        return ints4c2e

    def fourCenterTwoElecSymmNumbawrap(basis):
        # This is robust Numba implementation
        # Here the lists are converted to numpy arrays for better use with Numba.
        # Once these conversions are done we pass these to a Numba decorated
        # function that uses prange, etc. to calculate the 4c2e integrals efficiently.
        # This function calculates the electron-electron Coulomb potential matrix for a given basis object and mol object.
        # The basis object holds the information of basis functions like: exponents, coeffs, etc.
        
        # The integrals are performed using the formulas https://pubs.acs.org/doi/full/10.1021/acs.jchemed.8b00255
        # Some useful resources:
        # https://chemistry.stackexchange.com/questions/71527/how-does-one-compute-the-number-of-unique-2-electron-integrals-for-a-given-basis
        # https://chemistry.stackexchange.com/questions/82532/practical-differences-between-storing-2-electron-integrals-and-calculating-them
        # Density matrix based screening Section (5.1.2): https://www.diva-portal.org/smash/get/diva2:740301/FULLTEXT02.pdf
        # Lots of notes on screening and efficiencies (3.3 pg 31): https://www.zora.uzh.ch/id/eprint/44716/1/Diss.123456.pdf
        # In the above resource in section 3.4 it also says that storing in 64 bit floating point precision is usually not required. 
        # So, the storage requirements can be reduced by 4 to 8 times.
        

        # This function calculates the 4c2e integrals for a given basis object.
        # The basis object holds the information of basis functions like: exponents, coeffs, etc.
        # It is possible to only calculate a slice (block) of the complete set of integrals.
        # slice is an 8 element list whose first and second elements give the range of the A functions to be calculated.
        # and so on.

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
            


        
        slice1 = [0,basis.bfs_nao,0,basis.bfs_nao,0,basis.bfs_nao,0,basis.bfs_nao]
            
        #Limits for the calculation of 4c2e integrals
        indx_startA = int(slice1[0])
        indx_endA = int(slice1[1])
        indx_startB = int(slice1[2])
        indx_endB = int(slice1[3])
        indx_startC = int(slice1[4])
        indx_endC = int(slice1[5])
        indx_startD = int(slice1[6])
        indx_endD = int(slice1[7])

        ints4c2e = fourCenterTwoElecSymmNumba2(bfs_coords[0], bfs_contr_prim_norms[0], bfs_lmn[0], bfs_nprim[0], bfs_coeffs, bfs_prim_norms, bfs_expnts,indx_startA,indx_endA,indx_startB,indx_endB,indx_startC,indx_endC,indx_startD,indx_endD)

        return ints4c2e

    def fourCenterTwoElecDiagSymmNumbawrap(basis):
        # Used for Schwarz inequality test

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
            



        ints4c2e_diag = eri_4c2e_diagonal_numba2(bfs_coords[0], bfs_contr_prim_norms[0], bfs_lmn[0], bfs_nprim[0], bfs_coeffs, bfs_prim_norms, bfs_expnts)

        return ints4c2e_diag
    @njit(parallel=True, cache=True)
    def calc_indices_3c2e_schwarz(eri_4c2e_diag, ints2c2e, nao, naux, threshold):
        # This function will return a numpy array of the same size as ints3c2e array (nao*nao*naux)
        # The array will have a value of 1 where there is a significant contribution and 0 otherwise.
        # Later on this array can be used to find the indices of non-zero arrays
        indices = np.zeros((nao, nao, naux), dtype=np.uint8)
        # Loop over the lower-triangular ints3c2e array
        for i in range(nao):
            for j in range(i+1):
                for k in prange(naux):
                    if np.sqrt(eri_4c2e_diag[i,j])*np.sqrt(ints2c2e[k,k])>threshold:
                        indices[i,j,k] = 1
        return indices
                        

    def RystwoCenterTwoElecSymmNumbawrap(basis):
        # This is robust Numba implementation
        # Here the lists are converted to numpy arrays for better use with Numba.
        # Once these conversions are done we pass these to a Numba decorated
        # function that uses prange, etc. to calculate the 4c2e integrals efficiently.
        # This function calculates the electron-electron Coulomb potential matrix for a given basis object and mol object.
        # The basis object holds the information of basis functions like: exponents, coeffs, etc.
        
        # The integrals are performed using the formulas https://pubs.acs.org/doi/full/10.1021/acs.jchemed.8b00255
        # Some useful resources:
        # https://chemistry.stackexchange.com/questions/71527/how-does-one-compute-the-number-of-unique-2-electron-integrals-for-a-given-basis
        # https://chemistry.stackexchange.com/questions/82532/practical-differences-between-storing-2-electron-integrals-and-calculating-them
        # Density matrix based screening Section (5.1.2): https://www.diva-portal.org/smash/get/diva2:740301/FULLTEXT02.pdf
        # Lots of notes on screening and efficiencies (3.3 pg 31): https://www.zora.uzh.ch/id/eprint/44716/1/Diss.123456.pdf
        # In the above resource in section 3.4 it also says that storing in 64 bit floating point precision is usually not required. 
        # So, the storage requirements can be reduced by 4 to 8 times.
        

        # This function calculates the 4c2e integrals for a given basis object.
        # The basis object holds the information of basis functions like: exponents, coeffs, etc.
        # It is possible to only calculate a slice (block) of the complete set of integrals.
        # slice is an 8 element list whose first and second elements give the range of the A functions to be calculated.
        # and so on.

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
            


        
        slice = [0,basis.bfs_nao,0,basis.bfs_nao,0]
            
        #Limits for the calculation of overlap integrals
        a = int(slice[0]) #row start index
        b = int(slice[1]) #row end index
        c = int(slice[2]) #column start index
        d = int(slice[3]) #column end index

        ints2c2e = rys2c2eSymmNumba2(bfs_coords[0], bfs_contr_prim_norms[0], bfs_lmn[0], bfs_nprim[0], bfs_coeffs, bfs_prim_norms, bfs_expnts,a,b,c,d)
        return ints2c2e
    
    def twoCenterTwoElecSymmNumbawrap(basis):
        # This is robust Numba implementation
        # Here the lists are converted to numpy arrays for better use with Numba.
        # Once these conversions are done we pass these to a Numba decorated
        # function that uses prange, etc. to calculate the 4c2e integrals efficiently.
        # This function calculates the electron-electron Coulomb potential matrix for a given basis object and mol object.
        # The basis object holds the information of basis functions like: exponents, coeffs, etc.
        
        # The integrals are performed using the formulas https://pubs.acs.org/doi/full/10.1021/acs.jchemed.8b00255
        # Some useful resources:
        # https://chemistry.stackexchange.com/questions/71527/how-does-one-compute-the-number-of-unique-2-electron-integrals-for-a-given-basis
        # https://chemistry.stackexchange.com/questions/82532/practical-differences-between-storing-2-electron-integrals-and-calculating-them
        # Density matrix based screening Section (5.1.2): https://www.diva-portal.org/smash/get/diva2:740301/FULLTEXT02.pdf
        # Lots of notes on screening and efficiencies (3.3 pg 31): https://www.zora.uzh.ch/id/eprint/44716/1/Diss.123456.pdf
        # In the above resource in section 3.4 it also says that storing in 64 bit floating point precision is usually not required. 
        # So, the storage requirements can be reduced by 4 to 8 times.
        

        # This function calculates the 4c2e integrals for a given basis object.
        # The basis object holds the information of basis functions like: exponents, coeffs, etc.
        # It is possible to only calculate a slice (block) of the complete set of integrals.
        # slice is an 8 element list whose first and second elements give the range of the A functions to be calculated.
        # and so on.

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
            


        
        slice = [0,basis.bfs_nao,0,basis.bfs_nao,0]
            
        #Limits for the calculation of overlap integrals
        a = int(slice[0]) #row start index
        b = int(slice[1]) #row end index
        c = int(slice[2]) #column start index
        d = int(slice[3]) #column end index

        ints2c2e = twoCenterTwoElecSymmNumba2(bfs_coords[0], bfs_contr_prim_norms[0], bfs_lmn[0], bfs_nprim[0], bfs_coeffs, bfs_prim_norms, bfs_expnts,a,b,c,d)
        return ints2c2e

    def RysthreeCenterTwoElecSymmNumbawrap(basis, auxbasis):
        # This is robust Numba implementation
        # Here the lists are converted to numpy arrays for better use with Numba.
        # Once these conversions are done we pass these to a Numba decorated
        # function that uses prange, etc. to calculate the 4c2e integrals efficiently.
        # This function calculates the electron-electron Coulomb potential matrix for a given basis object and mol object.
        # The basis object holds the information of basis functions like: exponents, coeffs, etc.
        
        # The integrals are performed using the formulas https://pubs.acs.org/doi/full/10.1021/acs.jchemed.8b00255
        # Some useful resources:
        # https://chemistry.stackexchange.com/questions/71527/how-does-one-compute-the-number-of-unique-2-electron-integrals-for-a-given-basis
        # https://chemistry.stackexchange.com/questions/82532/practical-differences-between-storing-2-electron-integrals-and-calculating-them
        # Density matrix based screening Section (5.1.2): https://www.diva-portal.org/smash/get/diva2:740301/FULLTEXT02.pdf
        # Lots of notes on screening and efficiencies (3.3 pg 31): https://www.zora.uzh.ch/id/eprint/44716/1/Diss.123456.pdf
        # In the above resource in section 3.4 it also says that storing in 64 bit floating point precision is usually not required. 
        # So, the storage requirements can be reduced by 4 to 8 times.
        

        # This function calculates the 4c2e integrals for a given basis object.
        # The basis object holds the information of basis functions like: exponents, coeffs, etc.
        # It is possible to only calculate a slice (block) of the complete set of integrals.
        # slice is an 8 element list whose first and second elements give the range of the A functions to be calculated.
        # and so on.

        #We convert the required properties to numpy arrays as this is what Numba likes.
        bfs_coords = np.array([basis.bfs_coords])
        bfs_contr_prim_norms = np.array([basis.bfs_contr_prim_norms])
        bfs_lmn = np.array([basis.bfs_lmn])
        bfs_nprim = np.array([basis.bfs_nprim])

        #We convert the required properties to numpy arrays as this is what Numba likes.
        aux_bfs_coords = np.array([auxbasis.bfs_coords])
        aux_bfs_contr_prim_norms = np.array([auxbasis.bfs_contr_prim_norms])
        aux_bfs_lmn = np.array([auxbasis.bfs_lmn])
        aux_bfs_nprim = np.array([auxbasis.bfs_nprim])
           

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

        maxnprimaux = max(auxbasis.bfs_nprim)
        aux_bfs_coeffs = np.zeros([auxbasis.bfs_nao, maxnprimaux])
        aux_bfs_expnts = np.zeros([auxbasis.bfs_nao, maxnprimaux])
        aux_bfs_prim_norms = np.zeros([auxbasis.bfs_nao, maxnprimaux])
        for i in range(auxbasis.bfs_nao):
            for j in range(auxbasis.bfs_nprim[i]):
                aux_bfs_coeffs[i,j] = auxbasis.bfs_coeffs[i][j]
                aux_bfs_expnts[i,j] = auxbasis.bfs_expnts[i][j]
                aux_bfs_prim_norms[i,j] = auxbasis.bfs_prim_norms[i][j]
            


        
        slice = [0,basis.bfs_nao,0,basis.bfs_nao,0,auxbasis.bfs_nao]
            
        #Limits for the calculation of overlap integrals
        a = int(slice[0]) #row start index
        b = int(slice[1]) #row end index
        c = int(slice[2]) #column start index
        d = int(slice[3]) #column end index
        e = int(slice[4])
        f = int(slice[5])

        ints3c2e = rys3c2eSymmNumba2(bfs_coords[0], bfs_contr_prim_norms[0], bfs_lmn[0], bfs_nprim[0], bfs_coeffs, bfs_prim_norms, bfs_expnts,aux_bfs_coords[0], aux_bfs_contr_prim_norms[0], aux_bfs_lmn[0], aux_bfs_nprim[0], aux_bfs_coeffs, aux_bfs_prim_norms, aux_bfs_expnts,a,b,c,d,e,f)
        # ints3c2e = threeCenterTwoElecSymmNumba2(bfs_coords[0], bfs_contr_prim_norms[0], bfs_lmn[0], bfs_nprim[0], bfs_coeffs, bfs_prim_norms, bfs_expnts,aux_bfs_coords[0], aux_bfs_contr_prim_norms[0], aux_bfs_lmn[0], aux_bfs_nprim[0], aux_bfs_coeffs, aux_bfs_prim_norms, aux_bfs_expnts,a,b,c,d,e,f)
        # ints3c2e = rys3c2eSymmNumba_tri2(bfs_coords[0], bfs_contr_prim_norms[0], bfs_lmn[0], bfs_nprim[0], bfs_coeffs, bfs_prim_norms, bfs_expnts,aux_bfs_coords[0], aux_bfs_contr_prim_norms[0], aux_bfs_lmn[0], aux_bfs_nprim[0], aux_bfs_coeffs, aux_bfs_prim_norms, aux_bfs_expnts,a,b,c,d,e,f)
        return ints3c2e
    
    def RysthreeCenterTwoElecSymmTriNumbawrap(basis, auxbasis):
        # This is robust Numba implementation
        # Here the lists are converted to numpy arrays for better use with Numba.
        # Once these conversions are done we pass these to a Numba decorated
        # function that uses prange, etc. to calculate the 4c2e integrals efficiently.
        # This function calculates the electron-electron Coulomb potential matrix for a given basis object and mol object.
        # The basis object holds the information of basis functions like: exponents, coeffs, etc.
        
        # The integrals are performed using the formulas https://pubs.acs.org/doi/full/10.1021/acs.jchemed.8b00255
        # Some useful resources:
        # https://chemistry.stackexchange.com/questions/71527/how-does-one-compute-the-number-of-unique-2-electron-integrals-for-a-given-basis
        # https://chemistry.stackexchange.com/questions/82532/practical-differences-between-storing-2-electron-integrals-and-calculating-them
        # Density matrix based screening Section (5.1.2): https://www.diva-portal.org/smash/get/diva2:740301/FULLTEXT02.pdf
        # Lots of notes on screening and efficiencies (3.3 pg 31): https://www.zora.uzh.ch/id/eprint/44716/1/Diss.123456.pdf
        # In the above resource in section 3.4 it also says that storing in 64 bit floating point precision is usually not required. 
        # So, the storage requirements can be reduced by 4 to 8 times.
        

        # This function calculates the 4c2e integrals for a given basis object.
        # The basis object holds the information of basis functions like: exponents, coeffs, etc.
        # It is possible to only calculate a slice (block) of the complete set of integrals.
        # slice is an 8 element list whose first and second elements give the range of the A functions to be calculated.
        # and so on.

        #We convert the required properties to numpy arrays as this is what Numba likes.
        bfs_coords = np.array([basis.bfs_coords])
        bfs_contr_prim_norms = np.array([basis.bfs_contr_prim_norms])
        bfs_lmn = np.array([basis.bfs_lmn])
        bfs_nprim = np.array([basis.bfs_nprim])

        #We convert the required properties to numpy arrays as this is what Numba likes.
        aux_bfs_coords = np.array([auxbasis.bfs_coords])
        aux_bfs_contr_prim_norms = np.array([auxbasis.bfs_contr_prim_norms])
        aux_bfs_lmn = np.array([auxbasis.bfs_lmn])
        aux_bfs_nprim = np.array([auxbasis.bfs_nprim])
           

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

        maxnprimaux = max(auxbasis.bfs_nprim)
        aux_bfs_coeffs = np.zeros([auxbasis.bfs_nao, maxnprimaux])
        aux_bfs_expnts = np.zeros([auxbasis.bfs_nao, maxnprimaux])
        aux_bfs_prim_norms = np.zeros([auxbasis.bfs_nao, maxnprimaux])
        for i in range(auxbasis.bfs_nao):
            for j in range(auxbasis.bfs_nprim[i]):
                aux_bfs_coeffs[i,j] = auxbasis.bfs_coeffs[i][j]
                aux_bfs_expnts[i,j] = auxbasis.bfs_expnts[i][j]
                aux_bfs_prim_norms[i,j] = auxbasis.bfs_prim_norms[i][j]
            


        
        slice = [0,basis.bfs_nao,0,basis.bfs_nao,0,auxbasis.bfs_nao]
            
        #Limits for the calculation of overlap integrals
        a = int(slice[0]) #row start index
        b = int(slice[1]) #row end index
        c = int(slice[2]) #column start index
        d = int(slice[3]) #column end index
        e = int(slice[4])
        f = int(slice[5])

        ints3c2e = rys3c2eSymmNumba_tri2(bfs_coords[0], bfs_contr_prim_norms[0], bfs_lmn[0], bfs_nprim[0], bfs_coeffs, bfs_prim_norms, bfs_expnts,aux_bfs_coords[0], aux_bfs_contr_prim_norms[0], aux_bfs_lmn[0], aux_bfs_nprim[0], aux_bfs_coeffs, aux_bfs_prim_norms, aux_bfs_expnts,a,b,c,d,e,f)
        return ints3c2e

    def RysthreeCenterTwoElecSymmTriSchwarzNumbawrap(basis, auxbasis, indicesA, indicesB, indicesC):
        # Wrapper for hybrid Rys+conv. 3c2e integral calculator
        # using a list of significant contributions obtained via Schwarz screening.
        # It returns the 3c2e integrals in triangular form.

        #We convert the required properties to numpy arrays as this is what Numba likes.
        bfs_coords = np.array([basis.bfs_coords])
        bfs_contr_prim_norms = np.array([basis.bfs_contr_prim_norms])
        bfs_lmn = np.array([basis.bfs_lmn])
        bfs_nprim = np.array([basis.bfs_nprim])

        #We convert the required properties to numpy arrays as this is what Numba likes.
        aux_bfs_coords = np.array([auxbasis.bfs_coords])
        aux_bfs_contr_prim_norms = np.array([auxbasis.bfs_contr_prim_norms])
        aux_bfs_lmn = np.array([auxbasis.bfs_lmn])
        aux_bfs_nprim = np.array([auxbasis.bfs_nprim])
           

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

        maxnprimaux = max(auxbasis.bfs_nprim)
        aux_bfs_coeffs = np.zeros([auxbasis.bfs_nao, maxnprimaux])
        aux_bfs_expnts = np.zeros([auxbasis.bfs_nao, maxnprimaux])
        aux_bfs_prim_norms = np.zeros([auxbasis.bfs_nao, maxnprimaux])
        for i in range(auxbasis.bfs_nao):
            for j in range(auxbasis.bfs_nprim[i]):
                aux_bfs_coeffs[i,j] = auxbasis.bfs_coeffs[i][j]
                aux_bfs_expnts[i,j] = auxbasis.bfs_expnts[i][j]
                aux_bfs_prim_norms[i,j] = auxbasis.bfs_prim_norms[i][j]
            


        

        ints3c2e = rys3c2eSymmNumba_tri_schwarz2(bfs_coords[0], bfs_contr_prim_norms[0], bfs_lmn[0], \
                  bfs_nprim[0], bfs_coeffs, bfs_prim_norms, bfs_expnts,aux_bfs_coords[0], aux_bfs_contr_prim_norms[0], aux_bfs_lmn[0], aux_bfs_nprim[0], \
                  aux_bfs_coeffs, aux_bfs_prim_norms, aux_bfs_expnts, indicesA, indicesB, indicesC, basis.bfs_nao, auxbasis.bfs_nao)
        return ints3c2e

    def threeCenterTwoElecSymmNumbawrap(basis, auxbasis):
        # This is robust Numba implementation
        # Here the lists are converted to numpy arrays for better use with Numba.
        # Once these conversions are done we pass these to a Numba decorated
        # function that uses prange, etc. to calculate the 4c2e integrals efficiently.
        # This function calculates the electron-electron Coulomb potential matrix for a given basis object and mol object.
        # The basis object holds the information of basis functions like: exponents, coeffs, etc.
        
        # The integrals are performed using the formulas https://pubs.acs.org/doi/full/10.1021/acs.jchemed.8b00255
        # Some useful resources:
        # https://chemistry.stackexchange.com/questions/71527/how-does-one-compute-the-number-of-unique-2-electron-integrals-for-a-given-basis
        # https://chemistry.stackexchange.com/questions/82532/practical-differences-between-storing-2-electron-integrals-and-calculating-them
        # Density matrix based screening Section (5.1.2): https://www.diva-portal.org/smash/get/diva2:740301/FULLTEXT02.pdf
        # Lots of notes on screening and efficiencies (3.3 pg 31): https://www.zora.uzh.ch/id/eprint/44716/1/Diss.123456.pdf
        # In the above resource in section 3.4 it also says that storing in 64 bit floating point precision is usually not required. 
        # So, the storage requirements can be reduced by 4 to 8 times.
        

        # This function calculates the 4c2e integrals for a given basis object.
        # The basis object holds the information of basis functions like: exponents, coeffs, etc.
        # It is possible to only calculate a slice (block) of the complete set of integrals.
        # slice is an 8 element list whose first and second elements give the range of the A functions to be calculated.
        # and so on.

        #We convert the required properties to numpy arrays as this is what Numba likes.
        bfs_coords = np.array([basis.bfs_coords])
        bfs_contr_prim_norms = np.array([basis.bfs_contr_prim_norms])
        bfs_lmn = np.array([basis.bfs_lmn])
        bfs_nprim = np.array([basis.bfs_nprim])

        #We convert the required properties to numpy arrays as this is what Numba likes.
        aux_bfs_coords = np.array([auxbasis.bfs_coords])
        aux_bfs_contr_prim_norms = np.array([auxbasis.bfs_contr_prim_norms])
        aux_bfs_lmn = np.array([auxbasis.bfs_lmn])
        aux_bfs_nprim = np.array([auxbasis.bfs_nprim])
           

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

        maxnprimaux = max(auxbasis.bfs_nprim)
        aux_bfs_coeffs = np.zeros([auxbasis.bfs_nao, maxnprimaux])
        aux_bfs_expnts = np.zeros([auxbasis.bfs_nao, maxnprimaux])
        aux_bfs_prim_norms = np.zeros([auxbasis.bfs_nao, maxnprimaux])
        for i in range(auxbasis.bfs_nao):
            for j in range(auxbasis.bfs_nprim[i]):
                aux_bfs_coeffs[i,j] = auxbasis.bfs_coeffs[i][j]
                aux_bfs_expnts[i,j] = auxbasis.bfs_expnts[i][j]
                aux_bfs_prim_norms[i,j] = auxbasis.bfs_prim_norms[i][j]
            


        
        slice = [0,basis.bfs_nao,0,basis.bfs_nao,0,auxbasis.bfs_nao]
            
        #Limits for the calculation of overlap integrals
        a = int(slice[0]) #row start index
        b = int(slice[1]) #row end index
        c = int(slice[2]) #column start index
        d = int(slice[3]) #column end index
        e = int(slice[4])
        f = int(slice[5])

        ints3c2e = threeCenterTwoElecSymmNumba2(bfs_coords[0], bfs_contr_prim_norms[0], bfs_lmn[0], bfs_nprim[0], bfs_coeffs, bfs_prim_norms, bfs_expnts,aux_bfs_coords[0], aux_bfs_contr_prim_norms[0], aux_bfs_lmn[0], aux_bfs_nprim[0], aux_bfs_coeffs, aux_bfs_prim_norms, aux_bfs_expnts,a,b,c,d,e,f)
        return ints3c2e

    def multipoleMomentSymmNumbawrap(basis, auxbasis):
        # The multipole moments are calculated using the Obara-saika recurrence formula here: http://www.esqc.org/lectures/WK4.pdf
        # Furhtermore, the above resource can be useful in implementing analytical derivatives, as well as Obara-Saika based 4c2e, 3c2e or 2c2e
        # This is robust Numba implementation
        # Here the lists are converted to numpy arrays for better use with Numba.
        # Once these conversions are done we pass these to a Numba decorated
        # function that uses prange, etc. to calculate the 4c2e integrals efficiently.
        # This function calculates the electron-electron Coulomb potential matrix for a given basis object and mol object.
        # The basis object holds the information of basis functions like: exponents, coeffs, etc.
        
        # The integrals are performed using the formulas https://pubs.acs.org/doi/full/10.1021/acs.jchemed.8b00255
        # Some useful resources:
        # https://chemistry.stackexchange.com/questions/71527/how-does-one-compute-the-number-of-unique-2-electron-integrals-for-a-given-basis
        # https://chemistry.stackexchange.com/questions/82532/practical-differences-between-storing-2-electron-integrals-and-calculating-them
        # Density matrix based screening Section (5.1.2): https://www.diva-portal.org/smash/get/diva2:740301/FULLTEXT02.pdf
        # Lots of notes on screening and efficiencies (3.3 pg 31): https://www.zora.uzh.ch/id/eprint/44716/1/Diss.123456.pdf
        # In the above resource in section 3.4 it also says that storing in 64 bit floating point precision is usually not required. 
        # So, the storage requirements can be reduced by 4 to 8 times.
        

        # This function calculates the 4c2e integrals for a given basis object.
        # The basis object holds the information of basis functions like: exponents, coeffs, etc.
        # It is possible to only calculate a slice (block) of the complete set of integrals.
        # slice is an 8 element list whose first and second elements give the range of the A functions to be calculated.
        # and so on.

        #We convert the required properties to numpy arrays as this is what Numba likes.
        bfs_coords = np.array([basis.bfs_coords])
        bfs_contr_prim_norms = np.array([basis.bfs_contr_prim_norms])
        bfs_lmn = np.array([basis.bfs_lmn])
        bfs_nprim = np.array([basis.bfs_nprim])

        #We convert the required properties to numpy arrays as this is what Numba likes.
        aux_bfs_coords = np.array([auxbasis.bfs_coords])
        aux_bfs_contr_prim_norms = np.array([auxbasis.bfs_contr_prim_norms])
        aux_bfs_lmn = np.array([auxbasis.bfs_lmn])
        aux_bfs_nprim = np.array([auxbasis.bfs_nprim])
           

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

        maxnprimaux = max(auxbasis.bfs_nprim)
        aux_bfs_coeffs = np.zeros([auxbasis.bfs_nao, maxnprimaux])
        aux_bfs_expnts = np.zeros([auxbasis.bfs_nao, maxnprimaux])
        aux_bfs_prim_norms = np.zeros([auxbasis.bfs_nao, maxnprimaux])
        for i in range(auxbasis.bfs_nao):
            for j in range(auxbasis.bfs_nprim[i]):
                aux_bfs_coeffs[i,j] = auxbasis.bfs_coeffs[i][j]
                aux_bfs_expnts[i,j] = auxbasis.bfs_expnts[i][j]
                aux_bfs_prim_norms[i,j] = auxbasis.bfs_prim_norms[i][j]
            


        
        slice = [0,basis.bfs_nao,0,basis.bfs_nao,0,auxbasis.bfs_nao]
            
        #Limits for the calculation of overlap integrals
        a = int(slice[0]) #row start index
        b = int(slice[1]) #row end index
        c = int(slice[2]) #column start index
        d = int(slice[3]) #column end index
        e = int(slice[4])
        f = int(slice[5])

        ints3c2e = threeCenterTwoElecSymmNumba2(bfs_coords[0], bfs_contr_prim_norms[0], bfs_lmn[0], bfs_nprim[0], bfs_coeffs, bfs_prim_norms, bfs_expnts,aux_bfs_coords[0], aux_bfs_contr_prim_norms[0], aux_bfs_lmn[0], aux_bfs_nprim[0], aux_bfs_coeffs, aux_bfs_prim_norms, aux_bfs_expnts,a,b,c,d,e,f)
        return ints3c2e

    def RysfourCenterTwoElecSymmNumbawrap(basis):
        # This is robust Numba implementation
        # Here the lists are converted to numpy arrays for better use with Numba.
        # Once these conversions are done we pass these to a Numba decorated
        # function that uses prange, etc. to calculate the 4c2e integrals efficiently.
        # This function calculates the electron-electron Coulomb potential matrix for a given basis object and mol object.
        # The basis object holds the information of basis functions like: exponents, coeffs, etc.
        
        # The integrals are performed using the formulas https://pubs.acs.org/doi/full/10.1021/acs.jchemed.8b00255
        # Some useful resources:
        # https://chemistry.stackexchange.com/questions/71527/how-does-one-compute-the-number-of-unique-2-electron-integrals-for-a-given-basis
        # https://chemistry.stackexchange.com/questions/82532/practical-differences-between-storing-2-electron-integrals-and-calculating-them
        # Density matrix based screening Section (5.1.2): https://www.diva-portal.org/smash/get/diva2:740301/FULLTEXT02.pdf
        # Lots of notes on screening and efficiencies (3.3 pg 31): https://www.zora.uzh.ch/id/eprint/44716/1/Diss.123456.pdf
        # In the above resource in section 3.4 it also says that storing in 64 bit floating point precision is usually not required. 
        # So, the storage requirements can be reduced by 4 to 8 times.
        

        # This function calculates the 4c2e integrals for a given basis object.
        # The basis object holds the information of basis functions like: exponents, coeffs, etc.
        # It is possible to only calculate a slice (block) of the complete set of integrals.
        # slice is an 8 element list whose first and second elements give the range of the A functions to be calculated.
        # and so on.

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
            


        
        slice1 = [0,basis.bfs_nao,0,basis.bfs_nao,0,basis.bfs_nao,0,basis.bfs_nao]
            
        #Limits for the calculation of 4c2e integrals
        indx_startA = int(slice1[0])
        indx_endA = int(slice1[1])
        indx_startB = int(slice1[2])
        indx_endB = int(slice1[3])
        indx_startC = int(slice1[4])
        indx_endC = int(slice1[5])
        indx_startD = int(slice1[6])
        indx_endD = int(slice1[7])

        ints4c2e = rys4c2eSymmNumba2(bfs_coords[0], bfs_contr_prim_norms[0], bfs_lmn[0], bfs_nprim[0], bfs_coeffs, bfs_prim_norms, bfs_expnts,indx_startA,indx_endA,indx_startB,indx_endB,indx_startC,indx_endC,indx_startD,indx_endD)

        return ints4c2e

    

    def fourCenterTwoElecFastNumbawrap(basis, slice1=None):
        # This implementation is based on quadruplets, i.e., instead of running loops on (ij|kl), we can simply generate a list of indices(quadruplets)
        # by using np.stack. Why is this efficient? Because this allows more efficient parallelization as the prange of Numba is only being used once over
        # a relatively larger list rather than 4 different pranges over smaller lists.
        # This is robust Numba implementation
        # Here the lists are converted to numpy arrays for better use with Numba.
        # Once these conversions are done we pass these to a Numba decorated
        # function that uses prange, etc. to calculate the 4c2e integrals efficiently.
        # This function calculates the electron-electron Coulomb potential matrix for a given basis object and mol object.
        # The basis object holds the information of basis functions like: exponents, coeffs, etc.
        
        # It is possible to only calculate a slice (block) of the complete matrix.
        # slice is a 4 element list whose first and second elements give the range of the rows to be calculated.
        # the third and fourth element give the range of columns to be calculated.
        # The integrals are performed using the formulas https://pubs.acs.org/doi/full/10.1021/acs.jchemed.8b00255
        # Some useful resources:
        # https://chemistry.stackexchange.com/questions/71527/how-does-one-compute-the-number-of-unique-2-electron-integrals-for-a-given-basis
        # https://chemistry.stackexchange.com/questions/82532/practical-differences-between-storing-2-electron-integrals-and-calculating-them
        # Density matrix based screening Section (5.1.2): https://www.diva-portal.org/smash/get/diva2:740301/FULLTEXT02.pdf
        # Lots of notes on screening and efficiencies (3.3 pg 31): https://www.zora.uzh.ch/id/eprint/44716/1/Diss.123456.pdf
        # In the above resource in section 3.4 it also says that storing in 64 bit floating point precision is usually not required. 
        # So, the storage requirements can be reduced by 4 to 8 times.
        

        # This function calculates the 4c2e integrals for a given basis object.
        # The basis object holds the information of basis functions like: exponents, coeffs, etc.
        # It is possible to only calculate a slice (block) of the complete set of integrals.
        # slice is an 8 element list whose first and second elements give the range of the A functions to be calculated.
        # and so on.

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
            


        #If the user doesn't provide a slice then calculate the complete set of 4c2e integrals for all the BFs
        if slice1==None:
            slice1 = [0,basis.bfs_nao,0,basis.bfs_nao,0,basis.bfs_nao,0,basis.bfs_nao]

        
        #Get a list of quadruplets
        #The quadruplets list is just a combinations of the i,j,k,l indices for the BFs 
        durationQuadruplets = 0.0
        start = timer()
        quadruplets = []
        # Method 1 to generate the quadruplets list (Slow af)
        # import itertools
        # for i,j,k,l in itertools.product(range(slice1[0],slice1[1]), range(slice1[2],slice1[3]), range(slice1[4],slice1[5]), range(slice1[6],slice1[7])):
        #     quadruplets.append([i,j,k,l])
        # Method 2 to generate the quadruplets (Again slow af)
        # quadruplets = [[i,j,k,l] for i,j,k,l in itertools.product(range(slice1[0],slice1[1]), range(slice1[2],slice1[3]), range(slice1[4],slice1[5]), range(slice1[6],slice1[7]))]
        # Method 3 Extremely efficient way to generate the list with numpy instead of itertools
        quadruplets = np.stack(np.meshgrid(range(slice1[0],slice1[1]), range(slice1[2],slice1[3]), range(slice1[4],slice1[5]), range(slice1[6],slice1[7])),-1).reshape(-1,4)
        # quadruplets = np.array(quadruplets, dtype=int)
        durationQuadruplets = timer() - start
        # print(quadruplets)
        # print(durationQuadruplets)
        
        
            
        #Limits for the calculation of 4c2e integrals
        indx_startA = int(slice1[0])
        indx_endA = int(slice1[1])
        indx_startB = int(slice1[2])
        indx_endB = int(slice1[3])
        indx_startC = int(slice1[4])
        indx_endC = int(slice1[5])
        indx_startD = int(slice1[6])
        indx_endD = int(slice1[7])



        ints4c2e = fourCenterTwoElecFastNumba2(quadruplets, bfs_coords[0], bfs_contr_prim_norms[0], bfs_lmn[0], bfs_nprim[0], bfs_coeffs, bfs_prim_norms, bfs_expnts,indx_startA,indx_endA,indx_startB,indx_endB,indx_startC,indx_endC,indx_startD,indx_endD)

        return ints4c2e
        # returns (AB|CD) 


    def fourCenterTwoElecFastSymmNumbawrap(basis):
        # This one uses symmetrically unique quadruplets
        # This implementation is based on quadruplets, i.e., instead of running loops on (ij|kl), we can simply generate a list of indices(quadruplets)
        # by using np.stack. Why is this efficient? Because this allows more efficient parallelization as the prange of Numba is only being used once over
        # a relatively larger list rather than 4 different pranges over smaller lists.
        # This is robust Numba implementation
        # Here the lists are converted to numpy arrays for better use with Numba.
        # Once these conversions are done we pass these to a Numba decorated
        # function that uses prange, etc. to calculate the 4c2e integrals efficiently.
        # This function calculates the electron-electron Coulomb potential matrix for a given basis object and mol object.
        # The basis object holds the information of basis functions like: exponents, coeffs, etc.
        
        # It is possible to only calculate a slice (block) of the complete matrix.
        # slice is a 4 element list whose first and second elements give the range of the rows to be calculated.
        # the third and fourth element give the range of columns to be calculated.
        # The integrals are performed using the formulas https://pubs.acs.org/doi/full/10.1021/acs.jchemed.8b00255
        # Some useful resources:
        # https://chemistry.stackexchange.com/questions/71527/how-does-one-compute-the-number-of-unique-2-electron-integrals-for-a-given-basis
        # https://chemistry.stackexchange.com/questions/82532/practical-differences-between-storing-2-electron-integrals-and-calculating-them
        # Density matrix based screening Section (5.1.2): https://www.diva-portal.org/smash/get/diva2:740301/FULLTEXT02.pdf
        # Lots of notes on screening and efficiencies (3.3 pg 31): https://www.zora.uzh.ch/id/eprint/44716/1/Diss.123456.pdf
        # In the above resource in section 3.4 it also says that storing in 64 bit floating point precision is usually not required. 
        # So, the storage requirements can be reduced by 4 to 8 times.
        

        # This function calculates the 4c2e integrals for a given basis object.
        # The basis object holds the information of basis functions like: exponents, coeffs, etc.
        # It is possible to only calculate a slice (block) of the complete set of integrals.
        # slice is an 8 element list whose first and second elements give the range of the A functions to be calculated.
        # and so on.

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
            


        slice1 = [0,basis.bfs_nao,0,basis.bfs_nao,0,basis.bfs_nao,0,basis.bfs_nao]
        

        
        #Get a list of quadruplets
        #The quadruplets list is just a combinations of the i,j,k,l indices for the BFs 
        # durationQuadruplets = 0.0
        # start = timer()
        quadruplets = []
        n=basis.bfs_nao
        size=int(n*(n+1)*(n*n+n+2)/8)
        # print(size)
        quadruplets = np.zeros((size,4),dtype=np.int)
        print('Quadruplets size in MB: ',quadruplets.nbytes/1e6, flush=True)
        quadruplets = genQuadrupletsNumba2(basis.bfs_nao,quadruplets)
        # durationQuadruplets=timer()-start
        # print('Duration quads: ', durationQuadruplets)
        
        
            
        #Limits for the calculation of 4c2e integrals
        indx_startA = int(slice1[0])
        indx_endA = int(slice1[1])
        indx_startB = int(slice1[2])
        indx_endB = int(slice1[3])
        indx_startC = int(slice1[4])
        indx_endC = int(slice1[5])
        indx_startD = int(slice1[6])
        indx_endD = int(slice1[7])



        ints4c2e = fourCenterTwoElecFastSymmNumba2(quadruplets, bfs_coords[0], bfs_contr_prim_norms[0], bfs_lmn[0], bfs_nprim[0], bfs_coeffs, bfs_prim_norms, bfs_expnts,indx_startA,indx_endA,indx_startB,indx_endB,indx_startC,indx_endC,indx_startD,indx_endD)

        return ints4c2e
        # returns (AB|CD) 

    
    def overlapMat(basis, slice=None):
        # This function calculates the overlap matrix for a given basis object.
        # The basis object holds the information of basis functions like: exponents, coeffs, etc.
        # It is possible to only calculate a slice (block) of the complete matrix.
        # slice is a 4 element list whose first and second elements give the range of the rows to be calculated.
        # the third and fourth element give the range of columns to be calculated.
        # The integrals are performed using the formulas

        #If the user doesn't provide a slice then calculate the complete overlap matrix for all the BFs
        if slice==None:
            slice = [0,basis.bfs_nao,0,basis.bfs_nao]
        #S = np.zeros([basis.bfs_nao,basis.bfs_nao])
        S = np.zeros([slice[1]-slice[0],slice[3]-slice[2]])
        #Loop pver BFs
        for i in range(slice[0],slice[1]):
            for j in range(slice[2],slice[3]):
                S[i,j]=0
                I = basis.bfs_coords[i]
                J = basis.bfs_coords[j]
                IJ = I - J  
                Ni = basis.bfs_contr_prim_norms[i]
                Nj = basis.bfs_contr_prim_norms[j]
                lmni = basis.bfs_lmn[i]
                lmnj = basis.bfs_lmn[j]
                #Loop over primitives
                for ik in range(basis.bfs_nprim[i]):
                    for jk in range(basis.bfs_nprim[j]):
                        dik = basis.bfs_coeffs[i][ik]
                        djk = basis.bfs_coeffs[j][jk] 
                        Nik = basis.bfs_prim_norms[i][ik]
                        Njk = basis.bfs_prim_norms[j][jk]
                        alphaik = basis.bfs_expnts[i][ik]
                        alphajk = basis.bfs_expnts[j][jk]
                        gamma = alphaik + alphajk
                        P = (alphaik*I + alphajk*J)/gamma
                        PI = P - I
                        PJ = P - J
                        Sx = Integrals.calcS(lmni[0],lmnj[0],gamma,PI[0],PJ[0])
                        Sy = Integrals.calcS(lmni[1],lmnj[1],gamma,PI[1],PJ[1])
                        Sz = Integrals.calcS(lmni[2],lmnj[2],gamma,PI[2],PJ[2])
                        temp = dik*djk
                        temp = temp*Nik*Njk
                        temp = temp*Ni*Nj
                        temp = temp*np.exp(-alphaik*alphajk/gamma*sum(IJ**2))*Sx*Sy*Sz
                        S[i,j] = S[i,j] + temp
        return S

    def overlapMatOS(basis, slice=None):
        # This function calculates the overlap matrix for a given basis object using the OBARA-SAIKA recurrence relations.
        # The basis object holds the information of basis functions like: exponents, coeffs, etc.
        # It is possible to only calculate a slice (block) of the complete matrix.
        # slice is a 4 element list whose first and second elements give the range of the rows to be calculated.
        # the third and fourth element give the range of columns to be calculated.
        # The integrals are performed using the formulas

        #sys.setrecursionlimit(1500)

        #If the user doesn't provide a slice then calculate the complete overlap matrix for all the BFs
        if slice==None:
            slice = [0,basis.bfs_nao,0,basis.bfs_nao]
        #S = np.zeros([basis.bfs_nao,basis.bfs_nao])
        S = np.zeros([slice[1]-slice[0],slice[3]-slice[2]])
        #Loop pver BFs
        for i in range(slice[0],slice[1]):
            for j in range(slice[2],slice[3]):
                S[i,j]=0
                I = basis.bfs_coords[i]
                J = basis.bfs_coords[j]
                IJ = I - J  
                Ni = basis.bfs_contr_prim_norms[i]
                Nj = basis.bfs_contr_prim_norms[j]
                lmni = basis.bfs_lmn[i] #i in OS notation
                lmnj = basis.bfs_lmn[j] #j in OS notation
                #Loop over primitives
                for ik in range(basis.bfs_nprim[i]):
                    for jk in range(basis.bfs_nprim[j]):
                        alphaik = basis.bfs_expnts[i][ik]
                        alphajk = basis.bfs_expnts[j][jk]
                        gamma = alphaik + alphajk #p in OS notation
                        Kab = np.exp(-alphaik*alphajk/gamma*(IJ**2))
                        #if Kab<1.0e-12:   
                        #    continue
                        dik = basis.bfs_coeffs[i][ik]
                        djk = basis.bfs_coeffs[j][jk] 
                        Nik = basis.bfs_prim_norms[i][ik]
                        Njk = basis.bfs_prim_norms[j][jk]
                        
                        
                        P = (alphaik*I + alphajk*J)/gamma
                        PI = P - I
                        PJ = P - J
                        
                        # Sx = Integrals.overlapPrimitivesOS(lmni[0],lmnj[0],gamma,PI[0],PJ[0],Kab[0])
                        # Sy = Integrals.overlapPrimitivesOS(lmni[1],lmnj[1],gamma,PI[1],PJ[1],Kab[1])
                        # Sz = Integrals.overlapPrimitivesOS(lmni[2],lmnj[2],gamma,PI[2],PJ[2],Kab[2])

                        Sx = Integrals.overlapPrimitivesOS_2(lmni[0],lmnj[0],gamma,PI[0],PJ[0],Kab[0])
                        Sy = Integrals.overlapPrimitivesOS_2(lmni[1],lmnj[1],gamma,PI[1],PJ[1],Kab[1])
                        Sz = Integrals.overlapPrimitivesOS_2(lmni[2],lmnj[2],gamma,PI[2],PJ[2],Kab[2])

                        temp = dik*djk
                        temp = temp*Nik*Njk
                        temp = temp*Ni*Nj
                        temp = temp*Sx*Sy*Sz
                        S[i,j] = S[i,j] + temp
        return S

    @functools.lru_cache(10)
    def overlapPrimitivesOS(i, j, p, PA, PB, Kab):
        #Calculates and returns the overlap integral of two Gaussian primitives using OBARA-SAIKA scheme.
        #This function uses the recursion technique, which was found to be slow when compiled with NUMBA.
        #This could have been used in the overlapMat function, but wasn't.
        #But instead it would probably be used to evaluate the kinMat (KE mat).

        if i==0 and j==0:
            return Kab*np.sqrt(np.pi/p)
        elif i==1 and j==0:
            return PA*Integrals.overlapPrimitivesOS(0,0,p,PA,PB,Kab)
        elif j==0:
            return PA*Integrals.overlapPrimitivesOS(i-1,0,p,PA,PB,Kab) + 1/2/p*((i-1)*Integrals.overlapPrimitivesOS(i-2,0,p,PA,PB,Kab))
        elif i==0 and j==1:
            return PB*Integrals.overlapPrimitivesOS(0,0,p,PA,PB,Kab)
        elif i==0:
            return PB*Integrals.overlapPrimitivesOS(0,j-1,p,PA,PB,Kab) + 1/2/p*((j-1)*Integrals.overlapPrimitivesOS(0,j-2,p,PA,PB,Kab))
        elif i==1 and j==1: 
            return PA*Integrals.overlapPrimitivesOS(0,1,p,PA,PB,Kab) + 1/2/p*(Integrals.overlapPrimitivesOS(0,0,p,PA,PB,Kab))
        elif i==1:
            return PB*Integrals.overlapPrimitivesOS(1,j-1,p,PA,PB,Kab) + 1/2/p*(Integrals.overlapPrimitivesOS(0,j-1,p,PA,PB,Kab) + (j-1)*Integrals.overlapPrimitivesOS(1,j-2,p,PA,PB,Kab))
        elif j==1:
            return PA*Integrals.overlapPrimitivesOS(i-1,1,p,PA,PB,Kab) + 1/2/p*((i-1)*Integrals.overlapPrimitivesOS(i-2,1,p,PA,PB,Kab) + Integrals.overlapPrimitivesOS(i-1,0,p,PA,PB,Kab))
        else:
            return PA*Integrals.overlapPrimitivesOS(i-1,j,p,PA,PB,Kab) + 1/2/p*((i-1)*Integrals.overlapPrimitivesOS(i-2,j,p,PA,PB,Kab) + j*Integrals.overlapPrimitivesOS(i-1,j-1,p,PA,PB,Kab))

    def overlapPrimitivesOS_2(i, j, p, PA, PB, Kab):
        #Calculates and returns the overlap integral of two Gaussian primitives using OBARA-SAIKA scheme.
        #This function uses loops instead fo recursion of function.
        #This could have been used in the overlapMat function, but wasn't.
        #But instead it would probably be used to evaluate the kinMat (KE mat).

        S = np.empty((i+3,j+3))
        S[0,:] = 0.0
        S[:,0] = 0.0
        S[1,1] = Kab*np.sqrt(np.pi/p)
        for ii in range(1,i+2):
            for jj in range(1,j+2):
                # S[ii+2,jj+1] = PA*S[ii+1,jj+1] + 0.5/p*((ii+1)*S[ii,jj+1] + (jj+1)*S[ii+1,jj])
                # S[ii+1,jj+2] = PB*S[ii+1,jj+1] + 0.5/p*((ii+1)*S[ii,jj+1] + (jj+1)*S[ii+1,jj])
                S[ii+1,jj] = PA*S[ii,jj] + 0.5/p*((ii-1)*S[ii-1,jj] + (jj-1)*S[ii,jj-1])
                S[ii,jj+1] = PB*S[ii,jj] + 0.5/p*((ii-1)*S[ii-1,jj] + (jj-1)*S[ii,jj-1])

        
        Sij = S[i+1,j+1]
        return Sij
        
    

    
    
    def overlapMatNumba(basis, slice=None):
        #This is a semi-numba implementation.
        #Basically, the function is the same as before (pure python) except
        #that calcS and c2k are Numba versions

        # This function calculates the overlap matrix for a given basis object.
        # The basis object holds the information of basis functions like: exponents, coeffs, etc.
        # It is possible to only calculate a slice (block) of the complete matrix.
        # slice is a 4 element list whose first and second elements give the range of the rows to be calculated.
        # the third and fourth element give the range of columns to be calculated.
        # The integrals are performed using the formulas

        #If the user doesn't provide a slice then calculate the complete overlap matrix fora ll the BFs
        if slice==None:
            slice = [0,basis.bfs_nao,0,basis.bfs_nao]
        #S = np.zeros([basis.bfs_nao,basis.bfs_nao])
        S = np.zeros([slice[1]-slice[0],slice[3]-slice[2]])
        for i in range(slice[0],slice[1]):
            for j in range(slice[2],slice[3]):
                S[i,j]=0
                I = basis.bfs_coords[i]
                J = basis.bfs_coords[j]
                IJ = I - J  
                Ni = basis.bfs_contr_prim_norms[i]
                Nj = basis.bfs_contr_prim_norms[j]
                lmni = basis.bfs_lmn[i]
                lmnj = basis.bfs_lmn[j]
                for ik in range(basis.bfs_nprim[i]):
                    for jk in range(basis.bfs_nprim[j]):
                        dik = basis.bfs_coeffs[i][ik]
                        djk = basis.bfs_coeffs[j][jk] 
                        Nik = basis.bfs_prim_norms[i][ik]
                        Njk = basis.bfs_prim_norms[j][jk]
                        alphaik = basis.bfs_expnts[i][ik]
                        alphajk = basis.bfs_expnts[j][jk]
                        gamma = alphaik + alphajk
                        P = (alphaik*I + alphajk*J)/gamma
                        PI = P - I
                        PJ = P - J
                        Sx = calcSNumba(lmni[0],lmnj[0],gamma,PI[0],PJ[0])
                        Sy = calcSNumba(lmni[1],lmnj[1],gamma,PI[1],PJ[1])
                        Sz = calcSNumba(lmni[2],lmnj[2],gamma,PI[2],PJ[2])
                        temp = dik*djk
                        temp = temp*Nik*Njk
                        temp = temp*Ni*Nj
                        temp = temp*np.exp(-alphaik*alphajk/gamma*sum(IJ**2))*Sx*Sy*Sz
                        S[i,j] = S[i,j] + temp
        return S

    def overlapMatSymmNumbawrap(basis):
        #This is robust Numba implementation
        #Here the lists are converted to numpy arrays for better use with Numba.
        #Once these conversions are done we pass these to a Numba decorated
        #function that uses prange, etc. to calculate the matrix efficiently.

        # This function calculates the overlap matrix for a given basis object.
        # The basis object holds the information of basis functions like: exponents, coeffs, etc.
        # It is possible to only calculate a slice (block) of the complete matrix.
        # slice is a 4 element list whose first and second elements give the range of the rows to be calculated.
        # the third and fourth element give the range of columns to be calculated.
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
            
  
        slice1 = [0,basis.bfs_nao,0,basis.bfs_nao]
            
        #Limits for the calculation of overlap integrals
        a = int(slice1[0])
        b = int(slice1[1])
        c = int(slice1[2])
        d = int(slice1[3])
            
        #start=timer()
        S = overlapMatSymmNumba2(bfs_coords[0], bfs_contr_prim_norms[0], bfs_lmn[0], bfs_nprim[0], bfs_coeffs, bfs_prim_norms, bfs_expnts,a,b,c,d)
        #duration = timer() - start
        #print('Duration Inside: ',duration)
        return S

    def overlapMatNumbawrap(basis, slice1=None):
        #This is robust Numba implementation
        #Here the lists are converted to numpy arrays for better use with Numba.
        #Once these conversions are done we pass these to a Numba decorated
        #function that uses prange, etc. to calculate the matrix efficiently.

        # This function calculates the overlap matrix for a given basis object.
        # The basis object holds the information of basis functions like: exponents, coeffs, etc.
        # It is possible to only calculate a slice (block) of the complete matrix.
        # slice is a 4 element list whose first and second elements give the range of the rows to be calculated.
        # the third and fourth element give the range of columns to be calculated.
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
            


        #If the user doesn't provide a slice then calculate the complete overlap matrix for all the BFs
        if slice1==None:
            slice1 = [0,basis.bfs_nao,0,basis.bfs_nao]
            
        #Limits for the calculation of overlap integrals
        a = int(slice1[0])
        b = int(slice1[1])
        c = int(slice1[2])
        d = int(slice1[3])
            
        #start=timer()
        S = overlapMatNumba2(bfs_coords[0], bfs_contr_prim_norms[0], bfs_lmn[0], bfs_nprim[0], bfs_coeffs, bfs_prim_norms, bfs_expnts,a,b,c,d)
        #duration = timer() - start
        #print('Duration Inside: ',duration)
        return S

    def overlapMatSymmCwrap(basis):
        _liboverlap = ctypes.CDLL('/Users/admin/Python/FDE/crysxdft/overlap.so')
        _liboverlap.overlapMat.argtypes = (ctypes.c_int, ctypes.c_int,
        np.ctypeslib.ndpointer(dtype=np.float64,
            ndim=2,
            flags='C_CONTIGUOUS'
        ),
        np.ctypeslib.ndpointer(dtype=np.float64,
            ndim=1,
            flags='C_CONTIGUOUS'
        ),
        np.ctypeslib.ndpointer(dtype=np.float64,
            ndim=2,
            flags='C_CONTIGUOUS'
        ),
        np.ctypeslib.ndpointer(dtype=np.float64,
            ndim=1,
            flags='C_CONTIGUOUS'
        ),
        np.ctypeslib.ndpointer(dtype=np.float64,
            ndim=2,
            flags='C_CONTIGUOUS'
        ),
        np.ctypeslib.ndpointer(dtype=np.float64,
            ndim=2,
            flags='C_CONTIGUOUS'
        ),
        np.ctypeslib.ndpointer(dtype=np.float64,
            ndim=2,
            flags='C_CONTIGUOUS'
        ),
        np.ctypeslib.ndpointer(dtype=np.float64,
            ndim=2,
            flags='C_CONTIGUOUS'
        )
        )
        _liboverlap.overlapMat.restype = None
        start=timer()
        #This is robust Numba implementation
        #Here the lists are converted to numpy arrays for better use with Numba.
        #Once these conversions are done we pass these to a Numba decorated
        #function that uses prange, etc. to calculate the matrix efficiently.

        # This function calculates the overlap matrix for a given basis object.
        # The basis object holds the information of basis functions like: exponents, coeffs, etc.
        # It is possible to only calculate a slice (block) of the complete matrix.
        # slice is a 4 element list whose first and second elements give the range of the rows to be calculated.
        # the third and fourth element give the range of columns to be calculated.
        # The integrals are performed using the formulas

        #We convert the required properties to numpy arrays as this is what Numba likes.
        bfs_coords = np.array([basis.bfs_coords])
        # print(bfs_coords)
        # exit()
        bfs_contr_prim_norms = np.array([basis.bfs_contr_prim_norms])
        bfs_lmn = np.array([basis.bfs_lmn], dtype=np.float64)
        bfs_nprim = np.array([basis.bfs_nprim], dtype=np.float64)
           

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
            


        slice1 = [0,basis.bfs_nao,0,basis.bfs_nao]
            
        #Limits for the calculation of overlap integrals
        a = int(slice1[0])
        b = int(slice1[1])
        c = int(slice1[2])
        d = int(slice1[3])
            
        S = np.zeros((basis.bfs_nao, basis.bfs_nao), dtype= np.float64)
        _liboverlap.overlapMat(basis.bfs_nao, maxnprim, bfs_coords[0], bfs_contr_prim_norms[0], bfs_lmn[0], bfs_nprim[0], bfs_coeffs, bfs_prim_norms, bfs_expnts, S)
        #S = overlapMatNumba2(bfs_coords[0], bfs_contr_prim_norms[0], bfs_lmn[0], bfs_nprim[0], bfs_coeffs, bfs_prim_norms, bfs_expnts,a,b,c,d)
        duration = timer() - start
        print('Duration using C (serial) for overlap matrix: ',duration, flush=True)
        return S 

    def overlapMatOSNumbawrap(basis, slice1=None):
        #This is robust Numba implementation
        #Here the lists are converted to numpy arrays for better use with Numba.
        #Once these conversions are done we pass these to a Numba decorated
        #function that uses prange, etc. to calculate the matrix efficiently.

        # This function calculates the overlap matrix for a given basis object.
        # The basis object holds the information of basis functions like: exponents, coeffs, etc.
        # It is possible to only calculate a slice (block) of the complete matrix.
        # slice is a 4 element list whose first and second elements give the range of the rows to be calculated.
        # the third and fourth element give the range of columns to be calculated.
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
            


        #If the user doesn't provide a slice then calculate the complete overlap matrix for all the BFs
        if slice1==None:
            slice1 = [0,basis.bfs_nao,0,basis.bfs_nao]
            
        #Limits for the calculation of overlap integrals
        a = int(slice1[0])
        b = int(slice1[1])
        c = int(slice1[2])
        d = int(slice1[3])
            
        #start=timer()
        S = overlapMatOSNumba2(bfs_coords[0], bfs_contr_prim_norms[0], bfs_lmn[0], bfs_nprim[0], bfs_coeffs, bfs_prim_norms, bfs_expnts,a,b,c,d)
        #duration = timer() - start
        #print('Duration Inside: ',duration)
        return S 
    
    def overlapMatNumerical(basis, coords, weights, slice=None):
        #Calculates the overlap matrix numerically on a given grid.
        # coords - cooridnates of grid points (m x 3) array
        # weights - weights of grid points (m) array
        #If the user doesn't provide a slice then calculate the complete overlap matrix fora ll the BFs
        if slice==None:
            slice = [0,basis.bfs_nao,0,basis.bfs_nao]
        #S = np.zeros([basis.bfs_nao,basis.bfs_nao])
        S = np.zeros([slice[1]-slice[0],slice[3]-slice[2]])

        blocksize = 50000
        ngrids = coords.shape[0]
        nblocks = ngrids//blocksize
        for iblock in range(nblocks+1):
            offset = iblock*blocksize
            #rho_block = rho[offset : min(offset+blocksize,ngrids)]
            #ao_value_block = ao_value[offset : min(offset+blocksize,ngrids)]
            weights_block = weights[offset : min(offset+blocksize,ngrids)]
            coords_block = coords[offset : min(offset+blocksize,ngrids)] 

            #BF values at grid points (m x nbf)
            # bf_values = Integrals.evalBFs(basis, coords)
            bf_values = Integrals.evalBFsNumbawrap(basis, coords_block)
            #Scale them with weights
            bf_values = bf_values.T*np.sqrt(weights_block)
            bf_values = bf_values.T
            # S = np.einsum('mi,mj->ij',bf_values,bf_values)
            S += contract('mi,mj->ij',bf_values,bf_values)

        return S

    def nucMatNumerical(basis, coords, weights, mol, slice=None):
        #Calculates the overlap matrix numerically on a given grid.
        # coords - cooridnates of grid points (m x 3) array
        # weights - weights of grid points (m) array
        #If the user doesn't provide a slice then calculate the complete overlap matrix fora ll the BFs
        if slice==None:
            slice = [0,basis.bfs_nao,0,basis.bfs_nao]
        #S = np.zeros([basis.bfs_nao,basis.bfs_nao])
        Vnuc = np.zeros([slice[1]-slice[0],slice[3]-slice[2]])

        blocksize = 50000
        ngrids = coords.shape[0]
        nblocks = ngrids//blocksize
        for iblock in range(nblocks+1):
            offset = iblock*blocksize
            #rho_block = rho[offset : min(offset+blocksize,ngrids)]
            #ao_value_block = ao_value[offset : min(offset+blocksize,ngrids)]
            weights_block = weights[offset : min(offset+blocksize,ngrids)]
            coords_block = coords[offset : min(offset+blocksize,ngrids)] 

            #BF values at grid points (m x nbf)
            # bf_values = Integrals.evalBFs(basis, coords)
            bf_values = Integrals.evalBFsNumbawrap(basis, coords_block)
            #Scale them with weights
            bf_values = bf_values.T*np.sqrt(weights_block)
            bf_values = bf_values.T
            # Loop over nuclei
            for iatom in range(mol.natoms):
                Rc = mol.coordsBohrs[iatom]
                Zc = mol.Zcharges[iatom]
                pot =  -Zc/np.linalg.norm(coords_block-Rc,ord=2,axis=1) #https://stackoverflow.com/questions/1401712/how-can-the-euclidean-distance-be-calculated-with-numpy

                Vnuc += contract('mi,m,mj->ij',bf_values,pot,bf_values)

        return Vnuc


    def kinMatSymmNumbawrap(basis):
        #This is robust Numba implementation
        # SYmmetric version of the normal function
        #Here the lists are converted to numpy arrays for better use with Numba.
        #Once these conversions are done we pass these to a Numba decorated
        #function that uses prange, etc. to calculate the matrix efficiently.

        # This function calculates the kinetic matrix for a given basis object.
        # The basis object holds the information of basis functions like: exponents, coeffs, etc.
        # It is possible to only calculate a slice (block) of the complete matrix.
        # slice is a 4 element list whose first and second elements give the range of the rows to be calculated.
        # the third and fourth element give the range of columns to be calculated.
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
            


        slice1 = [0,basis.bfs_nao,0,basis.bfs_nao]
            
        #Limits for the calculation of overlap integrals
        a = int(slice1[0])
        b = int(slice1[1])
        c = int(slice1[2])
        d = int(slice1[3])
            
        #start=timer()
        T = kinMatSymmNumba2(bfs_coords[0], bfs_contr_prim_norms[0], bfs_lmn[0], bfs_nprim[0], bfs_coeffs, bfs_prim_norms, bfs_expnts,a,b,c,d)
        #duration = timer() - start
        #print('Duration Inside: ',duration)
        return T 

    def kinMatNumbawrap(basis, slice1=None):
        #This is robust Numba implementation
        #Here the lists are converted to numpy arrays for better use with Numba.
        #Once these conversions are done we pass these to a Numba decorated
        #function that uses prange, etc. to calculate the matrix efficiently.

        # This function calculates the kinetic matrix for a given basis object.
        # The basis object holds the information of basis functions like: exponents, coeffs, etc.
        # It is possible to only calculate a slice (block) of the complete matrix.
        # slice is a 4 element list whose first and second elements give the range of the rows to be calculated.
        # the third and fourth element give the range of columns to be calculated.
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
            


        #If the user doesn't provide a slice then calculate the complete overlap matrix for all the BFs
        if slice1==None:
            slice1 = [0,basis.bfs_nao,0,basis.bfs_nao]
            
        #Limits for the calculation of overlap integrals
        a = int(slice1[0])
        b = int(slice1[1])
        c = int(slice1[2])
        d = int(slice1[3])
            
        #start=timer()
        T = kinMatNumba2(bfs_coords[0], bfs_contr_prim_norms[0], bfs_lmn[0], bfs_nprim[0], bfs_coeffs, bfs_prim_norms, bfs_expnts,a,b,c,d)
        #duration = timer() - start
        #print('Duration Inside: ',duration)
        return T 

    def nucMatNumbawrap(basis, mol, slice=None):
        #This is robust Numba implementation
        #Here the lists are converted to numpy arrays for better use with Numba.
        #Once these conversions are done we pass these to a Numba decorated
        #function that uses prange, etc. to calculate the matrix efficiently.

        # This function calculates the nuclear matrix for a given basis object.
        # The basis object holds the information of basis functions like: exponents, coeffs, etc.
        # It is possible to only calculate a slice (block) of the complete matrix.
        # slice is a 4 element list whose first and second elements give the range of the rows to be calculated.
        # the third and fourth element give the range of columns to be calculated.
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
            


        #If the user doesn't provide a slice then calculate the complete overlap matrix for all the BFs
        if slice==None:
            slice = [0,basis.bfs_nao,0,basis.bfs_nao]
            
        #Limits for the calculation of overlap integrals
        a = int(slice[0]) #row start index
        b = int(slice[1]) #row end index
        c = int(slice[2]) #column start index
        d = int(slice[3]) #column end index
        
        #start=timer()
        V = nucMatNumba2(bfs_coords[0], bfs_contr_prim_norms[0], bfs_lmn[0], bfs_nprim[0], bfs_coeffs, bfs_prim_norms, bfs_expnts,a,b,c,d, Z[0], coordsBohrs[0], natoms)
        #duration = timer() - start
        #print('Duration Inside: ',duration)
        return V

    def nucMatSymmNumbawrap(basis, mol):
        #Symmetric variant
        #This is robust Numba implementation
        #Here the lists are converted to numpy arrays for better use with Numba.
        #Once these conversions are done we pass these to a Numba decorated
        #function that uses prange, etc. to calculate the matrix efficiently.

        # This function calculates the nuclear matrix for a given basis object.
        # The basis object holds the information of basis functions like: exponents, coeffs, etc.
        # It is possible to only calculate a slice (block) of the complete matrix.
        # slice is a 4 element list whose first and second elements give the range of the rows to be calculated.
        # the third and fourth element give the range of columns to be calculated.
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
            


        
        slice = [0,basis.bfs_nao,0,basis.bfs_nao]
            
        #Limits for the calculation of overlap integrals
        a = int(slice[0]) #row start index
        b = int(slice[1]) #row end index
        c = int(slice[2]) #column start index
        d = int(slice[3]) #column end index
        
        #start=timer()
        V = nucMatSymmNumba2(bfs_coords[0], bfs_contr_prim_norms[0], bfs_lmn[0], bfs_nprim[0], bfs_coeffs, bfs_prim_norms, bfs_expnts,a,b,c,d, Z[0], coordsBohrs[0], natoms)
        #duration = timer() - start
        #print('Duration Inside: ',duration)
        return V 

    def overlapPrimitives(alphaA, coordA, lmnA, alphaB, coordB, lmnB):
        #Calculates and returns the overlap integral of two Gaussian primitives.
        #This could have been used in the overlapMat function, but wasn't.
        #But instead it would probably be used to evaluate the kinMat (KE mat).
        
        AB = coordA - coordB
        gamma = alphaA + alphaB
        P = (alphaA*coordA + alphaB*coordB)/gamma
        PA = P - coordA
        PB = P - coordB
        Sx = Integrals.calcS(lmnA[0],lmnB[0],gamma,PA[0],PB[0])
        Sy = Integrals.calcS(lmnA[1],lmnB[1],gamma,PA[1],PB[1])
        Sz = Integrals.calcS(lmnA[2],lmnB[2],gamma,PA[2],PB[2])
        out = np.exp(-alphaA*alphaB/gamma*sum(AB**2))*Sx*Sy*Sz

        return out

    def kinMat(basis, slice=None):
        # This function calculates the kinetic matrix for a given basis object.
        # The basis object holds the information of basis functions like: exponents, coeffs, etc.
        # It is possible to only calculate a slice (block) of the complete matrix.
        # slice is a 4 element list whose first and second elements give the range of the rows to be calculated.
        # the third and fourth element give the range of columns to be calculated.
        # The integrals are performed using the formulas here https://pubs.acs.org/doi/full/10.1021/acs.jchemed.8b00255

        #If the user doesn't provide a slice then calculate the complete kinetic matrix for all the BFs
        if slice==None:
            slice = [0,basis.bfs_nao,0,basis.bfs_nao]
        T = np.zeros([slice[1]-slice[0],slice[3]-slice[2]])
        for i in range(slice[0],slice[1]):
            for j in range(slice[2],slice[3]):
                T[i,j]=0
                I = basis.bfs_coords[i]
                J = basis.bfs_coords[j]
                IJ = I - J  
                Ni = basis.bfs_contr_prim_norms[i]
                Nj = basis.bfs_contr_prim_norms[j]
                lmni = basis.bfs_lmn[i]
                lmnj = basis.bfs_lmn[j]
                for ik in range(basis.bfs_nprim[i]):
                    for jk in range(basis.bfs_nprim[j]):
                        dik = basis.bfs_coeffs[i][ik]
                        djk = basis.bfs_coeffs[j][jk] 
                        Nik = basis.bfs_prim_norms[i][ik]
                        Njk = basis.bfs_prim_norms[j][jk]
                        alphaik = basis.bfs_expnts[i][ik]
                        alphajk = basis.bfs_expnts[j][jk]
                        temp = dik*djk  #coeff of primitives as read from basis set
                        temp = temp*Nik*Njk #normalization factors of primitives
                        temp = temp*Ni*Nj #normalization factor of the contraction of primitives

                        overlap1 = Integrals.overlapPrimitives(alphaik, I, lmni, alphajk, J, lmnj)
                        overlap2 = Integrals.overlapPrimitives(alphaik, I, lmni, alphajk, J, [lmnj[0]+2,lmnj[1],lmnj[2]])
                        overlap3 = Integrals.overlapPrimitives(alphaik, I, lmni, alphajk, J, [lmnj[0],lmnj[1]+2,lmnj[2]])
                        overlap4 = Integrals.overlapPrimitives(alphaik, I, lmni, alphajk, J, [lmnj[0],lmnj[1],lmnj[2]+2])
                        overlap5 = Integrals.overlapPrimitives(alphaik, I, lmni, alphajk, J, [lmnj[0]-2,lmnj[1],lmnj[2]])
                        overlap6 = Integrals.overlapPrimitives(alphaik, I, lmni, alphajk, J, [lmnj[0],lmnj[1]-2,lmnj[2]])
                        overlap7 = Integrals.overlapPrimitives(alphaik, I, lmni, alphajk, J, [lmnj[0],lmnj[1],lmnj[2]-2])

                        part1 = overlap1*alphajk*(2*(lmnj[0]+lmnj[1]+lmnj[2])+3)
                        part2 = 2*alphajk**2*(overlap2+overlap3+overlap4)
                        part3 = (lmnj[0]*(lmnj[0]-1))*overlap5
                        part4 = (lmnj[1]*(lmnj[1]-1))*overlap6
                        part5 = (lmnj[2]*(lmnj[2]-1))*overlap7

                        result = temp*(part1 - part2 - 0.5*(part3+part4+part5))
                        
                        T[i,j] = T[i,j] + result
        return T

    def evalGTO(alpha, coeff, coordCenter, lmn, coord):
        #This function evaluates the value of a given Gaussian primitive 
        # with given values of alpha (exponent), coefficient, and angular momentum
        # centered at 'coordCenter' (3 comp, numpy array). The value is calculated at
        # a given 'coord'.

        # alpha = float(alpha)
        # coeff = float(coeff)
        # coordCenter = np.float_(coordCenter)
        # lmn = np.float_(lmn)
        # coord = np.float_(coord)
        
        value = coeff*((coord[0]-coordCenter[0])**lmn[0]*(coord[1]-coordCenter[1])**lmn[1]*(coord[2]-coordCenter[2])**lmn[2])*np.exp(-alpha*((coord[0]-coordCenter[0])**2+(coord[1]-coordCenter[1])**2+(coord[2]-coordCenter[2])**2))

        return value

    def evalGTOautograd(coordx, coordy, coordz, alpha, coeff, coordCenterx, coordCentery, coordCenterz, lmn):
        #This function evaluates the value of a given Gaussian primitive 
        # with given values of alpha (exponent), coefficient, and angular momentum
        # centered at 'coordCenter' (3 comp, numpy array). The value is calculated at
        # a given 'coord'.

        # Although, this function uses autograd, it would be better to use explicit expressions for derivatives.
        # As these can then be used with NUMBA.
        # The following are the derivative expressions evaluated using Wolfram alpha
        # First derivative wrt x:
        # https://www.wolframalpha.com/input/?i=d%28x%5El+y%5Em+z%5En+exp%28-alpha+%28%28x-a%29%5E2+%2B+%28y-b%29%5E2+%2B+%28z-c%29%5E2%29%29%29%2Fdx
        # Second derivative wrt x:
        # https://www.wolframalpha.com/input/?i=d2%28x%5El+y%5Em+z%5En+exp%28-alpha+%28%28x-a%29%5E2+%2B+%28y-b%29%5E2+%2B+%28z-c%29%5E2%29%29%29%2Fdx2
        
        value = coeff*((coordx-coordCenterx)**lmn[0]*(coordy-coordCentery)**lmn[1]*(coordz-coordCenterz)**lmn[2])*npautograd.exp(-alpha*((coordx-coordCenterx)**2+(coordy-coordCentery)**2+(coordz-coordCenterz)**2))

        return value

    def evalBFi(basis, i, coord):
        #This function evaluates the value of a given Basis function at a given grid point (coord)
        #'coord' is a 3 element 1d-array
        value = 0.0
        coordi = basis.bfs_coords[i]
        Ni = basis.bfs_contr_prim_norms[i]
        lmni = basis.bfs_lmn[i]
        # print(lmni)
        #Loop over primitives
        for ik in range(basis.bfs_nprim[i]):
            dik = basis.bfs_coeffs[i][ik] 
            Nik = basis.bfs_prim_norms[i][ik]
            alphaik = basis.bfs_expnts[i][ik]
            value = value + Integrals.evalGTO(alphaik, Ni*Nik*dik, coordi, lmni, coord)
            # print(Ni*Nik*dik)
            # print(Integrals.evalGTO(alphaik, Ni*Nik*dik, coordi, lmni, coord))
            

        return value

    

    def evalBFs(basis, coord):
        #This function evaluates the value of all the given Basis functions on the grid (coord).
        # 'coord' should be a nx3 array
        
        nao = basis.bfs_nao
        ncoord = coord.shape[0]
        result = np.zeros((ncoord, nao))
        #Loop over grid points
        for k in range(ncoord):
            #Loop over BFs
            for i in range(nao):
                value = 0.0
                coordi = basis.bfs_coords[i]
                Ni = basis.bfs_contr_prim_norms[i]
                lmni = basis.bfs_lmn[i]
                for ik in range(basis.bfs_nprim[i]):
                    dik = basis.bfs_coeffs[i][ik] 
                    Nik = basis.bfs_prim_norms[i][ik]
                    alphaik = basis.bfs_expnts[i][ik]
                    value = value + Integrals.evalGTO(alphaik, Ni*Nik*dik, coordi, lmni, coord[k])
                result[k,i] = value
                #result[k,i] = Integrals.evalBFi(basis,i,coord[k])


        return result

    def evalBFsNumbawrap(basis, coord, parallel=True, non_zero_indices=None):

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
        bfs_radius_cutoff = np.zeros([basis.bfs_nao])
        for i in range(basis.bfs_nao):
            for j in range(basis.bfs_nprim[i]):
                bfs_coeffs[i,j] = basis.bfs_coeffs[i][j]
                bfs_expnts[i,j] = basis.bfs_expnts[i][j]
                bfs_prim_norms[i,j] = basis.bfs_prim_norms[i][j]
                bfs_radius_cutoff[i] = basis.bfs_radius_cutoff[i]

        if parallel:
            if non_zero_indices is not None:
                bf_values = evalBFsSparseNumba2(bfs_coords[0], bfs_contr_prim_norms[0], bfs_nprim[0], bfs_lmn[0], bfs_coeffs, bfs_prim_norms, bfs_expnts, bfs_radius_cutoff, coord, non_zero_indices)
            else:
                bf_values = evalBFsNumba2(bfs_coords[0], bfs_contr_prim_norms[0], bfs_nprim[0], bfs_lmn[0], bfs_coeffs, bfs_prim_norms, bfs_expnts, bfs_radius_cutoff, coord)
        else:
            if non_zero_indices is not None:
                bf_values = evalBFsSparse_serialNumba2(bfs_coords[0], bfs_contr_prim_norms[0], bfs_nprim[0], bfs_lmn[0], bfs_coeffs, bfs_prim_norms, bfs_expnts, bfs_radius_cutoff, coord, non_zero_indices)
            else:
                bf_values = evalBFs_serialNumba2(bfs_coords[0], bfs_contr_prim_norms[0], bfs_nprim[0], bfs_lmn[0], bfs_coeffs, bfs_prim_norms, bfs_expnts, bfs_radius_cutoff, coord)
        return bf_values

    def evalBFsgradNumbawrap(basis, coord, deriv=1):

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

        bf_grad_values = evalBFsgradNumba2(bfs_coords[0], bfs_contr_prim_norms[0], bfs_nprim[0], bfs_lmn[0], bfs_coeffs, bfs_prim_norms, bfs_expnts, coord)

        return bf_grad_values
    
    def evalBFsandgradNumbawrap(basis, coord, deriv=1):

        #We convert the required properties to numpy arrays as this is what Numba likes.
        bfs_coords = np.array([basis.bfs_coords])
        bfs_contr_prim_norms = np.array([basis.bfs_contr_prim_norms])
        bfs_lmn = np.array([basis.bfs_lmn])
        bfs_nprim = np.array([basis.bfs_nprim])

           

        #The remaining properties like bfs_coeffs are a list of lists of unequal sizes.
        #Numba won't be able to work with these efficiently.
        #So, we convert them to a numpy 2d array by applying a trick,
        #that the second dimension is that of the largest list. So that
        #it can accomodate all the lists.
        maxnprim = max(basis.bfs_nprim)
        bfs_coeffs = np.zeros([basis.bfs_nao, maxnprim])
        bfs_expnts = np.zeros([basis.bfs_nao, maxnprim])
        bfs_prim_norms = np.zeros([basis.bfs_nao, maxnprim])
        bfs_radius_cutoff = np.zeros([basis.bfs_nao])
        for i in range(basis.bfs_nao):
            for j in range(basis.bfs_nprim[i]):
                bfs_coeffs[i,j] = basis.bfs_coeffs[i][j]
                bfs_expnts[i,j] = basis.bfs_expnts[i][j]
                bfs_prim_norms[i,j] = basis.bfs_prim_norms[i][j]
                bfs_radius_cutoff[i] = basis.bfs_radius_cutoff[i]

        # bf_grad_values = evalBFsandgradNumba2(bfs_coords[0], bfs_contr_prim_norms[0], bfs_nprim[0], bfs_lmn[0], bfs_coeffs, bfs_prim_norms, bfs_expnts, coord)
        bf_values, bf_grad_values = evalBFsandgradNumba2(bfs_coords[0], bfs_contr_prim_norms[0], bfs_nprim[0], bfs_lmn[0], bfs_coeffs, bfs_prim_norms, bfs_expnts, bfs_radius_cutoff, coord)

        # return bf_grad_values
        return bf_values, bf_grad_values

    


    def evalRhoi(basis, densmat, coord):
        # Evaluates the value of density at a given point ('coord')
        # For this we need the density matrix as well as the information 
        # the basis functions in the basis object.
        # rho at the grid point m is given as:
        # \rho^m = \sum_{\mu}\sum_{\nu} D_{\mu \nu} \mu^m \nu^m
        rho = 0.0
        #Loop over BFs
        for i in range(basis.bfs_nao):
            for j in range(basis.bfs_nao):
                mu = evalBFi(basis, i, coord)
                nu = evalBFi(basis, j, coord)
                rho = rho + densmat[i,j]*mu*nu

        return rho 

    def evalRho(basis, densmat, coords):
        # Evaluates the value of density at a given grid ('coord') nx3 array
        # For this we need the density matrix as well as the information 
        # the basis functions in the basis object.
        # rho at the grid point m is given as:
        # \rho^m = \sum_{\mu}\sum_{\nu} D_{\mu \nu} \mu^m \nu^m
        rho = np.zeros((coords.shape[0]))
        bf_values = Integrals.evalBFs(basis, coords)
        #Loop over grid points
        # for m in range(coords.shape[0]):
        #     #Loop over BFs
        #     for i in range(basis.bfs_nao):
        #         for j in range(basis.bfs_nao):
        #             mu = bf_values[m,i]
        #             nu = bf_values[m,j]
        #             rho[m] = rho[m] + densmat[i,j]*mu*nu

        # Instead of the above loops, the performance can be greatly (drastically) 4-5 times improved with einsum
        # But still extreeemely slower (at least more than 100 times) than the Numba version
        # This goes on to show the power of NUMBA especially when loops are concerned.
        rho = np.einsum('ij,mi,mj->m',densmat,bf_values,bf_values)
        #TODO What if we save some things that can be reused in SCF iterations.
        #For example the bf_values should be calculated only once at the start of SCF.
        #Similarly one should look for any other such optimizaitons that can be made.

        return rho 

    def evalRhoNumbawrap(basis, densmat, coords):
        # Evaluates the value of density at a given grid ('coord') nx3 array
        # For this we need the density matrix as well as the information 
        # the basis functions in the basis object.
        # rho at the grid point m is given as:
        # \rho^m = \sum_{\mu}\sum_{\nu} D_{\mu \nu} \mu^m \nu^m
        bf_values = Integrals.evalBFsNumbawrap(basis, coords)
        
        rho = evalRhoNumba2(bf_values, densmat)

        return rho 

    def funcgrad(func, partial, deriv=1):
        #func: the function whose derivative is required
        #deriv: the order of derivative
        for i in range(deriv):
            func = grad(func, partial)
        return func

    def evalBFigrad(basis, i, coord, deriv=1):
        #This function evaluates the value of the derivative of a given Basis function at a given grid point (coord)
        #'coord' is a 3 element 1d-array
        value = 0.0
        coordi = basis.bfs_coords[i]
        Ni = basis.bfs_contr_prim_norms[i]
        lmni = basis.bfs_lmn[i]
        # print(lmni)
        #Loop over primitives
        for ik in range(basis.bfs_nprim[i]):
            dik = basis.bfs_coeffs[i][ik] 
            Nik = basis.bfs_prim_norms[i][ik]
            alphaik = basis.bfs_expnts[i][ik]
            gradGTO = Integrals.funcgrad(Integrals.evalGTO, deriv)
            value = value + gradGTO(alphaik, Ni*Nik*dik, coordi, lmni, coord)
            # print(Ni*Nik*dik)
            # print(Integrals.evalGTO(alphaik, Ni*Nik*dik, coordi, lmni, coord))
            

        return value

    def evalBFsgrad(basis, coord, deriv=1):
        #This function evaluates the value of all the given Basis functions on the grid (coord).
        # 'coord' should be a nx3 array
        
        nao = basis.bfs_nao
        ncoord = coord.shape[0]
        if deriv==1:
            result = np.zeros((3, ncoord, nao))
            #Loop over BFs
            for i in range(nao):
                coordi = basis.bfs_coords[i]
                Ni = basis.bfs_contr_prim_norms[i]
                lmni = basis.bfs_lmn[i]
                for ik in range(basis.bfs_nprim[i]):
                    dik = basis.bfs_coeffs[i][ik] 
                    Nik = basis.bfs_prim_norms[i][ik]
                    alphaik = basis.bfs_expnts[i][ik]
                    # gradGTOx = grad(Integrals.evalGTOautograd, 0)
                    # gradGTOy = grad(Integrals.evalGTOautograd, 1)
                    # gradGTOz = grad(Integrals.evalGTOautograd, 2)
                    gradGTOx = Integrals.funcgrad(Integrals.evalGTOautograd, 0, deriv=1)
                    gradGTOy = Integrals.funcgrad(Integrals.evalGTOautograd, 1, deriv=1)
                    gradGTOz = Integrals.funcgrad(Integrals.evalGTOautograd, 2, deriv=1)
                    for k in range(ncoord):
                        value = gradGTOx(*coord[k], alphaik, Ni*Nik*dik, *coordi, lmni)
                        result[0, k,i] = result[0, k,i] + value
                        value = gradGTOy(*coord[k], alphaik, Ni*Nik*dik, *coordi, lmni)
                        result[1, k,i] = result[1, k,i] + value
                        value = gradGTOz(*coord[k], alphaik, Ni*Nik*dik, *coordi, lmni)
                        result[2, k,i] = result[2, k,i] + value

        return result



    def eval_dens_func2(basis, dmat, weights, coords, funcid=[1,7], moCoeff=None, moOcc=None, spin=0, rho_grid_old=0, blocksize=50000, debug=False, list_nonzero_indices=None, count_nonzero_indices=None):
        # This uses blocks
        # Stable + Memory Efficient and approx 3 times slower than eval_dens_func3
        # In order to evaluate a density functional we will use the 
        # libxc library with Python bindings.
        # However, some sort of simple functionals like LDA, GGA, etc would
        # need to be implemented in CrysX also, so that the it doesn't depend
        # on external libraries so heavily that it becomes unusable without those.

        #This function uses the divide and conquer approach to calculate the required stuff.

        #Useful links:
        # LibXC manual: https://www.tddft.org/programs/libxc/manual/
        # LibXC gitlab: https://gitlab.com/libxc/libxc/-/tree/master
        # LibXC python interface code: https://gitlab.com/libxc/libxc/-/blob/master/pylibxc/functional.py
        # LibXC python version installation and example: https://www.tddft.org/programs/libxc/installation/
        # Formulae for XC energy and potential calculation: https://pubs.acs.org/doi/full/10.1021/ct200412r
        # LibXC code list: https://github.com/pyscf/pyscf/blob/master/pyscf/dft/libxc.py
        # PySCF nr_rks code: https://github.com/pyscf/pyscf/blob/master/pyscf/dft/numint.py
        # https://www.osti.gov/pages/servlets/purl/1650078



        #OUTPUT
        #Functional energy
        efunc = 0.0
        #Functional potential V_{\mu \nu} = \mu|\partial{f}/\partial{\rho}|\nu
        v = np.zeros((basis.bfs_nao, basis.bfs_nao))
        #TODO it only works for LDA functionals for now.
        #Need to make it work for GGA, Hybrid, range-separated Hybrid and MetaGGA functionals as well.
        
        #Calculate number of blocks/batches
        ngrids = coords.shape[0]
        nblocks = ngrids//blocksize
        nelec = 0.0

        if list_nonzero_indices is not None:
            dmat_orig = dmat

        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax = Axes3D(fig)

        ### Calculate stuff necessary for bf/ao evaluation on grid points
        ### Doesn't make any difference for 510 bfs but might be significant for >1000 bfs
        # This will help to make the call to evalbfnumba1 faster by skipping the mediator evalbfnumbawrap function 
        # that prepares the following stuff at every iteration
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
        bfs_radius_cutoff = np.zeros([basis.bfs_nao])
        for i in range(basis.bfs_nao):
            for j in range(basis.bfs_nprim[i]):
                bfs_coeffs[i,j] = basis.bfs_coeffs[i][j]
                bfs_expnts[i,j] = basis.bfs_expnts[i][j]
                bfs_prim_norms[i,j] = basis.bfs_prim_norms[i][j]
                bfs_radius_cutoff[i] = basis.bfs_radius_cutoff[i]
        # Now bf/ao values can be evaluated by calling the following
        # bf_values = evalBFsNumba2(bfs_coords[0], bfs_contr_prim_norms[0], bfs_nprim[0], bfs_lmn[0], bfs_coeffs, bfs_prim_norms, bfs_expnts, bfs_radius_cutoff, coord)



        # OPT EINSUM STUFF TO GENERATE AN EXPRESSION AND PATH ONCE AND USE IT AGAIN
        # eq = 'ij,mi,mj->m'
        # # mark the first arrays as constant
        # constant = [0]
        # # supplied ops are now mix of shapes and arrays
        # ops = dmat, (ngrids, basis.bfs_nao), (ngrids, basis.bfs_nao)
        # expr_rho = contract_expression(eq, *ops, constants=constant)


        # Sparse stuff
        # dmat[np.abs(dmat) < 1.0E-10] = 0
        # dmat_sp = sparse.COO(dmat)

        # eq = 'mi,ij,mj->m'
        # # mark the first arrays as constant
        # constant = [0]
        # # supplied ops are now mix of shapes and arrays
        # ops = dmat, (ngrids, basis.bfs_nao), (ngrids, basis.bfs_nao)
        # expr_rho = contract_expression(eq, *ops, constants=constant)

        # print('Number of blocks: ', nblocks)

        durationLibxc = 0.0
        durationE = 0.0
        durationF = 0.0
        durationZ = 0.0
        durationV = 0.0
        durationRho = 0.0
        durationAO = 0.0

        xc_family_dict = {1:'LDA',2:'GGA',4:'MGGA'} 

        # Create a LibXC object  
        funcx = pylibxc.LibXCFunctional(funcid[0], "unpolarized")
        funcc = pylibxc.LibXCFunctional(funcid[1], "unpolarized")
        x_family_code = funcx.get_family()
        c_family_code = funcc.get_family()


        # print('\n\n------------------------------------------------------')
        # print('Exchange-Correlation Functional')
        # print('------------------------------------------------------\n')
        # print('XC Functional IDs supplied: ', funcid)
        # print('\n\nDescription of exchange functional: \n')
        # print('The Exchange function belongs to the family:', xc_family_dict[x_family_code])
        # print(funcx.describe())
        # print('\n\nDescription of correlation functional: \n')
        # print(' The Correlation function belongs to the family:', xc_family_dict[c_family_code])
        # print(funcc.describe())
        # print('------------------------------------------------------\n')
        # print('\n\n')

        
        # def get_ordered_list(points, x, y, z):
        #     points.sort(key = lambda p: (p[0] - x)**2 + (p[1] - y)**2 + (p[2] - z)**2)
        #     # print(points[0:10])
        #     return points

        # coords_weights = np.c_[coords, weights]

        # coords_weights = np.array(get_ordered_list(coords_weights.tolist(), min(coords[:,0]), min(coords[:,1]), min(coords[:,2])))
        # print(min(coords[:,0]))
        # print(min(coords[:,1]))
        # print(min(coords[:,2]))

        # print(max(coords[:,0]))
        # print(max(coords[:,1]))
        # print(max(coords[:,2]))

        # exit()
        # print(coords[:,0])
        # ax.scatter(coords[:,0], coords[:,1], 0.0)
        # plt.show()
        for iblock in range(nblocks+1):
            offset = iblock*blocksize

            weights_block = weights[offset : min(offset+blocksize,ngrids)]
            coords_block = coords[offset : min(offset+blocksize,ngrids)] 

            if list_nonzero_indices is not None:
                non_zero_indices = list_nonzero_indices[iblock][0:count_nonzero_indices[iblock]]
                # Get the subset of density matrix corresponding to the siginificant basis functions
                dmat = dmat_orig[np.ix_(non_zero_indices, non_zero_indices)]

            # weights_block = coords_weights[offset : min(offset+blocksize,ngrids),3]
            # coords_block = coords_weights[offset : min(offset+blocksize,ngrids),0:3] 
            # fig = plt.figure()
            # ax = fig.add_subplot(111, projection='3d')
            # ax = Axes3D(fig)
            # ax.scatter(coords_block[:,0], coords_block[:,1], coords_block[:,2])
            # ax.set_xlabel('x')
            # ax.set_ylabel('y')
            # ax.set_zlabel('z')
            # ax.scatter(coords[:,0], coords[:,1], coords[:,2])
            # plt.show()


            # print(coords_block.shape)
            # print(np.max(coords_block[:,0]))
            # print(np.max(coords_block[:,1]))
            # print(np.max(coords_block[:,2]))
            # print(np.min(coords_block[:,0]))
            # print(np.min(coords_block[:,1]))
            # print(np.min(coords_block[:,2]))

            # exit()

            # Evaluating the BF values on grid points, takes up a lot of memory which 
            # which leads to problems for large systems(not in terms of nbfs but in terms of natoms)
            # Because large number of atoms or spatial coverage leads to a very large grid.
            # So we can try to see how is the performance if everything is evaluated for grid blocks.
            # UPDATE: Evaluating AO values and everything else as well inside the 
            # blocks loop offered significant!!! improvement in speed as well as memory used was
            # extremely low for large systems as well. 
            # So, this should be final. Everything is calculated fairly fast.
            # The V calculation takes the longest.
            # Benchmark for Cholesterol.xyz with def2-TZVPP basis (PySCF 253 sec, CrysX 578 sec)
            # Benchmark for Cholesterol.xyz with def2-TZVPP basis (PySCF 1215 sec, CrysX 2673 sec out of which V took 2500 sec)
            # Duration for AO values:  37.74370314400039
            # Duration for Rho at grid points:  19.064996759999076
            # Duration for LibXC computaitons at grid points:  0.06629791900036253
            # Duration for calculation of total density functional energy:  0.12889835499981928
            # Duration for calculation of F:  0.0027878710004642926
            # Duration for calculation of z :  10.410915866001005
            # Duration for V:  511.2656810230003
            # Seems like the calculaiotn of AO values and V could use some further optimization.
            # Although, it is unclear if it it is possible or not.
            if debug:
                startAO = timer()
            if NUMBA_EXISTS:
                if xc_family_dict[x_family_code]=='LDA' and xc_family_dict[c_family_code]=='LDA':
                      # ao_value_block = Integrals.evalBFsNumbawrap(basis, coords_block)       
                      if list_nonzero_indices is not None:
                          ao_value_block = evalBFsSparseNumba2(bfs_coords[0], bfs_contr_prim_norms[0], bfs_nprim[0], bfs_lmn[0], bfs_coeffs, bfs_prim_norms, bfs_expnts, bfs_radius_cutoff, coords_block, non_zero_indices)
                      else:
                          ao_value_block = evalBFsNumba2(bfs_coords[0], bfs_contr_prim_norms[0], bfs_nprim[0], bfs_lmn[0], bfs_coeffs, bfs_prim_norms, bfs_expnts, bfs_radius_cutoff, coords_block)
                      # nonzero_indices, count = nonZeroBFIndicesNumba2(coords_block, ao_value_block, 1e-10)
                      # if debug:
                      #     print('Number of non-zero bfs in this batch: ', count)
                      # ao_value_block, rho_block = evalBFsandRhoNumba2(bfs_coords[0], bfs_contr_prim_norms[0], bfs_nprim[0], bfs_lmn[0], bfs_coeffs, bfs_prim_norms, bfs_expnts, bfs_radius_cutoff, coords_block, dmat)

                # If either x or c functional is of GGA/MGGA type we need ao_grad_values
                if xc_family_dict[x_family_code]!='LDA' or xc_family_dict[c_family_code]!='LDA':
                    # ao_value_block = Integrals.evalBFsNumbawrap(basis, coords_block)
                    # ao_values_grad_block = Integrals.evalBFsgradNumbawrap(basis, coords_block, deriv=1)

                    # Calculating ao values and gradients together, didn't really do much improvement in computational speed
                    ao_value_block, ao_values_grad_block = Integrals.evalBFsandgradNumbawrap(basis, coords_block, deriv=1)
                    
                    # ao_value_block = ao_values_and_grad_block[0,:,:]
                    # ao_values_grad_block = ao_values_and_grad_block[1:4,:,:]

            else:
                ao_value_block = Integrals.evalBFs(basis, coords_block)
                # If either x or c functional is of GGA/MGGA type we need ao_grad_values
                if xc_family_dict[x_family_code]!='LDA' or xc_family_dict[c_family_code]!='LDA':
                    ao_values_grad_block = Integrals.evalBFsgrad(basis, coords_block, deriv=1)
            if debug:
                durationAO = durationAO + timer() - startAO
                print('Duration for AO values: ', durationAO) 

            #UPDATE: Calculating rho in blocks seems to offer either the same speed
            # or most likely slower than just calculating the Rho altogether.
            #However, this would still be useful in cases where grids are too large to fit in memory.
            #So, it would be preferable that it works at same speed at before and not slower.
            if debug:
                startRho = timer()
            if NUMBA_EXISTS:
                # Although, in intial test the numpy einsum was deemed to be slow,
                # which is why the Numba version was created.
                # But now in real application of DFT, einsum is at least 3-5 times faster.
                # rho_block = np.einsum('ij,mi,mj->m',dmat,ao_value_block,ao_value_block)
                if moOcc is not None:
                    pos = moOcc > 1.0E-12
                    if pos.sum()>0:
                        cpos = contract('ij,j->ij', moCoeff[:,pos], np.sqrt(moOcc[pos]))
                        # rho_block = contract('ij,mi,mj->m',cpos,ao_value_block,ao_value_block[:,pos])
                        tempC = contract('ij,mi->mj',cpos,ao_value_block)
                        rho_block = contract('mi,mi->m', tempC, tempC)

                    else:
                        # if xctype == 'LDA' or xctype == 'HF':
                        rho_block = np.zeros(blocksize)
                    neg = moOcc < -1.0E-12
                    if neg.sum() > 0:
                        cneg = contract('ij,j->ij', moCoeff[:,neg], np.sqrt(-moOcc[neg]))
                        rho_block = contract('ij,mi,mj->m',cneg,ao_value_block,ao_value_block)
                else:
                    # rho_block = np.zeros(blocksize)
                    # rho_block = contract('ij,mi,mj->m',dmat,ao_value_block,ao_value_block)
                    # print(mask.sum)
                    # maskdmat = np.invert(np.isclose(np.abs(dmat),0.0,atol=1E-12) )
                    # print(maskdmat)
                    # print(dmat[maskdmat].shape)
                    # rho_block = evalRhoNumba2(ao_value_block[:,maskdmat[0,:]], dmat[maskdmat], coords_block)
                    # rho_block = contract('ij,mi,mj->m',dmat[maskdmat],ao_value_block[:,maskdmat[0,:]],ao_value_block[:,maskdmat[0,:]])
                    # rho_block = contract('ij,mi,mj->m',dmat,ao_value_block,ao_value_block) # Original (pretty fast)

                    # Sparse stuff (slows down the program)
                    # ao_value_block[np.abs(ao_value_block) < 1.0E-8] = 0
                    # ao_value_block_sp = sparse.COO(ao_value_block)
                    
                    
                    # rho_block = expr_rho(ao_value_block,ao_value_block) # Probably the fastest uptil now
                    if list_nonzero_indices is not None:
                        rho_block = contract('ij,mi,mj->m', dmat, ao_value_block, ao_value_block) # Original (pretty fast)
                    else:
                        rho_block = evalRhoNumba2(ao_value_block, dmat) # This is by-far the fastest now <-----
                    # Some tricks to make the evalrho faster when density matrix is almost converged (Turns out it makes no difference)
                    # rho_block += rho_grid_old[offset : min(offset+blocksize,ngrids)]
                    # rho_grid[offset : min(offset+blocksize,ngrids)] = rho_block

                    # The following two also offer the same speed as the above
                    # tempo = dmat @ ao_value_block.T
                    # rho_block = contract('mi, im -> m', ao_value_block, tempo)

                    # The following two also offers the same speed as the above examples
                    # tempo = ao_value_block @ dmat
                    # rho_block = contract('mi, mi -> m', ao_value_block, tempo)

                    # Sparse version of the above (Extremely slowww (Probably due to the todense call))
                    # tempo = (ao_value_block_sp @ dmat_sp)
                    # rho_block = contract('mi, mi -> m', ao_value_block, tempo.todense())

                    # tempo = dmat @ ao_value_block.T
                    # rho_block = np.tensordot(ao_value_block, tempo, [1,0])

                    # rho_block = np.tensordot(tempo, ao_value_block.T, axes=1)

                    # rho_block = contract('mi,mi->m', ao_value_block, ao_value_block @ dmat) # As fast as using the expressions and constant of opt_einsum
                    # rho_block = np.diagonal(ao_value_block @ dmat @ ao_value_block.T) # Extremely slowwwwwwwwww!
                    
                # If either x or c functional is of GGA/MGGA type we need rho_grad_values
                if xc_family_dict[x_family_code]!='LDA' or xc_family_dict[c_family_code]!='LDA':
                    rho_grad_block_x = contract('ij,mi,mj->m',dmat,ao_values_grad_block[0],ao_value_block)+\
                                            contract('ij,mi,mj->m',dmat,ao_value_block,ao_values_grad_block[0])
                    rho_grad_block_y = contract('ij,mi,mj->m',dmat,ao_values_grad_block[1],ao_value_block)+\
                                            contract('ij,mi,mj->m',dmat,ao_value_block,ao_values_grad_block[1])
                    rho_grad_block_z = contract('ij,mi,mj->m',dmat,ao_values_grad_block[2],ao_value_block)+\
                                            contract('ij,mi,mj->m',dmat,ao_value_block,ao_values_grad_block[2])
                    sigma_block = np.zeros((3,weights_block.shape[0]))
                    sigma_block[1] = rho_grad_block_x**2 + rho_grad_block_y**2 + rho_grad_block_z**2
            else:
                rho_block = contract('ij,mi,mj->m',dmat,ao_value_block,ao_value_block)
                # rho_block = np.einsum('ij,mi,mj->m',dmat,ao_value_block,ao_value_block)
                # rho_block = Integrals.evalRho(basis, dmat, coords_block)
            #rho_block = Integrals.evalRho(basis, dmat, coords_block)
            if debug:
                durationRho = durationRho + timer() - startRho
                print('Duration for Rho at grid points: ',durationRho)


            #TODO
            #Although. the XC energy and matrix calculation is now much more optimized than before,
            #one more optimization that can be made is to perform the following calculations in blocks.
            #i.e divide the rho, weights and ao_values in block os smaller arrays.
            #This will reduce the burden on memory as well as offer speed improvements.
        
            #LibXC stuff
            # Exchange
            # startLibxc = timer()
            # Input dictionary for libxc
            inp = {}
            # Input dictionary needs density values at grid points
            inp['rho'] = rho_block
            if xc_family_dict[x_family_code]!='LDA':
                # Input dictionary needs sigma (\nabla \rho \cdot \nabla \rho) values at grid points
                inp['sigma'] = sigma_block[1]
            # Calculate the necessary quantities using LibXC
            retx = funcx.compute(inp)
            # durationLibxc = durationLibxc + timer() - startLibxc
            # print('Duration for LibXC computations at grid points: ',durationLibxc)

            # Correlation
            # startLibxc = timer()
            # Input dictionary for libxc
            inp = {}
            # Input dictionary needs density values at grid points
            inp['rho'] = rho_block
            if xc_family_dict[c_family_code]!='LDA':
                # Input dictionary needs sigma (\nabla \rho \cdot \nabla \rho) values at grid points
                inp['sigma'] = sigma_block[1]
            # Calculate the necessary quantities using LibXC
            retc = funcc.compute(inp)
            # durationLibxc = durationLibxc + timer() - startLibxc
            # print('Duration for LibXC computations at grid points: ',durationLibxc)

            # startE = timer()
            #ENERGY-----------
            e = retx['zk'] + retc['zk'] # Functional values at grid points
            # Testing CrysX's own implmentation
            #e = densfuncs.lda_x(rho)

            # Calculate the total energy 
            # Multiply the density at grid points by weights
            den = rho_block*weights_block #elementwise multiply
            efunc = efunc + np.dot(den, e) #Multiply with functional values at grid points and sum
            nelec = nelec + np.sum(den)
            # durationE = durationE + timer() - startE
            # print('Duration for calculation of total density functional energy: ',durationE)

            #POTENTIAL----------
            # The derivative of functional wrt density is vrho
            vrho = retx['vrho'] + retc['vrho']
            vsigma = 0
            # If either x or c functional is of GGA/MGGA type we need rho_grad_values
            if xc_family_dict[x_family_code]!='LDA':
                # The derivative of functional wrt grad \rho square.
                vsigma = retx['vsigma']
            if xc_family_dict[c_family_code]!='LDA':
                # The derivative of functional wrt grad \rho square.
                vsigma += retc['vsigma']
            

            # startF = timer()
            # F = np.multiply(weights_block,vrho[:,0]) #This is fast enough.
            v_rho_temp = vrho[:,0]
            F = numexpr.evaluate('(weights_block*v_rho_temp)')
            # If either x or c functional is of GGA/MGGA type we need rho_grad_values
            if xc_family_dict[x_family_code]!='LDA' or xc_family_dict[c_family_code]!='LDA':
                Ftemp = 2*weights_block*vsigma[:,0]
                Fx = Ftemp*rho_grad_block_x
                Fy = Ftemp*rho_grad_block_y
                Fz = Ftemp*rho_grad_block_z
                # Ftemp = 2*np.multiply(weights_block,vsigma.T)
                # Fx = Ftemp*rho_grad_block_x
                # Fy = Ftemp*rho_grad_block_y
                # Fz = Ftemp*rho_grad_block_z
            # durationF = durationF + timer() - startF
            # print('Duration for calculation of F: ',durationF)
            # startZ = timer()
            #TODO: If possible optimize z calculation. It is the slowest part right now
            #Tried to do this with NUMBA. But that was slower.
            #UPDATE: Using numexpr: Numexpr is faster than einsum.
            #NUMEXPR should be used whenever available makes the calculation significantly faster (more than 5 times for C12H12)
            if NUMEXPR_EXISTS:
                ao_value_block_T = ao_value_block.T
                z = numexpr.evaluate('(0.5*F*ao_value_block_T)')
                # z = 0.5*np.einsum('m,mi->mi',F,ao_value_block)
                # If either x or c functional is of GGA/MGGA type we need rho_grad_values
                if xc_family_dict[x_family_code]!='LDA' or xc_family_dict[c_family_code]!='LDA':
                    z = z + Fx*ao_values_grad_block[0].T + Fy*ao_values_grad_block[1].T + Fz*ao_values_grad_block[2].T
                
                # z = z
            else:
                z = 0.5*np.einsum('m,mi->mi',F,ao_value_block, optimize=True)
            #z = 0.5*(np.multiply(F,ao_value.T)).T #This just made it slower Try NUMBA maybe. UPDATE: NUMBA IS 5-10 times slower for C12H12.
            #z = calcZ2(F,ao_value)
            # durationZ = durationZ + timer() - startZ
            # print('Duration for calculation of z : ',durationZ)
            # Free memory
            F = 0
            ao_value_block_T = 0
            vrho = 0

            if debug:
                startV = timer()
            #TODO: If possible optimize this, very slow right now
            #UPDATE: After improving the calculaiton time of z using numexpr,
            #now this (v) remains the slowest part of this code.
            #Perhaps one could try using gemm from blas directly with the help of scipy.
            
            # v_temp = np.einsum('mi,mj->ij',ao_value_block, z, optimize=True)#.sum(axis=0)
            # v_temp = v_temp.sum(axis=0)
            
            #Dot seems only very slightly faster. Needs more testing. But considering how slow einsum is 
            #dot should be much faster.
            #TODO: An important thing to check in the future is to maybe use an optimized einsum 
            # here or maybe anywhere else in the code. Optimized einsum: https://github.com/dgasmith/opt_einsum
            # TODO: It turns out that numpy is not parallelized on MacOS test machine and neither was pyscf
            # which might explain why the calculation of density using numpy was slower.
            # This also explains how the performance reached a level equivalent to PySCF.
            # So, I need to perform tests with numpy and pyscf parallelised properly.
            # v_temp = np.zeros((basis.bfs_nao,basis.bfs_nao))
            # print(mask2.shape)
            # print(z.shape)
            # print(mask2.ndim)
            # print(ao_value_block.shape)

            # print(mask2.sum())
            # print(mask2)
            # mask3 = np.outer(mask2, mask2)

            # mask3 = np.outer(np.ones(basis.bfs_nao, dtype=bool), mask2)
            # v_temp2 = np.dot(z,ao_value_block[:,mask2]) 
            # np.putmask(v_temp,mask3,v_temp2) 

            # print(v_temp.shape)
            # print(v_temp2.shape)
            # print(mask3.shape)
            # print(mask3.sum())

            # print(v_temp.sum())
            # x=np.dot(z,ao_value_block)
            # print(x.sum())

            # print(np.abs(v_temp-x))
            # exit()
            # print(z.max())
            # print(z.min())
            # print(ao_value_block[:,mask2].max())
            # print(ao_value_block[:,mask2].min())
            # print(z.sum())
            # print(z[mask2,:].sum())
            # print(z.shape)
            # print(z[mask2,:].shape)

            # v_temp2 = np.dot(z[maskdmat[0,:],:],ao_value_block[:,maskdmat[0,:]]) 
            # np.putmask(v_temp,maskdmat,v_temp2) 

            # v_temp = np.dot(z,ao_value_block) 
            # print(z.flags)
            # print(ao_value_block.flags)

            # v_temp = calcVtempNumba2(z, ao_value_block, mask, blocksize, basis.bfs_nao) 
            # print(z.shape)
            # print(ao_value_block.shape)
            # v_temp = calcVtempNumba2(z.copy(order='F'), ao_value_block.copy(order='F')) 

            # v_temp = calcVtempNumba2(z, ao_value_block, basis.bfs_nao) # Almost as fast as the following snippet
            # v += v_temp
            
            # v_temp = z @ ao_value_block  # The fastest uptil now
            # v_temp += v_temp.T 
            # v += v_temp 

            # Numexpr
            v_temp = z @ ao_value_block  # The fastest uptil now 
            # v_temp = np.matmul(z, ao_value_block)
            v_temp_T = v_temp.T
            if list_nonzero_indices is not None:
                v[np.ix_(non_zero_indices, non_zero_indices)] += numexpr.evaluate('(v_temp + v_temp_T)')
            else:
                v = numexpr.evaluate('(v + v_temp + v_temp_T)')

            # Sparse version (Very slow)
            # z[np.abs(z) < 1.0E-10] = 0
            # z_sp = sparse.COO(z)
            # v_temp_sp = z_sp @ ao_value_block_sp
            # v_temp_sp += v_temp_sp.T
            # v += v_temp_sp.todense() 



            # v_temp = numexpr.evaluate('(z@ao_value_block)')
            # v_temp = scipy.linalg.blas.dsyr2k(1.0,z.T,ao_value_block)
            # v_temp = scipy.linalg.blas.dgemm(1.0,z,ao_value_block) 
            # print(z.shape)
            # print(ao_value_block.shape)
            # v_temp = ao_value_block.T
            # v_temp = csc_matrix(ao_value_block.T)@csc_matrix(z)
            # v_temp = matMulNumba2(ao_value_block.T,z.T)
            # v_temp = contract('mi,mj->ij',ao_value_block, z)#.sum(axis=0)
            # v_temp = v_temp.sum(axis=0)
            # v_temp += v_temp.T 
            # v += v_temp
            if debug:
                durationV = durationV + timer() - startV
                print('Duration for V: ',durationV)

            # exit()

        print('Number of electrons: ', nelec)

        return efunc, v#, rho_grid

    # @controller.wrap(limits=1, user_api='blas')
    @threadpool_limits.wrap(limits=1, user_api='blas')
    def block_dens_func(iblock, blocksize, weights_block, coords_block, ngrids, basis, dmat, funcid, bfs_data_as_np_arrays, non_zero_indices=None, ao_values=None, funcx=None, funcc=None, x_family_code=None, c_family_code=None, xc_family_dict=None):
        ### Setting the number of threads explicitly as whoen below doesn't work
        ### Therefore use threadpoolctl https://github.com/numpy/numpy/issues/11826
        # https://github.com/joblib/threadpoolctl
        # https://stackoverflow.com/questions/29559338/set-max-number-of-threads-at-runtime-on-numpy-openblas
        # numexpr.set_num_threads(1)
        # set_num_threads(1) # Numba
        # os.environ['OMP_NUM_THREADS'] = '1'
        # os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=4 
        # os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=6
        # os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=4
        # os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=6
        # os.system("export OMP_NUM_THREADS=1")
        # os.system("export OPENBLAS_NUM_THREADS=1")
        # os.system("export MKL_NUM_THREADS=1")
        # os.system("export NUMEXPR_NUM_THREADS=1")
        durationLibxc = 0.0
        durationE = 0.0
        durationF = 0.0
        durationZ = 0.0
        durationV = 0.0
        durationRho = 0.0
        durationAO = 0.0

        if funcx is None:
            xc_family_dict = {1:'LDA',2:'GGA',4:'MGGA'}
            funcx = pylibxc.LibXCFunctional(funcid[0], "unpolarized")
            funcc = pylibxc.LibXCFunctional(funcid[1], "unpolarized")
            x_family_code = funcx.get_family()
            c_family_code = funcc.get_family()

        
        bfs_coords = bfs_data_as_np_arrays[0]
        bfs_contr_prim_norms = bfs_data_as_np_arrays[1]
        bfs_nprim = bfs_data_as_np_arrays[2]
        bfs_lmn = bfs_data_as_np_arrays[3]
        bfs_coeffs = bfs_data_as_np_arrays[4]
        bfs_prim_norms = bfs_data_as_np_arrays[5]
        bfs_expnts = bfs_data_as_np_arrays[6]
        bfs_radius_cutoff = bfs_data_as_np_arrays[7]
        # startAO = timer()
        # LDA
        if xc_family_dict[x_family_code]=='LDA' and xc_family_dict[c_family_code]=='LDA':
            if ao_values is not None:
                ao_value_block = ao_values
            elif NUMBA_EXISTS:
                # ao_value_block = Integrals.evalBFsNumbawrap(basis, coords_block, parallel=False)
                if non_zero_indices is not None:
                    ao_value_block = evalBFsSparse_serialNumba2(bfs_coords, bfs_contr_prim_norms, bfs_nprim, bfs_lmn, bfs_coeffs, bfs_prim_norms, bfs_expnts, bfs_radius_cutoff, coords_block, non_zero_indices)
                else:
                    ao_value_block = evalBFs_serialNumba2(bfs_coords, bfs_contr_prim_norms, bfs_nprim, bfs_lmn, bfs_coeffs, bfs_prim_norms, bfs_expnts, bfs_radius_cutoff, coords_block)
                # ao_value_block, rho_block = evalBFsandRho_serialNumba2(bfs_coords, bfs_contr_prim_norms, bfs_nprim, bfs_lmn, bfs_coeffs, bfs_prim_norms, bfs_expnts, bfs_radius_cutoff, coords_block, dmat)
            else:
                ao_value_block = Integrals.evalBFs(basis, coords_block)
        # durationAO = durationAO + timer() - startAO
        # print('Duration for AO values: ', durationAO) 

        #UPDATE: Calculating rho in blocks seems to offer either the same speed
        # or most likely slower than just calculating the Rho altogether.
        #However, this would still be useful in cases where grids are too large to fit in memory.
        #So, it would be preferable that it works at same speed at before and not slower.
        # startRho = timer()
        if NUMBA_EXISTS:
            if non_zero_indices is not None:
                rho_block = contract('ij,mi,mj->m', dmat, ao_value_block, ao_value_block) # Original (pretty fast)
            else:
                rho_block = evalRho_serialNumba2(ao_value_block, dmat) # This is by-far the fastest now <-----
        else:
            rho_block = contract('ij,mi,mj->m',dmat,ao_value_block,ao_value_block)
                # rho_block = Integrals.evalRho(basis, dmat, coords_block)
            #rho_block = Integrals.evalRho(basis, dmat, coords_block)
        # durationRho = durationRho + timer() - startRho
        # print('Duration for Rho at grid points: ',durationRho)

        

        #TODO
        #Although. the XC energy and matrix calculation is now much more optimized than before,
        #one more optimization that can be made is to perform the following calculations in blocks.
            #i.e divide the rho, weights and ao_values in block os smaller arrays.
            #This will reduce the burden on memory as well as offer speed improvements.
        
            #LibXC stuff
            # Exchange
        # startLibxc = timer()
        
            # Input dictionary for libxc
        inp = {}
            # Input dictionary needs density values at grid points
        inp['rho'] = rho_block
            # Calculate the necessary quantities using LibXC
        retx = funcx.compute(inp)
        # durationLibxc = durationLibxc + timer() - startLibxc
            # print('Duration for LibXC computations at grid points: ',durationLibxc)

        # Correlation
        # startLibxc = timer()
        
        # Input dictionary for libxc
        inp = {}
        # Input dictionary needs density values at grid points
        inp['rho'] = rho_block
        # Calculate the necessary quantities using LibXC
        retc = funcc.compute(inp)
        # durationLibxc = durationLibxc + timer() - startLibxc
        # print('Duration for LibXC computations at grid points: ',durationLibxc)

        # startE = timer()
        #ENERGY-----------
        e = retx['zk'] + retc['zk'] # Functional values at grid points
        # Testing CrysX's own implmentation
        #e = densfuncs.lda_x(rho)

        # Calculate the total energy 
        # Multiply the density at grid points by weights
        den = rho_block*weights_block #elementwise multiply
        efunc = np.dot(den, e) #Multiply with functional values at grid points and sum
        nelec = np.sum(den)
        # durationE = durationE + timer() - startE
        # print('Duration for calculation of total density functional energy: ',durationE)

        #POTENTIAL----------
        # The derivative of functional wrt density is vrho
        vrho = retx['vrho'] + retc['vrho']
        retx = 0
        retc = 0
        func = 0
        startF = timer()
        v_rho_temp = vrho[:,0]
        F = numexpr.evaluate('(weights_block*v_rho_temp)')
        # durationF = durationF + timer() - startF
        # print('Duration for calculation of F: ',durationF)
        # startZ = timer()
        #TODO: If possible optimize z calculation. It is the slowest part right now
        #Tried to do this with NUMBA. But that was slower.
        #UPDATE: Using numexpr: Numexpr is faster than einsum.
        #NUMEXPR should be used whenever available makes the calculation significantly faster (more than 5 times for C12H12)
        if NUMEXPR_EXISTS:
            ao_value_block_T = ao_value_block.T
            z = numexpr.evaluate('(0.5*F*ao_value_block_T)')
        else:
            z = 0.5*np.einsum('m,mi->mi',F,ao_value_block, optimize=True)
            #z = 0.5*(np.multiply(F,ao_value.T)).T #This just made it slower Try NUMBA maybe. UPDATE: NUMBA IS 5-10 times slower for C12H12.
            #z = calcZ2(F,ao_value)
        # durationZ = durationZ + timer() - startZ
        # Free memory
        F = 0
        v_rho_temp = 0
        # print('Duration for calculation of z : ',durationZ)
        # startV = timer()
        # with threadpool_limits(limits=1, user_api='blas'):
        v_temp = z @ ao_value_block  # The fastest uptil now 
        v_temp_T = v_temp.T
        v = numexpr.evaluate('(v_temp + v_temp_T)')

        # free up memory
        # print("F size: %0.3f MB" % (F.nbytes / 1e6))
        # print("AO block size: %0.3f MB" % (ao_value_block.nbytes / 1e6))
        # print("Rho block size: %0.3f MB" % (rho_block.nbytes / 1e6))
        # print("z size: %0.3f MB" % (z.nbytes / 1e6))
        # print("vrho size: %0.3f MB" % (vrho.nbytes / 1e6))
        # print("temp size: %0.3f MB" % (temp.nbytes / 1e6))
        # print("Weights block size: %0.3f MB" % (weights_block.nbytes / 1e6))
        # print("Coords block size: %0.3f MB" % (coords_block.nbytes / 1e6))
        # print("Vxc size: %0.3f MB" % (v.nbytes / 1e6))
        # print("dmat size: %0.3f MB" % (dmat.nbytes / 1e6))
        # memory_usage_sum = (F.nbytes+ao_value_block.nbytes+rho_block.nbytes+z.nbytes+vrho.nbytes+temp.nbytes+weights_block.nbytes+coords_block.nbytes +v.nbytes +dmat.nbytes)/1e6
        # print('\nMemory usage while evaulating one block:'+str( memory_usage_sum) +' MB')
        F = 0
        z = 0
        ao_value_block = 0
        rho_block = 0
        temp = 0
        retx = 0
        retc = 0
        vrho = 0 
        weights_block=0
        coords_block=0
        func = 0
        dmat = 0
        v_temp_T = 0
        v_temp = 0
        del ao_value_block
        del rho_block
        del z
        del temp
        del F
        del weights_block
        del coords_block
        del dmat
        del vrho
        
        # gc.collect() # Too slowww
        # Getting % usage of virtual_memory ( 3rd field)
        # print('RAM memory % used:', psutil.virtual_memory()[2])



        return efunc, v, nelec

    def eval_dens_func3(basis, dmat, weights, coords, funcid=[1,7], spin=0, ncores=2, blocksize=50000, list_nonzero_indices=None, count_nonzero_indices=None, list_ao_values=None):
        # Fast but unstable on my Macbook pro with 16 Gigs of RAM
        # Could be a MacOs or my laptop issue (Needs extensive testing)
        # This uses parallelized blocks using joblib
        # In order to evaluate a density functional we will use the 
        # libxc library with Python bindings.
        # However, some sort of simple functionals like LDA, GGA, etc would
        # need to be implemented in CrysX also, so that the it doesn't depend
        # on external libraries so heavily that it becomes unusable without those.

        #This function uses the divide and conquer approach to calculate the required stuff.

        #Useful links:
        # LibXC manual: https://www.tddft.org/programs/libxc/manual/
        # LibXC gitlab: https://gitlab.com/libxc/libxc/-/tree/master
        # LibXC python interface code: https://gitlab.com/libxc/libxc/-/blob/master/pylibxc/functional.py
        # LibXC python version installation and example: https://www.tddft.org/programs/libxc/installation/
        # Formulae for XC energy and potential calculation: https://pubs.acs.org/doi/full/10.1021/ct200412r
        # LibXC code list: https://github.com/pyscf/pyscf/blob/master/pyscf/dft/libxc.py
        # PySCF nr_rks code: https://github.com/pyscf/pyscf/blob/master/pyscf/dft/numint.py
        # https://www.osti.gov/pages/servlets/purl/1650078

        #OUTPUT
        #Functional energy
        efunc = 0.0
        #Functional potential V_{\mu \nu} = \mu|\partial{f}/\partial{\rho}|\nu
        v = np.zeros((basis.bfs_nao, basis.bfs_nao))
        nelec = 0

        #TODO it only works for LDA functionals for now.
        #Need to make it work for GGA, Hybrid, range-separated Hybrid and MetaGGA functionals as well.
        

        ngrids = coords.shape[0]
        nblocks = ngrids//blocksize

        # print('Number of blocks: ', nblocks)

        durationLibxc = 0.0
        durationE = 0.0
        durationF = 0.0
        durationZ = 0.0
        durationV = 0.0
        durationRho = 0.0
        durationAO = 0.0

        ### Calculate stuff necessary for bf/ao evaluation on grid points
        ### Doesn't make any difference for 510 bfs but might be significant for >1000 bfs
        # This will help to make the call to evalbfnumba1 faster by skipping the mediator evalbfnumbawrap function 
        # that prepares the following stuff at every iteration
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
        bfs_radius_cutoff = np.zeros([basis.bfs_nao])
        for i in range(basis.bfs_nao):
            for j in range(basis.bfs_nprim[i]):
                bfs_coeffs[i,j] = basis.bfs_coeffs[i][j]
                bfs_expnts[i,j] = basis.bfs_expnts[i][j]
                bfs_prim_norms[i,j] = basis.bfs_prim_norms[i][j]
                bfs_radius_cutoff[i] = basis.bfs_radius_cutoff[i]
        # Now bf/ao values can be evaluated by calling the following
        # bf_values = evalBFsNumba2(bfs_coords[0], bfs_contr_prim_norms[0], bfs_nprim[0], bfs_lmn[0], bfs_coeffs, bfs_prim_norms, bfs_expnts, bfs_radius_cutoff, coord)
        bfs_data_as_np_arrays = [bfs_coords[0], bfs_contr_prim_norms[0], bfs_nprim[0], bfs_lmn[0], bfs_coeffs, bfs_prim_norms, bfs_expnts, bfs_radius_cutoff]

        xc_family_dict = {1:'LDA',2:'GGA',4:'MGGA'} 
        # Create a LibXC object  
        funcx = pylibxc.LibXCFunctional(funcid[0], "unpolarized")
        funcc = pylibxc.LibXCFunctional(funcid[1], "unpolarized")
        x_family_code = funcx.get_family()
        c_family_code = funcc.get_family()

        if list_nonzero_indices is not None:
            if list_ao_values is not None:
                output = Parallel(n_jobs=ncores, backend='threading', require='sharedmem')(delayed(Integrals.block_dens_func)(iblock, blocksize, weights[iblock*blocksize : min(iblock*blocksize+blocksize,ngrids)], coords[iblock*blocksize : min(iblock*blocksize+blocksize,ngrids)], ngrids, basis, dmat[np.ix_(list_nonzero_indices[iblock][0:count_nonzero_indices[iblock]], list_nonzero_indices[iblock][0:count_nonzero_indices[iblock]])], funcid, bfs_data_as_np_arrays, list_nonzero_indices[iblock][0:count_nonzero_indices[iblock]], list_ao_values[iblock], funcx=funcx, funcc=funcc, x_family_code=x_family_code, c_family_code=c_family_code, xc_family_dict=xc_family_dict) for iblock in range(nblocks+1))
            else:
              output = Parallel(n_jobs=ncores, backend='threading', require='sharedmem')(delayed(Integrals.block_dens_func)(iblock, blocksize, weights[iblock*blocksize : min(iblock*blocksize+blocksize,ngrids)], coords[iblock*blocksize : min(iblock*blocksize+blocksize,ngrids)], ngrids, basis, dmat[np.ix_(list_nonzero_indices[iblock][0:count_nonzero_indices[iblock]], list_nonzero_indices[iblock][0:count_nonzero_indices[iblock]])], funcid, bfs_data_as_np_arrays, list_nonzero_indices[iblock][0:count_nonzero_indices[iblock]], funcx=funcx, funcc=funcc, x_family_code=x_family_code, c_family_code=c_family_code, xc_family_dict=xc_family_dict) for iblock in range(nblocks+1))
        else:
            output = Parallel(n_jobs=ncores, backend='threading', require='sharedmem')(delayed(Integrals.block_dens_func)(iblock, blocksize, weights[iblock*blocksize : min(iblock*blocksize+blocksize,ngrids)], coords[iblock*blocksize : min(iblock*blocksize+blocksize,ngrids)], ngrids, basis, dmat, funcid, bfs_data_as_np_arrays, non_zero_indices=None, ao_values=None, funcx=funcx, funcc=funcc, x_family_code=x_family_code, c_family_code=c_family_code, xc_family_dict=xc_family_dict) for iblock in range(nblocks+1))
            
        # print(len(output))
        for iblock in range(0,len(output)):
            non_zero_indices = list_nonzero_indices[iblock][0:count_nonzero_indices[iblock]]
            efunc += output[iblock][0]
            if list_nonzero_indices is not None:
                v[np.ix_(non_zero_indices, non_zero_indices)] += output[iblock][1]
            else:
                v += output[iblock][1]
            nelec += output[iblock][2]
            # v = numexpr.evaluate('(v + output[iblock][1])')

        

            
        print('Number of electrons: ', nelec)

        ####### Free memory
        ## The following is very important to prevent memory leaks and also to make sure that the number of 
        # threads used by the program is same as that specified by the user 
        # gc.collect()  # Avoiding using it for now, as it is usually quite slow, although in this case it might not make much difference
        # Anyway, the following also works
        output = 0
        non_zero_indices = 0
        coords = 0

        return efunc, v


    def nonzero_ao_indices(basis, coords, blocksize, nblocks, ngrids):
        bfs_coords = np.array([basis.bfs_coords])
        bfs_radius_cutoff = np.zeros([basis.bfs_nao])
        for i in range(basis.bfs_nao):
            bfs_radius_cutoff[i] = basis.bfs_radius_cutoff[i]
        # Calculate the value of basis functions for all grid points in batches
        # and find the indices of basis functions that have a significant contribution to those batches for each batch
        list_nonzero_indices = []
        count_nonzero_indices = []
        # Loop over batches
        for iblock in range(nblocks+1):
            offset = iblock*blocksize
            coords_block = coords[offset : min(offset+blocksize,ngrids)]   
            nonzero_indices, count = nonZeroBFIndicesNumba2(coords_block, bfs_coords[0], bfs_radius_cutoff)
            list_nonzero_indices.append(nonzero_indices)
            count_nonzero_indices.append(count)
        return list_nonzero_indices, count_nonzero_indices

        
        

