from scipy.special import factorial, factorial2, binom, hyp1f1, gammainc, gamma # , comb
import numpy as np
from numba import njit 
import math
#
#
#  .d8888b.                            Y88b   d88P       8888888b.           8888888888                888      
# d88P  Y88b                            Y88b d88P        888   Y88b          888                       888      
# 888    888                             Y88o88P         888    888          888                       888      
# 888        888d888 888  888 .d8888b     Y888P          888   d88P 888  888 8888888  .d88b.   .d8888b 888  888 
# 888        888P"   888  888 88K         d888b          8888888P"  888  888 888     d88""88b d88P"    888 .88P 
# 888    888 888     888  888 "Y8888b.   d88888b  888888 888        888  888 888     888  888 888      888888K  
# Y88b  d88P 888     Y88b 888      X88  d88P Y88b        888        Y88b 888 888     Y88..88P Y88b.    888 "88b 
#  "Y8888P"  888      "Y88888  88888P' d88P   Y88b       888         "Y88888 888      "Y88P"   "Y8888P 888  888 
#                         888                                            888                                    
#                    Y8b d88P                                       Y8b d88P                                    
#                     "Y88P"                                         "Y88P"                                       

@njit(cache=True, fastmath=True, error_model="numpy")
def fac(n):
    if n <= 0:
        return 1
    else:
        return n * doublefactorial(n-1)

@njit(cache=True, fastmath=True, error_model="numpy")
def fastFactorial_old(n):
    # loop is working the best
    if n<= 1:
        return 1
    else:
        factorial = 1
        for i in range(2, n+1):
            factorial *= i
        return factorial
    
LOOKUP_TABLE = np.array([
    1, 1, 2, 6, 24, 120, 720, 5040, 40320,
    362880, 3628800, 39916800, 479001600,
    6227020800, 87178291200, 1307674368000,
    20922789888000, 355687428096000, 6402373705728000,
    121645100408832000, 2432902008176640000], dtype='int64')

@njit(cache=True, fastmath=True, error_model="numpy")
def fastFactorial(n):
    # 2-3x faster than the fastFactorial_old for values less than 21
    if n<= 1:
        return 1
    elif n<=20:
        return LOOKUP_TABLE[n]
    else:
        factorial = 1
        for i in range(2, n+1):
            factorial *= i
        return factorial


@njit(cache=True, fastmath=True, error_model="numpy")
def comb(x, y):
    if y == 0: 
        return 1
    if x == y: 
        return 1
    binom = fastFactorial(x) // fastFactorial(y) // fastFactorial(x - y)
    return binom

# More compilation time
# @njit(cache=True, fastmath=True, error_model="numpy")
# def comb(x, y):
#     return binom(float(x), float(y))

@njit(cache=True, fastmath=True, error_model="numpy")
def doublefactorial_old(n):
# Double Factorial Implementation based on recursion (not numba friendly)
     if n <= 0:
         return 1
     else:
         return n * doublefactorial(n-2)

@njit(cache=True, fastmath=True, error_model="numpy")
def doublefactorial(n):
    if n <= 0:
        return 1
    else:
        result = 1
        for i in range(n, 0, -2):
            result *= i
        return result
        

@njit(cache=True, fastmath=True, error_model="numpy")   
def c2k(k,la,lb,PA,PB):
    temp = 0.0
    for i in range(la+1):
        if i>k:
            continue
        factor1 = comb(la,i)
        factor2 = PA**(la-i)
        for j in range(lb+1):
            # if j>k:
            #     continue
            if (i+j)==k :
                temp +=  factor1*comb(lb,j)*factor2*PB**(lb-j)
    return temp

@njit(cache=True, fastmath=True, error_model="numpy")
def calcS(la,lb,gamma,PA,PB):
    temp = 0.0
    fac1 = np.sqrt(np.pi/gamma)
    fac2 = 2*gamma
    for k in range(0, int((la+lb)/2)+1):
        temp +=  c2k(2*k,la,lb,PA,PB)*fac1*doublefactorial(2*k-1)/(fac2)**k
    return temp

@njit(cache=True, fastmath=True, error_model="numpy")
def vlriPartial(Ci, l,r,i):
    return (-1)**l*((-1)**i*fastFactorial(l)*Ci**(l-2*r-2*i)/(fastFactorial(r)*fastFactorial(i)*fastFactorial(l-2*r-2*i)))



@njit(cache=True, fastmath=True, error_model="numpy")
def Fboys_old(v,x):
    # From: https://pubs.acs.org/doi/full/10.1021/acs.jchemed.8b00255
    #from scipy.special import gammainc, gamma
    if x >= 0 and x < 0.0000001:
        F = 1/(2*v+1) - x/(2*v+3)
    else:
        F = 0.5*x**(-(v+0.5))*gammainc(v+0.5,x)*gamma(v+0.5)
    return F


# Adapted from
# ----------
# McMurchie-Davidson project:
# https://github.com/jjgoings/McMurchie-Davidson
# Licensed under the BSD-3-Clause license
@njit(cache=True)
def Fboys_jjgoings(v,x):
    #from scipy.special import hyp1f1
    F = hyp1f1(v+0.5,v+1.5,-x)/(2.0*v+1.0)
    return F

@njit(cache=True, fastmath=True, error_model="numpy")
def theta(l,la,lb,PA,PB,gamma_,r):
    return c2k(l,la,lb,PA,PB)*fastFactorial(l)*(gamma_**(r-l))/(fastFactorial(r)*fastFactorial(l-2*r))

@njit(cache=True, fastmath=True, error_model="numpy")
def g(lp,lq,rp,rq,i,la,lb,lc,ld,gammaP,gammaQ,PA,PB,QC,QD,PQ,delta):
        temp = ((-1)**lp)*theta(lp,la,lb,PA,PB,gammaP,rp)*theta(lq,lc,ld,QC,QD,gammaQ,rq)
        numerator = temp*((-1)**i)*((2*delta)**(2*(rp+rq)))*fastFactorial(lp+lq-2*rp-2*rq)*(delta**i)*(PQ**(lp+lq-2*(rp+rq+i)))
        denominator = ((4*delta)**(lp+lq))*fastFactorial(i)*fastFactorial(lp+lq-2*(rp+rq+i))
        # print(numerator/temp)
        return (numerator/denominator)

@njit(cache=True, fastmath=True, error_model="numpy")
def innerLoop4c2e(la,lb,lc,ld,ma,mb,mc,md,na,nb,nc,nd,gammaP,gammaQ,PI,PJ,QK,QL,PQ,PQsqBy4delta,delta):
    sum1 = 0.0
    for lp in range(0,la+lb+1):
        for rp in range(0, int(lp/2)+1):
            for lq in range(0, lc+ld+1):
                for rq in range(0, int(lq/2)+1):
                    for i1 in range(0,int((lp+lq-2*rp-2*rq)/2)+1):
                        gx = g(lp,lq,rp,rq,i1,la,lb,lc,ld,gammaP,gammaQ,PI[0],PJ[0],QK[0],QL[0],PQ[0],delta)
                        sum2 = 0.0
                        for mp in range(0,ma+mb+1):
                            for sp in range(0, int(mp/2)+1):
                                for mq in range(0, mc+md+1):
                                    for sq in range(0, int(mq/2)+1):
                                        for j1 in range(0,int((mp+mq-2*sp-2*sq)/2)+1):
                                            gy = g(mp,mq,sp,sq,j1,ma,mb,mc,md,gammaP,gammaQ,PI[1],PJ[1],QK[1],QL[1],PQ[1],delta)
                                            sum3 = 0.0                                                                   
                                            for np1 in range(0,na+nb+1):
                                                for tp in range(0, int(np1/2)+1):
                                                    for nq in range(0, nc+nd+1):
                                                        for tq in range(0, int(nq/2)+1):
                                                            for k1 in range(0,int((np1+nq-2*tp-2*tq)/2)+1):
                                                                gz = g(np1,nq,tp,tq,k1,na,nb,nc,nd,gammaP,gammaQ,PI[2],PJ[2],QK[2],QL[2],PQ[2],delta)
                                                                v = lp+lq+mp+mq+np1+nq-2*(rp+rq+sp+sq+tp+tq)-(i1+j1+k1)
                                                                F = Fboys(v,PQsqBy4delta)
                                                                sum3 = sum3 + gz*F                                                                                   
                                            sum2 = sum2 + gy*sum3
                        sum1 = sum1 + gx*sum2
    return sum1

@njit(cache=False, fastmath=True, error_model="numpy")
def hermite_gauss_coeff(i,j,t,Qx,a,b,p=None,q=None): 
    ''' Recursive definition of Hermite Gaussian coefficients.
        Returns a float.
        Adapted from
        ----------
        McMurchie-Davidson project:
        https://github.com/jjgoings/McMurchie-Davidson
        Licensed under the BSD-3-Clause license
        Source: https://github.com/jjgoings/McMurchie-Davidson/blob/master/mmd/integrals/reference.py
        a: orbital exponent on Gaussian 'a' (e.g. alpha in the text)
        b: orbital exponent on Gaussian 'b' (e.g. beta in the text)
        i,j: orbital angular momentum number on Gaussian 'a' and 'b'
        t: number nodes in Hermite (depends on type of integral, 
           e.g. always zero for overlap integrals)
        Qx: distance between origins of Gaussian 'a' and 'b'
    '''
    if p is None:
        p = a + b
    if q is None:
        q = a*b/p
    if (t < 0) or (t > (i + j)):
        # out of bounds for t  
        return 0.0
    elif i == j == t == 0:
        # base case
        return np.exp(-q*Qx*Qx) # K_AB
    elif j == 0:
        # decrement index i
        return (1/(2*p))*hermite_gauss_coeff(i-1,j,t-1,Qx,a,b,p,q) - \
               (q*Qx/a)*hermite_gauss_coeff(i-1,j,t,Qx,a,b,p,q)    + \
               (t+1)*hermite_gauss_coeff(i-1,j,t+1,Qx,a,b,p,q)
    else:
        # decrement index j
        return (1/(2*p))*hermite_gauss_coeff(i,j-1,t-1,Qx,a,b,p,q) + \
               (q*Qx/b)*hermite_gauss_coeff(i,j-1,t,Qx,a,b,p,q)    + \
               (t+1)*hermite_gauss_coeff(i,j-1,t+1,Qx,a,b,p,q)
    

    
@njit(cache=False, fastmath=True, error_model="numpy")
def aux_hermite_int(t,u,v,n,p,PCx,PCy,PCz,RPC,T=None,boys=None,n_min=None):
    ''' Returns the Coulomb auxiliary Hermite integrals 
        Returns a float.
        Adapted from
        ----------
        McMurchie-Davidson project:
        https://github.com/jjgoings/McMurchie-Davidson
        Licensed under the BSD-3-Clause license
        Source: https://github.com/jjgoings/McMurchie-Davidson/blob/master/mmd/integrals/reference.py
        Arguments:
        t,u,v:   order of Coulomb Hermite derivative in x,y,z
                 (see defs in Helgaker and Taylor)
        n:       order of Boys function 
        PCx,y,z: Cartesian vector distance between Gaussian 
                 composite center P and nuclear center C
        RPC:     Distance between P and C
    '''
    if T is None:
        T = p*RPC*RPC
    val = 0.0
    # print(t,u,v,n)
    if t == u == v == 0:
        # print(t,u,v,n)
        # print(n_min)
        # print('here1')
        if boys is None and n_min is None:
            # print('Here')
            val += np.power(-2*p,n)*Fboys(n,T)
        else:
            # print('sss')
            val += boys[n-n_min]
    elif t == u == 0:
        # print('here2')
        if v > 1:
            # print('here3')
            val += (v-1)*aux_hermite_int(t,u,v-2,n+1,p,PCx,PCy,PCz,RPC,T,boys,n_min)
        # print('here4')
        val += PCz*aux_hermite_int(t,u,v-1,n+1,p,PCx,PCy,PCz,RPC,T,boys,n_min)
    elif t == 0:
        # print('here5')
        if u > 1:
            # print('here6')
            val += (u-1)*aux_hermite_int(t,u-2,v,n+1,p,PCx,PCy,PCz,RPC,T,boys,n_min)
        val += PCy*aux_hermite_int(t,u-1,v,n+1,p,PCx,PCy,PCz,RPC,T,boys,n_min)
    else:
        if t > 1:
            # print('here7')
            val += (t-1)*aux_hermite_int(t-2,u,v,n+1,p,PCx,PCy,PCz,RPC,T,boys,n_min)
        val += PCx*aux_hermite_int(t-1,u,v,n+1,p,PCx,PCy,PCz,RPC,T,boys,n_min)
    return val


# Alternative boys implementation from this repo: https://github.com/peter-reinholdt/pyboys
# The repo has the BSD-3 license.
# This implementation is faster and it should help provide upto
# 10-40 percent speed up.
# Another advantage of this implementation is that it can be cached by Numba, since it is not dependent on the functions
# from numba scipy package. 
import math
from .taylor import taylor

TAYLOR_THRESHOLD = -25.0


@njit(cache=True, fastmath=True, error_model="numpy")
def hyp0minus(x):
    z = math.sqrt(-x)
    return 0.5 * math.erf(z) * math.sqrt(math.pi) / z


@njit(cache=True, fastmath=True, error_model="numpy")
def hyp1f1_(m, z):
    # print('here')
    if z < TAYLOR_THRESHOLD:
        return hyp0minus(z) if m == 0 else (hyp1f1_(m-1, z)*(2*m+1) - math.exp(z))  / (-2*z)
    else:
        # print('taylor')
        return taylor(m, z)


@njit(cache=True, fastmath=True, error_model="numpy")
def Fboys(m, T):
    return hyp1f1_(m, -T) / (2*m+1)