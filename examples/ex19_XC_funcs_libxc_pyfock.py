import pylibxc
from pyfock import XC
import numpy as np


# TESTING
funcid = 1 
print('Functional ID in LibXC: ')
print(funcid)

rho = [1,2,3]
print('Density at grid points: ')
print(rho)

#LibXC stuff
# Create a LibXC object  
func = pylibxc.LibXCFunctional(funcid, "unpolarized")
print('Family: ', func.get_family())
print(func.describe())
# Input dictionary for libxc
inp = {}
# Input dictionary needs density values at grid points
inp['rho'] = rho
# Calculate the necessary quantities using LibXC
ret = func.compute(inp)
print('Functional values (energy density) at grid points:')
print(ret['zk'])
print('Functional potential at grid points:')
print(ret['vrho'])


# PyFock stuff
ret = XC.func_compute(funcid, rho, use_gpu=False)
print('Functional values (energy density) at grid points using PyFock implementation:')
print(ret[0])
print('Functional potential at grid points using PyFock implementation:')
print(ret[1])

