import numpy as np
import pylibxc

# https://th.fhi-berlin.mpg.de/th/Meetings/DFT-workshop-Berlin2011/presentations/2011-07-13_DellaSala_Fabio.pdf
# http://alps.comp-phys.org/mediawiki/images/8/83/Lecture2.pdf
# https://en.wikipedia.org/wiki/Local-density_approximation
# https://th.fhi-berlin.mpg.de/th/publications/santra_dissertation.pdf - Very good resource
# Dfauto: a good article. It also made me realize that instead of taking the derivative wrt density
# it can be better to take the derivative wrt density matrix elements
# https://www.sciencedirect.com/science/article/pii/S0010465501001485



def lda_x(rho):
    # name = "LDA"
    # full_name = "LDA (PZ parametrization)"
    # citation = "J. P. Perdew and A. Zunger, PRB 23, 5048 (1981)"

    lda_x_param = -0.7385587663820223   #-3/4*np.pow(3/np.pi,1/3)
    return lda_x_param*np.power(rho,1/3)


#def test():
#    # TESTING
#    funcid = 1 
#    rho = [1,2,3]
#    #LibXC stuff
#    # Create a LibXC object  
#    func = pylibxc.LibXCFunctional(funcid, "unpolarized")
#    # Input dictionary for libxc
#    inp = {}
#    # Input dictionary needs density values at grid points
#    inp['rho'] = rho
#    # Calculate the necessary quantities using LibXC
#    ret = func.compute(inp)
#    print(ret['zk'])
#
#    print(lda_x(rho))
