from pyscf import gto
from gpu4pyscf.dft import rks
import os
from timeit import default_timer as timer

#DFT SCF benchmark and comparison with PySCF
#Benchmarking and performance assessment and comparison using various techniques and different softwares

# LDA_X LDA_C_VWN 
# funcx = 1
# funcc = 7

# LDA_X LDA_C_PW 
# funcx = 1
# funcc = 12

# LDA_X LDA_C_PW_MOD 
# funcx = 1
# funcc = 13

# GGA_X_PBE, GGA_C_PBE (PBE)
funcx = 101
funcc = 130

# GGA_X_B88, GGA_C_LYP (BLYP)
# funcx = 106
# funcc = 131

funcidcrysx = [funcx, funcc]
funcidpyscf = str(funcx)+','+str(funcc)

# basis_set_name = 'sto-2g'
# basis_set_name = 'sto-3g'
# basis_set_name = 'sto-6g'
# basis_set_name = '6-31G'
basis_set_name = 'def2-SVP'
# basis_set_name = 'def2-SVPD'
# basis_set_name = 'def2-TZVP'
# basis_set_name = 'def2-QZVP'
# basis_set_name = 'def2-TZVPP'
# basis_set_name = 'def2-QZVPP'
# basis_set_name = 'def2-TZVPD'
# basis_set_name = 'def2-QZVPD'
# basis_set_name = 'def2-TZVPPD'
# basis_set_name = 'def2-QZVPPD'
# basis_set_name = 'cc-pVDZ'
# basis_set_name = 'ano-rcc'

auxbasis_name = 'def2-universal-jfit'
# auxbasis_name = 'def2-TZVP'
# auxbasis_name = 'sto-3g'
# auxbasis_name = 'def2-SVP'
# auxbasis_name = '6-31G'

# xyzFilename = 'Benzene-Fulvene_Dimer.xyz'
# xyzFilename = 'Adenine-Thymine.xyz'
# xyzFilename = 'Zn.xyz'
# xyzFilename = 'Zn_dimer.xyz'
# xyzFilename = 'TPP.xyz'
# xyzFilename = 'Zn_TPP.xyz'
# xyzFilename = 'H2O.xyz'

# xyzFilename = 'Caffeine.xyz'
# xyzFilename = 'Serotonin.xyz'
# xyzFilename = 'Cholesterol.xyz'
# xyzFilename = 'C60.xyz'
xyzFilename = 'Taxol.xyz'
# xyzFilename = 'Valinomycin.xyz'
# xyzFilename = 'Olestra.xyz'
# xyzFilename = 'Ubiquitin.xyz'

### 1D Carbon Alkanes
# xyzFilename = 'Ethane.xyz'
# xyzFilename = 'Decane_C10H22.xyz'
# xyzFilename = 'Icosane_C20H42.xyz'
# xyzFilename = 'Tetracontane_C40H82.xyz'
# xyzFilename = 'Pentacontane_C50H102.xyz'
# xyzFilename = 'Octacontane_C80H162.xyz'
# xyzFilename = 'Hectane_C100H202.xyz'
# xyzFilename = 'Icosahectane_C120H242.xyz'

### 2D Carbon
# xyzFilename = 'Graphene_C16.xyz'
# xyzFilename = 'Graphene_C76.xyz'
# xyzFilename = 'Graphene_C102.xyz'
# xyzFilename = 'Graphene_C184.xyz'
# xyzFilename = 'Graphene_C210.xyz'
# xyzFilename = 'Graphene_C294.xyz'

### 3d Carbon Fullerenes
# xyzFilename = 'C60.xyz'
# xyzFilename = 'C70.xyz'
# xyzFilename = 'Graphene_C102.xyz'
# xyzFilename = 'Graphene_C184.xyz'
# xyzFilename = 'Graphene_C210.xyz'
# xyzFilename = 'Graphene_C294.xyz'

molPySCF = gto.Mole()
molPySCF.atom = xyzFilename
molPySCF.basis = basis_set_name
#molPySCF.cart = True
molPySCF.verbose = 5
#molPySCF.max_memory=25000
# molPySCF.incore_anyway = True # Keeps the PySCF ERI integrals incore
molPySCF.build()

print('\n\nPySCF Results\n\n')
start=timer()
mf = rks.RKS(molPySCF).density_fit(auxbasis=auxbasis_name)
mf.xc = funcidpyscf
mf.verbose = 5
mf.direct_scf = False
# mf.with_df.max_memory = 25000
# dmat_init = mf.init_guess_by_1e(molPySCF)
# dmat_init = mf.init_guess_by_huckel(molPySCF)
mf.init_guess = 'minao'
dmat_init = mf.init_guess_by_minao(molPySCF)
# mf.init_guess = 'atom'
# dmat_init = mf.init_guess_by_atom(molPySCF)
mf.max_cycle = 30
mf.conv_tol = 1e-7
mf.grids.level = 5
# print('begin df build')
# start_df_pyscf=timer()
# mf.with_df.build()
# duration_df_pyscf = timer()- start_df_pyscf
# print('PySCF df time: ', duration_df_pyscf)
# print('end df build')
energyPyscf = mf.kernel(dm0=dmat_init)
print('Nuc-Nuc PySCF= ', molPySCF.energy_nuc())
print('One electron integrals energy',mf.scf_summary['e1'])
print('Coulomb energy ',mf.scf_summary['coul'])
print('EXC ',mf.scf_summary['exc'])
duration = timer()-start
print('PySCF time: ', duration)
pyscfGrids = mf.grids
print('PySCF Grid Size: ', pyscfGrids.weights.shape)
