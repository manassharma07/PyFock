import numpy as np
from pyfock import Mol, DFT, Basis, Integrals

ncores = 4

# Build molecule and run HF
#Initialize a Mol object with somewhat large geometry
mol = Mol(coordfile='H2O.xyz')
print('\n\nNatoms :',mol.natoms)

#Initialize a Basis object with a very large basis set
basis = Basis(mol, {'all':Basis.load(mol=mol, basis_name='def2-SVP')})
print('\n\nNAO :', basis.bfs_nao)

dftObj = DFT(mol, basis, xc='HF')
dftObj.ncores = ncores
dftObj.isDF = False
dftObj.rys = False
dftObj.coul_algo = 1
dftObj.direct_scf = False
dftObj.sao = False

energy, dmat = dftObj.scf()

# Grab ingredients
nao = basis.bfs_nao
nmo = dftObj.mo_coefficients.shape[1]
nocc = mol.nelectrons//2
nvir = nmo - nocc
C = dftObj.mo_coefficients
Co = C[:, :nocc]
Cv = C[:, nocc:]
eo = dftObj.mo_energies[:nocc]
ev = dftObj.mo_energies[nocc:]

# AO→MO transform of two-electron integrals: (ia|jb)
# Full AO ERIs
# eri_ao = Integrals.rys_4c2e_symm(basis)
# eri_ao = Integrals.os_4c2e_schwarz_symm(basis)
# eri_ao = Integrals.os_4c2e_symm(basis)
eri_ao = Integrals.conv_4c2e_symm(basis)

# Transform to (ia|jb) — only need occ-vir blocks
# (pq|rs) = C_μp C_νq (μν|λσ) C_λr C_σs
# Step by step: contract each index
tmp = np.einsum('uvls,ui->ivls', eri_ao, Co)
tmp = np.einsum('ivls,va->ials', tmp, Cv)
tmp = np.einsum('ials,lj->iajs', tmp, Co)
eri_iajb = np.einsum('iajs,sb->iajb', tmp, Cv)

# Build A matrix: A_{ia,jb} = (ea - ei) δij δab + 2(ia|jb) - (ij|ab)
# For TDHF/TDA (no XC kernel, just exchange)

# Need (ij|ab) as well
tmp2 = np.einsum('uvls,ui->ivls', eri_ao, Co)
tmp2 = np.einsum('ivls,lj->ivjs', tmp2, Co)
tmp2 = np.einsum('ivjs,va->iajs', tmp2, Cv)  # reuse index
eri_ijab = np.einsum('uajs,sb->uajb', tmp2, Cv)  # this is (ij|ab)
tmp3 = np.einsum('uvls,ui->ivls', eri_ao, Co)
tmp3 = np.einsum('ivls,vj->ijls', tmp3, Co)
tmp3 = np.einsum('ijls,la->ijas', tmp3, Cv)
eri_ijab = np.einsum('ijas,sb->ijab', tmp3, Cv)

# Reshape to 2D: composite index (ia), (jb)
A = np.zeros((nocc * nvir, nocc * nvir))

for i in range(nocc):
    for a in range(nvir):
        ia = i * nvir + a
        for j in range(nocc):
            for b in range(nvir):
                jb = j * nvir + b
                A[ia, jb] = 2.0 * eri_iajb[i, a, j, b] - eri_ijab[i, j, a, b]
                if i == j and a == b:
                    A[ia, jb] += ev[a] - eo[i]

# Diagonalize
eigenvalues, eigenvectors = np.linalg.eigh(A)

# Excitation energies in eV
hartree_to_ev = 27.2114
print("First 10 excitation energies using PyFock (eV):")
for i in range(min(10, len(eigenvalues))):
    print(f"  State {i+1}: {eigenvalues[i] * hartree_to_ev:.4f} eV")

# Oscillator strengths
dip_ao = Integrals.dipole_moment_mat_symm(basis)  # (3, nao, nao)
dip_mo_ia = np.einsum('xmn,mi,na->xia', dip_ao, Co, Cv)
dip_mo_ia = dip_mo_ia.reshape(3, nocc * nvir)

print("\n--- PyFock result ---")
print("\nState   Energy(eV)   f")
for s in range(min(10, len(eigenvalues))):
    tdm = dip_mo_ia @ eigenvectors[:, s]  # transition dipole (x,y,z)
    f = (2.0 / 3.0) * eigenvalues[s] * np.dot(tdm, tdm)
    print(f"  {s+1:3d}    {eigenvalues[s]*hartree_to_ev:8.4f}    {f:.6f}")

# Verify against PySCF's built-in TDHF/TDA
print("\n--- PySCF reference ---")
from pyscf import tdscf, gto, scf
# Build molecule and run HF
mol = gto.M(
    atom='H2O.xyz',
    basis='def2-SVP',
    cart=True
)
mf = scf.RHF(mol).run()
td = tdscf.TDA(mf).run(nstates=10)
td.analyze()