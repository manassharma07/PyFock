"""
Validate rys_3c2e_grad against PySCF numerical gradients.

PySCF sign conventions:
    int3c2e_ip1  ->  -d(AB|C)/dA   (negative sign!)
    int3c2e_ip2  ->  -d(AB|C)/dC   (negative sign!)
"""

import numpy as np
import os
from timeit import default_timer as timer

import numba
numba.set_num_threads(4)
os.environ['OMP_NUM_THREADS'] = '4'

from pyfock import Basis, Mol, Integrals
from pyscf import gto, df

# ── Molecule ──────────────────────────────────────────────────────────────────
xyzFilename    = 'H2O.xyz'
basis_set_name = 'sto-3g'
auxbasisName   = 'sto-6g'

mol      = Mol(coordfile=xyzFilename)
basis    = Basis(mol, {'all': Basis.load(mol=mol, basis_name=basis_set_name)})
auxbasis = Basis(mol, {'all': Basis.load(mol=mol, basis_name=auxbasisName)})

from pyfock import Integrals
d3c = Integrals.rys_3c2e_grad_symm(basis, auxbasis, schwarz=True)
print('d3c shape', d3c.shape)   # (natoms,3,nbf,nbf,naux)

# --- PySCF reference ---
from pyscf import gto, df
mol = gto.M(atom=xyzFilename, basis='6-31G', cart=True)
aux = df.make_auxmol(mol, 'def2-universal-jfit')

eri0 = df.incore.aux_e2(mol, aux, intor='int3c2e_sph')  # or 'int3c2e_cart'

h = 1e-4
at = 0; xyz = 0   # displace atom 0 along x
molp = mol.copy(); c = molp.atom_coords()
c[at,xyz] += h; molp.set_geom_(c, unit='Bohr')
erip = df.incore.aux_e2(molp, aux, intor='int3c2e_cart')

molm = mol.copy(); c = molm.atom_coords()
c[at,xyz] -= h; molm.set_geom_(c, unit='Bohr')
erim = df.incore.aux_e2(molm, aux, intor='int3c2e_cart')

fd = (erip - erim)/(2*h)
ana = d3c[at, xyz]   # our analytic

print('max |fd - ana| =', np.abs(fd-ana).max())

# # ── PySCF molecule ────────────────────────────────────────────────────────────
# molPySCF = gto.Mole()
# molPySCF.atom  = xyzFilename
# molPySCF.basis = basis_set_name
# molPySCF.cart  = True
# molPySCF.unit  = 'Angstrom'
# molPySCF.build()
# auxmol = df.addons.make_auxmol(molPySCF, auxbasisName)

# Nbf    = molPySCF.nao_cart()
# Nauxbf = auxmol.nao_cart()
# natoms = molPySCF.natm

# print(f"Nbf = {Nbf}, Nauxbf = {Nauxbf}, natoms = {natoms}")


# # ── AO -> atom map ────────────────────────────────────────────────────────────
# def make_ao_atom_map(mol_obj):
#     """
#     Returns int array of shape (nao,): entry i = atom index owning AO i.
#     Uses ao_labels() which exists in all PySCF versions.
#     """
#     labels   = mol_obj.ao_labels(fmt=False)   # list of (atom_idx, sym, nl, ml)
#     ao_atom  = np.array([lbl[0] for lbl in labels], dtype=int)
#     return ao_atom


# ao_atom     = make_ao_atom_map(molPySCF)
# aux_ao_atom = make_ao_atom_map(auxmol)

# print(f"ao_atom      : {ao_atom}")
# print(f"aux_ao_atom  : {aux_ao_atom[:10]} ...")


# # ── Sanity: base 3c2e integrals ───────────────────────────────────────────────
# print('\n' + '=' * 60)
# print('Sanity check: base 3c2e integrals')

# from pyfock.Integrals import rys_3c2e_symm   # adjust if needed
# ints_pyfock = rys_3c2e_symm(basis, auxbasis)           # (Nbf, Nbf, Nauxbf)
# ints_pyscf  = df.incore.aux_e2(molPySCF, auxmol, intor='int3c2e')

# print(f'  PyFock shape  : {ints_pyfock.shape}')
# print(f'  PySCF  shape  : {ints_pyscf.shape}')

# diff_base    = np.abs(ints_pyfock - ints_pyscf)
# rel_diff_base = diff_base / (np.abs(ints_pyscf) + 1e-30)

# print(f'  Max |diff|         : {diff_base.max():.3e}')
# print(f'  Mean |diff|        : {diff_base.mean():.3e}')
# print(f'  Max rel |diff|     : {rel_diff_base.max():.3e}')
# print(f'  PyFock[0,0,0]      : {ints_pyfock[0,0,0]:.10f}')
# print(f'  PySCF [0,0,0]      : {ints_pyscf[0,0,0]:.10f}')

# # Find where max diff occurs
# idx = np.unravel_index(diff_base.argmax(), diff_base.shape)
# print(f'  Max diff at index  : {idx}')
# print(f'    PyFock value     : {ints_pyfock[idx]:.10f}')
# print(f'    PySCF  value     : {ints_pyscf[idx]:.10f}')

# if diff_base.max() > 1e-4:
#     print('\n  *** Large base integral error — gradient comparison will be '
#           'unreliable until this is fixed. ***')
#     print('  Continuing anyway for diagnostic purposes ...')
# else:
#     print('  OK')


# # ── PyFock analytical gradient ────────────────────────────────────────────────
# print('\n' + '=' * 60)
# print('PyFock: analytical gradient of 3c2e integrals')
# t0 = timer()
# d3c2e = Integrals.rys_3c2e_grad_symm(basis, auxbasis)
# print(f'  Shape            : {d3c2e.shape}')
# print(f'  Time             : {timer()-t0:.3f} s')
# print(f'  Max abs value    : {np.max(np.abs(d3c2e)):.6e}')
# print(f'  Fraction > 1e-10 : {np.mean(np.abs(d3c2e) > 1e-10):.4f}')

# # Translational invariance
# pf_sum = d3c2e.sum(axis=0)
# print(f'  Trans. inv. max |sum| : {np.abs(pf_sum).max():.3e}')


# # ── PySCF analytical gradient ─────────────────────────────────────────────────
# print('\n' + '=' * 60)
# print('PySCF: analytical gradient of 3c2e integrals')
# t0 = timer()

# # ip1[xi, i, j, k] = -d(ij|k)/d(center_of_i)[xi]
# # ip2[xi, i, j, k] = -d(ij|k)/d(center_of_k)[xi]
# int3c2e_ip1 = df.incore.aux_e2(molPySCF, auxmol, intor='int3c2e_ip1')
# int3c2e_ip2 = df.incore.aux_e2(molPySCF, auxmol, intor='int3c2e_ip2')
# print(f'  int3c2e_ip1 shape : {int3c2e_ip1.shape}')   # (3, Nbf, Nbf, Nauxbf)
# print(f'  int3c2e_ip2 shape : {int3c2e_ip2.shape}')   # (3, Nbf, Nbf, Nauxbf)
# print(f'  Time              : {timer()-t0:.3f} s')

# # ip1_ji[xi, i, j, k] = ip1[xi, j, i, k]
# # = -d(ji|k)/d(center_of_j) = -d(ij|k)/d(center_of_j)   [since (ij|k)=(ji|k)]
# ip1_ji = np.transpose(int3c2e_ip1, (0, 2, 1, 3))   # (3, Nbf, Nbf, Nauxbf)

# # Build atom-resolved gradient using simple loops over atoms.
# # d3c2e_pyscf[atom, xi, i, j, k] = d(ij|k)/dR_atom[xi]
# #
# #   (1) d/dA : AO i on atom  ->  -ip1[xi, i, j, k]
# #   (2) d/dB : AO j on atom  ->  -ip1_ji[xi, i, j, k]  (= -ip1[xi, j, i, k])
# #   (3) d/dC : aux k on atom ->  -ip2[xi, i, j, k]

# d3c2e_pyscf = np.zeros((natoms, 3, Nbf, Nbf, Nauxbf))

# for atom in range(natoms):
#     mask_i = np.where(ao_atom     == atom)[0]   # primary AOs on this atom
#     mask_k = np.where(aux_ao_atom == atom)[0]   # aux AOs on this atom

#     # (1) d/dA : loop over AOs i on this atom
#     for i in mask_i:
#         # d3c2e_pyscf[atom, :, i, :, :] += -ip1[:, i, :, :]
#         d3c2e_pyscf[atom, :, i, :, :] += -int3c2e_ip1[:, i, :, :]

#     # (2) d/dB : AO j on this atom -> use ip1_ji
#     for j in mask_i:   # same atom mask for j
#         # d3c2e_pyscf[atom, :, :, j, :] += -ip1_ji[:, :, j, :]
#         #                                 = -ip1[:, j, :, :]  transposed back
#         d3c2e_pyscf[atom, :, :, j, :] += -ip1_ji[:, :, j, :]

#     # (3) d/dC : aux AO k on this atom
#     for k in mask_k:
#         # d3c2e_pyscf[atom, :, :, :, k] += -ip2[:, :, :, k]
#         d3c2e_pyscf[atom, :, :, :, k] += -int3c2e_ip2[:, :, :, k]

# print(f'\n  d3c2e_pyscf max abs       : {np.max(np.abs(d3c2e_pyscf)):.6e}')
# py_sum = d3c2e_pyscf.sum(axis=0)
# print(f'  PySCF trans. inv. max|sum|: {np.abs(py_sum).max():.3e}')


# # ── PySCF numerical gradient (finite difference) ──────────────────────────────
# def build_mol_pyscf(coords_bohr):
#     m = gto.Mole()
#     m.atom  = [(molPySCF.atom_symbol(i), coords_bohr[i])
#                for i in range(molPySCF.natm)]
#     m.basis = basis_set_name
#     m.cart  = True
#     m.unit  = 'Bohr'
#     m.build(dump_input=False, parse_arg=False, verbose=0)
#     return m


# def numerical_grad_pyscf(atom_idx, direction, h=1e-4):
#     """Central-difference: d(ij|k)/dR_atom[dir]. Shape (Nbf, Nbf, Nauxbf)."""
#     coords   = molPySCF.atom_coords().copy()   # Bohr

#     coords_p = coords.copy(); coords_p[atom_idx, direction] += h
#     coords_m = coords.copy(); coords_m[atom_idx, direction] -= h

#     mol_p = build_mol_pyscf(coords_p)
#     mol_m = build_mol_pyscf(coords_m)
#     aux_p = df.addons.make_auxmol(mol_p, auxbasisName)
#     aux_m = df.addons.make_auxmol(mol_m, auxbasisName)

#     eri_p = df.incore.aux_e2(mol_p, aux_p, intor='int3c2e')
#     eri_m = df.incore.aux_e2(mol_m, aux_m, intor='int3c2e')
#     return (eri_p - eri_m) / (2.0 * h)


# print('\nNumerical gradient (finite difference):')
# num_grad = np.zeros((natoms, 3, Nbf, Nbf, Nauxbf))
# for atom_idx in range(natoms):
#     for direction in range(3):
#         print(f'  FD: atom {atom_idx}, dir {direction}')
#         num_grad[atom_idx, direction] = numerical_grad_pyscf(atom_idx, direction)

# print(f'\n  num_grad max abs     : {np.max(np.abs(num_grad)):.6e}')
# print(f'  num_grad[0,0,0,0,0]  : {num_grad[0,0,0,0,0]:.10f}')
# print(f'  Fraction > 1e-10     : {np.mean(np.abs(num_grad) > 1e-10):.4f}')
# nu_sum = num_grad.sum(axis=0)
# print(f'  Trans. inv. max|sum| : {np.abs(nu_sum).max():.3e}')


# # ── Ground truth: PySCF analytical vs numerical ───────────────────────────────
# print('\n' + '=' * 60)
# print('GROUND TRUTH: PySCF analytical vs numerical')
# diff_py_nu = np.abs(d3c2e_pyscf - num_grad)
# print(f'  Max |diff| : {diff_py_nu.max():.3e}')
# print(f'  Mean|diff| : {diff_py_nu.mean():.3e}')
# print('  (expect ~1e-7 for h=1e-4 if sign/mapping is correct)')

# # Check if we have the sign wrong
# diff_py_nu_neg = np.abs(-d3c2e_pyscf - num_grad)
# print(f'  Max |diff| (negated pyscf) : {diff_py_nu_neg.max():.3e}')
# if diff_py_nu_neg.max() < diff_py_nu.max():
#     print('  *** Sign is flipped in PySCF atom-resolved gradient! ***')

# print('\n  Spot check [atom=0, dir=x, i=0:4, j=0:4, k=0]:')
# print('    PySCF_an :', d3c2e_pyscf[0, 0, :4, :4, 0])
# print('    NumGrad  :', num_grad[0, 0, :4, :4, 0])

# print('\n  Spot check [atom=0, dir=x, i=2, j=2, k=0:5]:')
# print('    PySCF_an :', d3c2e_pyscf[0, 0, 2, 2, :5])
# print('    NumGrad  :', num_grad[0, 0, 2, 2, :5])


# # ── PyFock vs numerical ───────────────────────────────────────────────────────
# print('\n' + '=' * 60)
# print('PyFock analytical vs numerical:')
# diff_pf_nu = np.abs(d3c2e - num_grad)
# print(f'  Max |diff| : {diff_pf_nu.max():.3e}')
# print(f'  Mean|diff| : {diff_pf_nu.mean():.3e}')

# print('\n  Spot check [atom=0, dir=x, i=0:4, j=0:4, k=0]:')
# print('    PyFock   :', d3c2e[0, 0, :4, :4, 0])
# print('    PySCF_an :', d3c2e_pyscf[0, 0, :4, :4, 0])
# print('    NumGrad  :', num_grad[0, 0, :4, :4, 0])

# print('\n  Spot check [atom=0, dir=x, i=2, j=2, k=0:5]:')
# print('    PyFock   :', d3c2e[0, 0, 2, 2, :5])
# print('    PySCF_an :', d3c2e_pyscf[0, 0, 2, 2, :5])
# print('    NumGrad  :', num_grad[0, 0, 2, 2, :5])

# print('\n  Spot check [atom=1, dir=x, i=5, j=5, k=61:66]:')
# print('    PyFock   :', d3c2e[1, 0, 5, 5, 61:66])
# print('    PySCF_an :', d3c2e_pyscf[1, 0, 5, 5, 61:66])
# print('    NumGrad  :', num_grad[1, 0, 5, 5, 61:66])


# # ── PyFock vs PySCF analytical ────────────────────────────────────────────────
# print('\n' + '=' * 60)
# print('PyFock analytical vs PySCF analytical:')
# diff_pf_py = np.abs(d3c2e - d3c2e_pyscf)
# print(f'  Max |diff| : {diff_pf_py.max():.3e}')
# print(f'  Mean|diff| : {diff_pf_py.mean():.3e}')

# # Find worst element
# idx = np.unravel_index(diff_pf_py.argmax(), diff_pf_py.shape)
# print(f'  Worst element index : {idx}')
# print(f'    PyFock   : {d3c2e[idx]:.10f}')
# print(f'    PySCF_an : {d3c2e_pyscf[idx]:.10f}')
# print(f'    NumGrad  : {num_grad[idx]:.10f}')


# # ── Summary ───────────────────────────────────────────────────────────────────
# print('\n' + '=' * 60)
# print('SUMMARY')
# print(f'  Base integral error (PyFock vs PySCF) : {diff_base.max():.3e}')
# print(f'  PySCF analytical vs numerical         : {diff_py_nu.max():.3e}')
# print(f'  PyFock analytical vs numerical        : {diff_pf_nu.max():.3e}')
# print(f'  PyFock analytical vs PySCF analytical : {diff_pf_py.max():.3e}')
# print(f'  PyFock trans. inv. max|sum|           : {np.abs(pf_sum).max():.3e}')
# print(f'  PySCF  trans. inv. max|sum|           : {np.abs(py_sum).max():.3e}')
# print(f'  Numerical trans. inv. max|sum|        : {np.abs(nu_sum).max():.3e}')