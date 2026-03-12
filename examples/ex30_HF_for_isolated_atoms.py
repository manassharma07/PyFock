from pyfock import Mol
from pyfock import Basis
from pyfock import HF_atoms

# 1. Create a Mol object for an isolated atom (placed at origin)
atom_symbol = 'C'  # Carbon atom
mol = Mol(atoms=[[atom_symbol, 0.0, 0.0, 0.0]])

# 2. Create a Basis object
basis = Basis(mol, {'all':Basis.load(mol=mol, basis_name='def2-SVP')})

# 3. Create the HF_atoms object and run SCF
hf = HF_atoms(mol, basis)
Etot, Da, Db = hf.scf()

# 4. Inspect results
print(f"Converged: {hf.converged}")
print(f"Total Energy: {hf.Total_energy} Hartree")
print(f"Method used: {'UHF' if hf.is_uhf else 'RHF'}")
print(f"Alpha MO energies: {hf.mo_energies_alpha}")
print(f"Beta MO energies:  {hf.mo_energies_beta}")