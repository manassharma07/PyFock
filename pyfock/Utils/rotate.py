# A Python script to rotate a molecule about an axis by theta degrees 
import numpy as np

def rotate2D(mol_in, theta, mol_out):
    # Rotates a molecules in the xy plane by theta degrees
    mol_out = mol_in
    for i in range(natoms):
        x = mol_in.coords[i][0]
        y = mol_in.coords[i][1]
        # Rotate
        x1 = x*np.cos(theta) - y*np.sin(theta)
        y1 = x*np.sin(theta) + y*np.cos(theta)
        mol_out.coords[i][0] = x1
        mol_out.coords[i][1] = y1

