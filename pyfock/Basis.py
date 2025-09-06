# Basis.py
# Author: Manas Sharma (manassharma07@live.com)
# This is a part of CrysX (https://bragitoff.com/crysx)
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
import numpy as np
import scipy 
from scipy.special import factorial2, binom
import re
from . import Data
from collections import Counter

import sys
import os

# print(__file__)
# print(sys.argv[0])
# print(sys.path[0])

#Class to store basis function properties
class Basis:
    """
    Class to store and manage basis function properties for a given molecular system.

    This class processes atomic basis set information for each atom in the molecule,
    including primitive Gaussian functions, shells, and contracted basis functions (AOs).
    It also supports TURBOMOLE-style shell ordering.
    """
    def __init__(self, mol, basis, tmoleFormat=False): 
        """
        Initialize a Basis object for storing basis function properties.
        
        Args:  
            mol: Mol object containing molecular information  
            basis: Dictionary specifying the basis set to be used for each atom, or None to use mol.basis  
            tmoleFormat (bool): Whether to use TURBOMOLE format ordering for basis functions  
            
        Returns:
            None
            
        Raises:
            None - prints error message if mol is None
        """
        if mol is None:
            print('Error: Please provide a Mol object.')
            return None
        #Basis set name
        #If no basis set name is provided specifically,
        #then use the basis set name of the mol.
        #'basis' is a dictionary specifying the basis set to be used for a particular atom
        if basis is None:
            self.basis = mol.basis
        else:
            self.basis = basis
        
        
        self.basisSet = ''
        """Basis set name or label.
        Type: str
        """
        
        self.nao = 0 
        """Number of basis functions (atomic orbitals).
        Type: int
        """

        self.nshells = 0
        """Number of shells.
        Type: int
        """
        
        self.totalnprims = 0
        """Total number of primitive Gaussian functions.
        Type: int
        """
        
        self.nprims = []
        """Number of primitives in each shell.
        Type: List[int]
        """

        self.prim_coeffs = []
        """Primitive contraction coefficients.
        Type: List[float]
        """
        #Exponents of primitives
        self.prim_expnts = []
        """Exponents of primitives.
        Type: List[float]
        """
        #Coords of primitives
        self.prim_coords = []
        """Coordinates of each primitive function.
        Type: List[List[float]]
        """
        #Normalization factors of primitives
        self.prim_norms = []
        """Normalization constants of primitives.
        Type: List[float]
        """
        # This list contains the indices of atoms (first = #0) that correspond to a particular primitive function
        self.prim_atoms = [] #(This list should be of the size of the number of self.totalnprims)
        """Atom index corresponding to each primitive.
        Type: List[int]
        """
        # A list of tuples (2D list) that gives the number of primitives belonging to each atomic index
        self.nprims_atoms = [] # Size is the same as that of no. of atoms. Looks like this: [(0, 12), (1, 5), (2, 5)]
        """Number of primitives per atom as (atom_index, nprims).
        Type: List[Tuple[int, int]]
        """
        # This list contains the indices of shells (first = #0) that correspond to a particular primitive function
        self.prim_shells = [] #(This list should be of the size of the number of self.totalnprims)
        """Shell index corresponding to each primitive.
        Type: List[int]
        """
        # A list of tuples (2D list) that gives the number of primitives belonging to each shell index
        self.nprims_shells = [] # Size is the same as that of no. of shells. Looks like this: [(0, 12), (1, 5), (2, 5)]
        """Number of primitives per shell as (shell_index, nprims).
        Type: List[Tuple[int, int]]
        """
        # A special 2d list that contains the angular momentum of shell in the first index and the no. of primitives corresponding to the shell as second index
        self.nprims_shell_l_list = [] # This will be of the same size as the no. of shells
        """Angular momentum and primitive count per shell as (l, nprims).
        Type: List[Tuple[int, int]]
        """
        # A list of size natoms, that contains the values of the largest primitive exponents for all the atoms
        self.alpha_max = []
        """Maximum exponent among primitives for each atom.
        Type: List[float]
        """
        # A list of size natoms, that contains the list of the smallest primitive exponents for each shell for all the atoms
        self.alpha_min = []
        """Minimum exponent of primitives per shell for each atom.
        Type: List[List[float]]
        """
        #Number of contractions
        self.ncontrs = 0
        """Total number of contracted basis functions.
        Type: int
        """
        #Normalization factors for contractions
        self.contrs_norm = []
        """Normalization factors for contracted basis functions.
        Type: List[float]
        """
        #Degeneracy of shells
        self.shell_degen = []
        """Degeneracy of each shell.
        Type: List[int]
        """
        #Shells
        self.shellsLabel = []
        """Label of each shell (e.g., 'S', 'P', etc.).
        Type: List[str]
        """
        self.shells = []
        """Shell index (0: s, 1: p, 2:d).
        Type: List[int]
        """
        self.shell_coords = []
        """Coordinates of each shell center.
        Type: List[List[float]]
        """
        #---------------------------------
        #Information for basis function
        #---------------------------------
        #Coordinates of basis functions (This list should be of the size of the number of AOs/BFs)
        self.bfs_coords = []
        """Coordinates of basis functions (same as their parent shell).
        Type: List[List[float]]
        """
        # Number of primitives corresponding to each basis function (This list should be of the size of the number of AOs/BFs)
        self.bfs_nprim = []
        """Number of primitives corresponding to each basis function.
        Type: List[int]
        """
        # Radius cutoff for each basis function (This list should be of the size of the number of AOs/BFs)
        # This is used to accelerate the evaluation of AOs on grid points, as those gridpoints farther than the radius cutoff
        # of the basis function can be discarded for calcualtion. 
        self.bfs_radius_cutoff = []
        """Radius cutoff for each basis function to speed up AO evaluation.
        Type: List[float]
        """
        #A list of the size of number of atomic orbitals (BFs), 
        #that contains lists of primitive contraction coefficients/exponents 
        #corresponding to every basis function
        self.bfs_expnts = []
        """Primitive exponents used in each basis function.
        Type: List[List[float]]
        """

        self.bfs_coeffs = []
        """Primitive coefficients for each basis function.
        Type: List[List[float]]
        """
        # A list of the size of number of AOs (BFs)
        # that contains the lists of normalization factors for 
        # every primitive corresponding to the basis functions
        self.bfs_prim_norms = []
        """Primitive normalization constants per basis function.
        Type: List[List[float]]
        """
        # Normalization factor for the contraction of primitives making up a BF (This list should be of the size of the number of AOs/BFs)
        self.bfs_contr_prim_norms = []
        """Normalization for the contraction of primitives forming a BF.
        Type: List[float]
        """
        # l,m,n indices of BFs (This list should be of the size of the number of AOs/BFs)
        self.bfs_lmn = []
        """Cartesian angular momentum indices (l, m, n) per basis function.
        Type: List[Tuple[int, int, int]]
        """
        # Angular momentum of a basis function (This list should be of the size of the number of AOs/BFs)
        self.bfs_lm = []
        """Angular momentum `l` for each basis function.
        Type: List[int]
        """
        # Label of a basis function (This list should be of the size of the number of AOs/BFs)
        self.bfs_label = []
        """Label for each basis function.
        Type: List[str]
        """
        # Number of BFs in a shell
        self.bfs_nbfshell = []
        """Number of BFs in the shell of each basis function.
        Type: List[int]
        """
        # Shell index of each bf
        self.bfs_shell_index =[]
        """Shell index for each basis function.
        Type: List[int]
        """
        # BF offset index of each shell (This should be of the size of the number of shells)
        self.shell_bfs_offset =[]
        """Offset index of the first basis function in each shell.
        Type: List[int]
        """
        # Total number of basis functions (BFs) also called Atomic orbitals (AOs)
        self.bfs_nao = []
        """Index of each basis function (redundant with `nao`).
        Type: List[int]
        """
        # This list contains the indices of atoms (first = #0) that correspond to a particular bf
        self.bfs_atoms = [] #(This list should be of the size of the number of AOs/BFs)
        """Atom index corresponding to each basis function.
        Type: List[int]
        """
        #---------------------------------
        #If all data was parsed successfully.
        self.success = False
        """Indicates if basis parsing was successful.
        Type: bool
        """
        #Convert the keys of basis dictionary to lower case to avoid ambiguity
        self.basisSet = self.createCompleteBasisSet(mol, basis)
        self.readBasisFunctionInfo(mol, self.basisSet)
        self.totalnprims = sum(self.nprims)
        self.nshells = len(self.nprims)
        # Now lets create information about the basis functions
        # A basis function is a combination of gaussian primitives,
        # so we need 
        # number of primitives for a given bf, 
        # the exponents and contraction coefficients of the primitives,
        # the l,m,n triplet information, 
        # the angular momentum l+m+n,
        # coords, 
        # normalization factors of primitives,
        # normalization factor for the contraction.
        #---------------------------------------
        offset = 0
        ishell = 0
        # Loop through all the shells
        for i in range(len(self.nprims)):
            # Number of primitives in a shell
            numprim = self.nprims[i]
            # Angular momentum of a shell
            lm = self.shells[i] - 1
            # no of bfs in this shell 
            nbf_shell = Data.shell_degen[lm]
            self.bfs_nbfshell.append(nbf_shell)
            # Assign the lmn triplets and other information corresponding to basis functions
            # by looping through the degeneracy of the shells.
            # Example: If the degeneracy is 1 then we loop through [0,1) that is we look
            # for the 0th element in the Data.shell_lmn dictionary as that corresponds to the 's' lmn triplet.
            # Similarly if the denegeracy was 3 as for 'p' shell, then we loop from [1,3] and get the 
            # lmn triplets corresponding to 'p' shell and so on. 
            offset_shell_labels = Data.shell_lmn_offset[lm]
            for j in range(offset_shell_labels, nbf_shell + offset_shell_labels):
                #Assign the label to the basis function
                if tmoleFormat:
                    label = list(Data.shell_lmn_tmole)[j]
                else:
                    label = list(Data.shell_lmn)[j]
                self.bfs_label.append(label)
                # Assign l,m,n triplet for a basis function
                if tmoleFormat:
                    lmn = Data.shell_lmn_tmole[label]
                else:
                    lmn = Data.shell_lmn[label]
                self.bfs_lmn.append(lmn)
                # Assign the number of primitives in a basis function 
                self.bfs_nprim.append(numprim)
                # Assign the the total angular momentum of a basis function
                self.bfs_lm.append(sum(lmn))
                # Assign the exponents and contraction coefficients 
                expnts = self.prim_expnts[offset : offset+numprim ]
                self.bfs_expnts.append(expnts)
                coeffs = self.prim_coeffs[offset : offset+numprim ]
                self.bfs_coeffs.append(coeffs)
                # Assign Radius Cutoffs
                radius_cutoff = 0.0
                # Loop over primitives
                eeta_log = np.log(1.0E-9)
                for iprim_ in range(len(expnts)):
                    radius_cutoff = np.maximum(radius_cutoff, np.sqrt(1.0/expnts[iprim_]*(np.log(expnts[iprim_])/2.0 - eeta_log)))
                self.bfs_radius_cutoff.append(radius_cutoff)
                # Assign the normalization factors for all the primitives corresponding to a BF
                norms = []
                for k in range(offset,offset+numprim):
                    norms.append(self.normalizationFactor(self.prim_expnts[k],lmn[0],lmn[1],lmn[2]))
                self.bfs_prim_norms.append(norms)
                self.bfs_contr_prim_norms.append(self.normalizationFactorContraction(expnts, coeffs, norms, lmn[0],lmn[1],lmn[2], lm))
                # Assign coordinates to the basis functions
                self.bfs_coords.append(self.shell_coords[i])
                # Assign shell index to the basis functions
                self.bfs_shell_index.append(ishell)
                
            ishell += 1    
            offset = offset + numprim
        self.bfs_nao = len(self.bfs_label)   

        ##### The following was mainly developed to calculated alpha_min and alpha_max for numgrid

        # Calculate the number of primitives corresponding to each atom 
        cnt = Counter(self.prim_atoms) # Gives a Counter object similar to a dictionary
        # Convert the Counter object to a list
        self.nprims_atoms = list(cnt.items()) #a list of tuples, each containing a value and its count in the list
        ### Start processing to calculate alpha_min and alpha_max for grid generation
        ## Calculate alpha_max (https://github.com/dftlibs/numgrid)
        self.alpha_max = []
        offset = 0
        for iat in range(mol.natoms):
            alpha_max = max(self.prim_expnts[offset : offset + self.nprims_atoms[iat][1]])
            self.alpha_max.append(alpha_max)
            offset = offset + self.nprims_atoms[iat][1]

        # Calculate the number of primitives corresponding to each shell 
        cnt = Counter(self.prim_shells) # Gives a Counter object similar to a dictionary
        # Convert the Counter object to a list
        self.nprims_shells = list(cnt.items()) #a list of tuples, each containing a value and its count in the list
        # Make a list that is pretty much the same as nprims_shells, however, it contains the shell angular momentum in the first index
        for ishell in range(self.nshells):
            self.nprims_shell_l_list.append((self.shells[ishell], self.nprims_shells[ishell][1]))
        # Create self.nprims_angmom_list
        # self.calc_nprim_for_each_angular_momentum_l()
        ## Calculate alpha_min
        ishell_offset = 0
        iprim_offset = 0
        for iat in range(mol.natoms):
            alpha_min_dict_iatom = {}
            nshell_atom = 0
            sum_prim = 0
            for ishell in range(ishell_offset, self.nshells):
                nshell_atom = nshell_atom + 1
                sum_prim = sum_prim + self.nprims_shells[ishell][1]
                if sum_prim==self.nprims_atoms[iat][1]:
                    break
            subset_nprims_shell_l_list = self.nprims_shell_l_list[ishell_offset : ishell_offset + nshell_atom]
            ishell_offset = ishell_offset + nshell_atom
            # Create self.nprims_angmom_list
            subset_nprims_angmom_list = self.calc_nprim_for_each_angular_momentum_l(subset_nprims_shell_l_list)
            # alpha_min
            for i in range(len(subset_nprims_angmom_list)):
                alpha_min_dict_iatom[subset_nprims_angmom_list[i][0] - 1] = min(self.prim_expnts[iprim_offset : iprim_offset + subset_nprims_angmom_list[i][1]])
                iprim_offset = iprim_offset + subset_nprims_angmom_list[i][1]

            self.alpha_min.append(alpha_min_dict_iatom)

        # Calculate the BF offset index for a given shell
        self.shell_bfs_offset = np.cumsum(self.bfs_nbfshell) - self.bfs_nbfshell    



    # NORMALIZATION STUFF
    #Normalization factor for a primitive gaussian
    def normalizationFactor(self, alpha, l, m, n):
        """
        Calculate the normalization factor for a primitive Gaussian function.
        
        Args:
            alpha (float): Exponent of the Gaussian primitive
            l (int): Angular momentum quantum number in x-direction
            m (int): Angular momentum quantum number in y-direction  
            n (int): Angular momentum quantum number in z-direction
            
        Returns:
            float: Normalization factor for the primitive Gaussian
        """
        return np.sqrt((2*alpha/np.pi)**(3/2)*(4*alpha)**(l+m+n)/(factorial2(2*l-1)*factorial2(2*m-1)*factorial2(2*n-1)))

    #Normalization factor for a contraction of gaussians
    def normalizationFactorContraction(self, alphas, coeffs, norms, l, m, n, lm):
        """
        Calculate the normalization factor for a contraction of Gaussian primitives.
        
        Args:
            alphas (list): List of exponents for the primitive Gaussians
            coeffs (list): List of contraction coefficients
            norms (list): List of normalization factors for individual primitives
            l (int): Angular momentum quantum number in x-direction
            m (int): Angular momentum quantum number in y-direction
            n (int): Angular momentum quantum number in z-direction
            lm (int): Total angular momentum (l+m+n)
            
        Returns:
            float: Normalization factor for the contracted Gaussian
        """

        temp = np.pi**(3/2)*factorial2(2*l-1)*factorial2(2*m-1)*factorial2(2*n-1)/(2**lm)
        sum = 0.0
        for i in range(len(alphas)):
            for j in range(len(alphas)):
                sum = sum + (coeffs[i]*coeffs[j]*norms[i]*norms[j]/((alphas[i]+alphas[j])**(lm+3/2)))
        factor = np.power(temp*sum,-1/2)
        return factor


    def readBasisSetFromFile(key, filename):
        """
        Read the basis set block corresponding to a particular atom from a TURBOMOLE format file.
        
        Args:
            key (str): Atomic species symbol to search for
            filename (str): Path to the basis set file
            
        Returns:
            str or bool: Basis set string for the atom, or False if not found
        """
        #This functions returns the basis set block corresponding to a particular atom (key)
        #The file to be read from is assumed to follow TURBOMOLE's format
        basisString = ''
        lookfor = ''
        file = open(filename, 'r')
        fileContentsInString=file.read()
        file.close()
        lines = fileContentsInString.splitlines()
        pattern = re.compile('\*\n(.*)\n\*')
        result = pattern.findall(fileContentsInString)
        if len(result)==0:
            print('The basis set corresponding to atom',key, ' was not found in ',filename)
            return False
        for res in result:
            if res.split()[0].lower()==key.lower():
                lookfor = res
                basisString = '\n*\n'+lookfor+'\n*'

        currentLineNo = 0
        startReading = False
        for line in lines:
            line = line.strip()
            if line==lookfor:
                startReading = True
                currentLineNo = 1
            if startReading and currentLineNo>=3:
                if line.strip()=='*':
                    break
                else:
                    basisString = basisString + '\n'+line
            currentLineNo = currentLineNo + 1
        basisString = basisString + '\n*'
        return basisString
        
        

    
      
    #Loads the complete basis set as a string for a given mol/atom and a standard basis set available in the 
    #CrysX library or same directory as the python script.
    def load( atom=None, mol=None, basis_name=None):
        """
        Load the complete basis set as a string for a given atom/molecule from the CrysX library.
        
        Args:
            atom (str, optional): Atomic species symbol
            mol (Mol, optional): Mol object containing molecular information
            basis_name (str, optional): Name of the basis set to load
            
        Returns:
            str: Complete basis set string in TURBOMOLE format
        """
        # The CrysX library contains basis sets in the TURBOMOLE format.
        # The basis sets are downloaded from https://www.basissetexchange.org
        # BSSE Github: https://github.com/MolSSI-BSE/basis_set_exchange
        # License of BSSE: BSD 3

        # We (CrysX) have saved the basis sets in the 'BasisSets' directory.
        # The basis set name should be in lower-case and followed by '.version' and '.tm' extension.
        # Ex: def2-tzvp.1.tm
        # We can't expect the user to enter the basis set name with such an extension 
        # or in lower or upper case. So we should process this input name.
        basis_name = basis_name.strip()
        #The basis set for atom / mol
        basisSet = basis_name+'\n'
        basis_name = basis_name.replace(" ", "")
        basis_name = basis_name.lower()
        basis_name = basis_name+'.1.tm'
        # Here we need to create the path name to the BasisSets directory supplied with crysx
        # Since, the path should be with respect to hte module's location and not the working directory we can follow
        # this link for various methods: https://stackoverflow.com/questions/4060221/how-to-reliably-open-a-file-in-the-same-directory-as-a-python-script
        # Method 1: __file__
        # Method 2: sys.argv[0]
        # Method 3: sys.path[0] (probably not relevant to our purpose)
        # Currently we use method 1
        temp = __file__
        temp = temp.replace('Basis.py', '')
        basis_name = temp + 'BasisSets/'+basis_name
        #basis_name = 'BasisSets/'+basis_name
        
        if basis_name is None:
            print('Error: A basis set name is needed!')
            return basisSet
        #If mol is provided then load the same basis set to all atoms
        if not mol is None:
            for i in range(mol.natoms):
                basisSet = basisSet + Basis.readBasisSetFromFile( mol.atomicSpecies[i], basis_name)
        elif atom is not None:
            basisSet = Basis.readBasisSetFromFile(atom, basis_name)
        basisSet =  basisSet.replace('D+', 'E+')
        basisSet =  basisSet.replace('D-', 'E-')
        return basisSet

    #Loads the complete basis set as a string for a given mol/atom from a file
    #specified by the user.
    def loadfromfile( atom=None, mol=None, basis_name=None):
        """
        Load the complete basis set as a string for a given atom/molecule from a user-specified file.
        
        Args:
            atom (str, optional): Atomic species symbol
            mol (Mol, optional): Mol object containing molecular information  
            basis_name (str, optional): Path to the basis set file
            
        Returns:
            str: Complete basis set string in TURBOMOLE format
        """
        #The basis set for atom / mol
        basisSet = ''
        if basis_name is None:
            print('Error: A basis set name is needed!')
            return basisSet
        #If mol is provided then load the same basis set to all atoms
        if not mol is None:
            for i in range(mol.natoms):
                basisSet = basisSet + Basis.readBasisSetFromFile( mol.atomicSpecies[i], basis_name)
        elif atom is not None:
            basisSet = Basis.readBasisSetFromFile(atom, basis_name)
        basisSet =  basisSet.replace('D+', 'E+')
        basisSet =  basisSet.replace('D-', 'E-')
        return basisSet

    #Creates the complete basis set for a given molecule along with basis dictionary
    def createCompleteBasisSet(self, mol, basis):
        """
        Create the complete basis set string for a given molecule using the basis dictionary.

        Args:
            mol: Mol object containing molecular information
            basis (dict): Dictionary mapping atom types to basis set strings
            
        Returns:
            str: Complete basis set string for all atoms in the molecule
        """
        totBasisSet = ''
        if basis is None:
            print('Error: Please provide a basis dictionary.')
            return totBasisSet
        else:
            if 'all' in basis:
                return basis['all']
            else:
                #TODO Change this loop to only read the basis set for unique atoms
                #TODO otherwise the basis set contains repeat basis sets for repeated atoms
                #TODO The current form may(it doesn't yet) cause problem when creating basis function information
                #TODO For even further generality let users label atoms, so that different basis may be used for same atoms with different labels
                for i in range(mol.natoms):
                    if mol.atomicSpecies[i].lower() in basis:
                        totBasisSet = totBasisSet+ basis[mol.atomicSpecies[i].lower()]+'\n'
                totBasisSet =  totBasisSet.replace('D+', 'E+')
                totBasisSet =  totBasisSet.replace('D-', 'E-')
                return totBasisSet
        totBasisSet =  totBasisSet.replace('D+', 'E+')
        totBasisSet =  totBasisSet.replace('D-', 'E-')
        return totBasisSet


    def cart2sph(l, ordering='pyscf'):
        """
        Get the transformation matrix from Cartesian to real spherical harmonics for a given angular momentum.
        
        Args:
            l (int): Angular momentum quantum number
            ordering (str): Ordering convention ('pyscf' or other)
            
        Returns:
            numpy.ndarray: Transformation matrix from Cartesian to spherical basis
        """
        # This routine is used to get the transformation matrix from cartesian to real spherical basis for a given l
        # This assumes that the basis functions are normalized.
        # TODO: Make it work for both normalized and unnormalized basis functions.

        # REFERENCE:
        # https://theochem.github.io/horton/2.0.1/tech_ref_gaussian_basis.html#collected-notes-on-gaussian-basis-sets
        # https://onlinelibrary.wiley.com/doi/epdf/10.1002/qua.560540202
        # PySCF ordering: https://github.com/pyscf/pyscf/issues/1023
        # https://en.wikipedia.org/wiki/Table_of_spherical_harmonics#Real_spherical_harmonics
        l0 = np.array([[1.0]])
        l1 = np.array([[0, 0, 1.0],
                      [1.0, 0, 0],
                      [0, 1.0, 0]])
        l1_pyscf = np.array([[1.0, 0, 0],
                      [0, 1.0, 0],
                      [0, 0, 1.0]])
        l2 =  np.array([[-0.5, 0, 0, -0.5, 0, 1.0],
                      [0, 0, 1.0, 0, 0, 0],
                      [0, 0, 0, 0, 1.0, 0],
                      [0.86602540378443864676, 0, 0, -0.86602540378443864676, 0, 0],
                      [0, 1.0, 0, 0, 0, 0],
                  ])
        # I don't remember what this was for (perhaps TURBOMOLE??)
        l2_alternative =np.array([[-0.5, 0, 0, -0.5, 0, 1],
                      [0, 0, 0.7071067811865475, 0, 0, 0],
                      [0, 0, 0, 0, 0.7071067811865475, 0],
                      [0.6123724356957945, 0, 0, -0.6123724356957945, 0, 0],
                      [0, 0.7071067811865475, 0, 0, 0, 0],
                  ])
        l2_pyscf =  np.array([[0, 1.0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 1.0, 0],
                      [-0.5, 0, 0, -0.5, 0, 1.0],
                      [0, 0, 1.0, 0, 0, 0],
                      [0.86602540378443864676, 0, 0, -0.86602540378443864676, 0, 0],
                  ])
        l3 = np.array([[0, 0, -0.67082039324993690892, 0, 0, 0, 0, -0.67082039324993690892, 0, 1.0],
                      [-0.61237243569579452455, 0, 0, -0.27386127875258305673, 0, 1.0954451150103322269, 0, 0, 0, 0],
                      [0, -0.27386127875258305673, 0, 0, 0, 0, -0.61237243569579452455, 0, 1.0954451150103322269, 0],
                      [0, 0, 0.86602540378443864676, 0, 0, 0, 0, -0.86602540378443864676, 0, 0],
                      [0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0],
                      [0.790569415042094833, 0, 0, -1.0606601717798212866, 0, 0, 0, 0, 0, 0],
                      [0, 1.0606601717798212866, 0, 0, 0, 0, -0.790569415042094833, 0, 0, 0],
                  ])
        l3_pyscf = np.array([[0, 1.0606601717798212866, 0, 0, 0, 0, -0.790569415042094833, 0, 0, 0],
                      [0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0],
                      [0, -0.27386127875258305673, 0, 0, 0, 0, -0.61237243569579452455, 0, 1.0954451150103322269, 0],
                      [0, 0, -0.67082039324993690892, 0, 0, 0, 0, -0.67082039324993690892, 0, 1.0],
                      [-0.61237243569579452455, 0, 0, -0.27386127875258305673, 0, 1.0954451150103322269, 0, 0, 0, 0],
                      [0, 0, 0.86602540378443864676, 0, 0, 0, 0, -0.86602540378443864676, 0, 0],
                      [0.790569415042094833, 0, 0, -1.0606601717798212866, 0, 0, 0, 0, 0, 0],
                  ])
        l4 = np.array([[0.375, 0, 0, 0.21957751641341996535, 0, -0.87831006565367986142, 0, 0, 0, 0, 0.375, 0, -0.87831006565367986142, 0, 1.0],
                      [0, 0, -0.89642145700079522998, 0, 0, 0, 0, -0.40089186286863657703, 0, 1.19522860933439364, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, -0.40089186286863657703, 0, 0, 0, 0, 0, 0, -0.89642145700079522998, 0, 1.19522860933439364, 0],
                      [-0.5590169943749474241, 0, 0, 0, 0, 0.9819805060619657157, 0, 0, 0, 0, 0.5590169943749474241, 0, -0.9819805060619657157, 0, 0],
                      [0, -0.42257712736425828875, 0, 0, 0, 0, -0.42257712736425828875, 0, 1.1338934190276816816, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0.790569415042094833, 0, 0, 0, 0, -1.0606601717798212866, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 1.0606601717798212866, 0, 0, 0, 0, 0, 0, -0.790569415042094833, 0, 0, 0],
                      [0.73950997288745200532, 0, 0, -1.2990381056766579701, 0, 0, 0, 0, 0, 0, 0.73950997288745200532, 0, 0, 0, 0],
                      [0, 1.1180339887498948482, 0, 0, 0, 0, -1.1180339887498948482, 0, 0, 0, 0, 0, 0, 0, 0],
                  ])
        l4_pyscf = np.array([[0, 1.1180339887498948482, 0, 0, 0, 0, -1.1180339887498948482, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 1.0606601717798212866, 0, 0, 0, 0, 0, 0, -0.790569415042094833, 0, 0, 0],
                      [0, -0.42257712736425828875, 0, 0, 0, 0, -0.42257712736425828875, 0, 1.1338934190276816816, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, -0.40089186286863657703, 0, 0, 0, 0, 0, 0, -0.89642145700079522998, 0, 1.19522860933439364, 0],
                      [0.375, 0, 0, 0.21957751641341996535, 0, -0.87831006565367986142, 0, 0, 0, 0, 0.375, 0, -0.87831006565367986142, 0, 1.0],
                      [0, 0, -0.89642145700079522998, 0, 0, 0, 0, -0.40089186286863657703, 0, 1.19522860933439364, 0, 0, 0, 0, 0],
                      [-0.5590169943749474241, 0, 0, 0, 0, 0.9819805060619657157, 0, 0, 0, 0, 0.5590169943749474241, 0, -0.9819805060619657157, 0, 0],
                      [0, 0, 0.790569415042094833, 0, 0, 0, 0, -1.0606601717798212866, 0, 0, 0, 0, 0, 0, 0],
                      [0.73950997288745200532, 0, 0, -1.2990381056766579701, 0, 0, 0, 0, 0, 0, 0.73950997288745200532, 0, 0, 0, 0],
                  ])
        l5 = np.array([[0, 0, 0.625, 0, 0, 0, 0, 0.36596252735569994226, 0, -1.0910894511799619063, 0, 0, 0, 0, 0, 0, 0.625, 0, -1.0910894511799619063, 0, 1.0],
                      [0.48412291827592711065, 0, 0, 0.21128856368212914438, 0, -1.2677313820927748663, 0, 0, 0, 0, 0.16137430609197570355, 0, -0.56694670951384084082, 0, 1.2909944487358056284, 0, 0, 0, 0, 0, 0],
                      [0, 0.16137430609197570355, 0, 0, 0, 0, 0.21128856368212914438, 0, -0.56694670951384084082, 0, 0, 0, 0, 0, 0, 0.48412291827592711065, 0, -1.2677313820927748663, 0, 1.2909944487358056284, 0],
                      [0, 0, -0.85391256382996653193, 0, 0, 0, 0, 0, 0, 1.1180339887498948482, 0, 0, 0, 0, 0, 0, 0.85391256382996653193, 0, -1.1180339887498948482, 0, 0],
                      [0, 0, 0, 0, -0.6454972243679028142, 0, 0, 0, 0, 0, 0, -0.6454972243679028142, 0, 1.2909944487358056284, 0, 0, 0, 0, 0, 0, 0],
                      [-0.52291251658379721749, 0, 0, 0.22821773229381921394, 0, 0.91287092917527685576, 0, 0, 0, 0, 0.52291251658379721749, 0, -1.2247448713915890491, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, -0.52291251658379721749, 0, 0, 0, 0, -0.22821773229381921394, 0, 1.2247448713915890491, 0, 0, 0, 0, 0, 0, 0.52291251658379721749, 0, -0.91287092917527685576, 0, 0, 0],
                      [0, 0, 0.73950997288745200532, 0, 0, 0, 0, -1.2990381056766579701, 0, 0, 0, 0, 0, 0, 0, 0, 0.73950997288745200532, 0, 0, 0, 0],
                      [0, 0, 0, 0, 1.1180339887498948482, 0, 0, 0, 0, 0, 0, -1.1180339887498948482, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0.7015607600201140098, 0, 0, -1.5309310892394863114, 0, 0, 0, 0, 0, 0, 1.169267933366856683, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 1.169267933366856683, 0, 0, 0, 0, -1.5309310892394863114, 0, 0, 0, 0, 0, 0, 0, 0, 0.7015607600201140098, 0, 0, 0, 0, 0],
                  ])
        l6 = np.array([[-0.3125, 0, 0, -0.16319780245846672329, 0, 0.97918681475080033975, 0, 0, 0, 0, -0.16319780245846672329, 0, 0.57335309036732873772, 0, -1.3055824196677337863, 0, 0, 0, 0, 0, 0, -0.3125, 0, 0.97918681475080033975, 0, -1.3055824196677337863, 0, 1.0],
                      [0, 0, 0.86356159963469679725, 0, 0, 0, 0, 0.37688918072220452831, 0, -1.6854996561581052156, 0, 0, 0, 0, 0, 0, 0.28785386654489893242, 0, -0.75377836144440905662, 0, 1.3816985594155148756, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0.28785386654489893242, 0, 0, 0, 0, 0, 0, 0.37688918072220452831, 0, -0.75377836144440905662, 0, 0, 0, 0, 0, 0, 0, 0, 0.86356159963469679725, 0, -1.6854996561581052156, 0, 1.3816985594155148756, 0],
                      [0.45285552331841995543, 0, 0, 0.078832027985861408788, 0, -1.2613124477737825406, 0, 0, 0, 0, -0.078832027985861408788, 0, 0, 0, 1.2613124477737825406, 0, 0, 0, 0, 0, 0, -0.45285552331841995543, 0, 1.2613124477737825406, 0, -1.2613124477737825406, 0, 0],
                      [0, 0.27308215547040717681, 0, 0, 0, 0, 0.26650089544451304287, 0, -0.95346258924559231545, 0, 0, 0, 0, 0, 0, 0.27308215547040717681, 0, -0.95346258924559231545, 0, 1.4564381625088382763, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, -0.81924646641122153043, 0, 0, 0, 0, 0.35754847096709711829, 0, 1.0660035817780521715, 0, 0, 0, 0, 0, 0, 0.81924646641122153043, 0, -1.4301938838683884732, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, -0.81924646641122153043, 0, 0, 0, 0, 0, 0, -0.35754847096709711829, 0, 1.4301938838683884732, 0, 0, 0, 0, 0, 0, 0, 0, 0.81924646641122153043, 0, -1.0660035817780521715, 0, 0, 0],
                      [-0.49607837082461073572, 0, 0, 0.43178079981734839863, 0, 0.86356159963469679725, 0, 0, 0, 0, 0.43178079981734839863, 0, -1.5169496905422946941, 0, 0, 0, 0, 0, 0, 0, 0, -0.49607837082461073572, 0, 0.86356159963469679725, 0, 0, 0, 0],
                      [0, -0.59829302641309923139, 0, 0, 0, 0, 0, 0, 1.3055824196677337863, 0, 0, 0, 0, 0, 0, 0.59829302641309923139, 0, -1.3055824196677337863, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0.7015607600201140098, 0, 0, 0, 0, -1.5309310892394863114, 0, 0, 0, 0, 0, 0, 0, 0, 1.169267933366856683, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 1.169267933366856683, 0, 0, 0, 0, 0, 0, -1.5309310892394863114, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.7015607600201140098, 0, 0, 0, 0, 0],
                      [0.67169328938139615748, 0, 0, -1.7539019000502850245, 0, 0, 0, 0, 0, 0, 1.7539019000502850245, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.67169328938139615748, 0, 0, 0, 0, 0, 0],
                      [0, 1.2151388809514737933, 0, 0, 0, 0, -1.9764235376052370825, 0, 0, 0, 0, 0, 0, 0, 0, 1.2151388809514737933, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  ])
        if ordering=='pyscf':
            c2s_l = [l0,l1_pyscf,l2_pyscf,l3_pyscf,l4_pyscf,l5,l6] # PYSCF ordering of SAOs
        else:
            c2s_l = [l0,l1,l2,l3,l4,l5,l6] # HORTON ordering of SAOs
        return c2s_l[l]

    def cart2sph_basis(self):
        """
        Return the complete Cartesian to spherical harmonic basis transformation matrix for all shells.
        
        Returns:
            matrix: Block diagonal transformation matrix for the entire basis set
        """
        # Returns the complete Cartesian to Spherical (real) basis transformation matrix
        cart2sph = []
        for i in range(self.nshells):
            l = self.shells[i]-1
            cart2sph.append(Basis.cart2sph(l))

        return scipy.linalg.block_diag(*cart2sph)
    
    def sph2cart_basis(self):
        """
        Return the complete spherical to Cartesian basis transformation matrix for all shells.

        Returns:
            matrix: Block diagonal transformation matrix (pseudoinverse of cart2sph)
        """
        # Returns the complete Cartesian to Spherical (real) basis transformation matrix
        sph2cart = []
        for i in range(self.nshells):
            l = self.shells[i]-1
            sph2cart_pseudo = np.linalg.pinv(Basis.cart2sph(l))
            sph2cart.append(sph2cart_pseudo)

        return scipy.linalg.block_diag(*sph2cart)


    # #Probably won't be used
    # def readBasisFunctionsfromFile(self, mol, basis_name):
    #     #This function will read and parse the basis sets for the basis functions and other properties.
    #     #Returns true if successful or false if failed.
    #     if mol is None:
    #         print('Error: Please provide a Mol object.')
    #         return False
    #     if basis_name is None:
    #         print('Error: Please provide a basis set name dictionary for Mol atoms.')
    #         return False
    #     #If all atoms are assigned the same basis set
    #     if 'all' in basis_name:
    #         for i in range(mol.natoms):
    #             self.basis_name_local[mol.atomicSpecies[i]] = basis_name['all']
    #     else:
    #         for i in range(mol.natoms):
    #             #If a particular basis set is assigned to an atom
    #             if mol.atomicSpecies[i] in basis_name:
    #                 self.basis_name_local[mol.atomicSpecies[i]] = basis_name[mol.atomicSpecies[i]]
    #             else:
    #             #Or assign the default basis set
    #                 self.basis_name_local[mol.atomicSpecies[i]] = basis_name_local['default']

    def readBasisFunctionInfo(self, mol, basisSet):
        """
        Read and parse the basis set information to populate basis function properties.
        
        Args:
            mol: Mol object containing molecular information
            basisSet (str): Complete basis set string in TURBOMOLE format
            
        Returns:
            None - populates internal data structures with basis function information
        """
        #TODO Add error checks,
        #TODO Like check if the basis set contains the needed atoms or not, and other checks that you can think of
        #TODO Also, add error messages explaining what is happening

        #Read the basis set and set the basis set information
        indx_shell = 0
        for i in range(mol.natoms):
            lookfor = ''
            lines = basisSet.splitlines()
            pattern = re.compile('\*\n(.*)\n\*')
            result = pattern.findall(basisSet)
            if len(result)==0:
                print('The basis set corresponding to atom ',mol.atomicSpecies[i], ' was not found in the provided basis set.')
            for res in result:
                if res.split()[0].lower()==mol.atomicSpecies[i].lower():
                    lookfor = res
            startReading = False
            #print(lines)
            currentLineNo = 0
            while currentLineNo<len(lines):
                line = lines[currentLineNo].strip()
                #print(line)
                if line==lookfor:
                    startReading = True
                    currentLineNo = currentLineNo + 2
                if startReading:
                    line = lines[currentLineNo].strip()
                    splitLine = line.split()
                    if '*' in line:
                        startReading = False
                        currentLineNo = currentLineNo + 1
                        break
                    #print(line)
                    #Check if the line looks like ' 3  s'
                    if isinstance(int(splitLine[0]), int) and isinstance(str(splitLine[1]), str):
                        #print('here')
                        #Read the shell information
                        #Read the number of primitives in the given shell  Ex: '3  s'
                        self.nprims.append(int(splitLine[0]))  #Read '3'
                        #Read the shell type Ex: '3  s'
                        self.shellsLabel.append(str(splitLine[1])) #Read 's'
                        #Get the angular momentum 'l' corresponding to the shell type.
                        #Ex: 's'->1, 'p'->2, 'd'->3, etc.
                        l = Data.shell_dict[splitLine[1]]
                        self.shells.append(l)
                        self.shell_degen.append(Data.shell_degen[l-1])
                        self.shell_coords.append(mol.coordsBohrs[i])
                        #create the information for basis functions
                        #for ibf in range(int(l*(l+1)/2.0)):
                        #    self.bfcoords.append(mol.coords[i])
                        #Run a loop over the number of primitives in the current shell
                        #and read the exponents and contraction coefficients
                        for k in range(int(splitLine[0])):
                            currentLineNo = currentLineNo + 1
                            line = lines[currentLineNo].strip()
                            splitLine = line.split()
                            if '*' in line:
                                startReading = False
                                currentLineNo = currentLineNo + 1
                                break
                            splitLine[0] = splitLine[0].replace('D+', 'E+')
                            splitLine[1] = splitLine[1].replace('D-', 'E-')
                            self.prim_expnts.append(float(splitLine[0]))
                            self.prim_coeffs.append(float(splitLine[1]))
                            self.prim_coords.append(mol.coordsBohrs[i])
                            self.prim_atoms.append(i)
                            self.prim_shells.append(indx_shell)
                            #self.prim_norms.append(float(normalizationFactor()))
                            #print(self.prim_coeffs, self.prim_expnts)
                            #print(currentLineNo)

                        indx_shell +=1
                        for ibf in range(int(l*(l+1)/2.0)):
                            self.bfs_atoms.append(i)

                currentLineNo = currentLineNo + 1

                        

        #Now that the reading of shells and their primitives is done
        
        #Run a loop over shells, and create information for 
        # basis functions
        #for i in range(shells

    def calc_nprim_for_each_angular_momentum_l(self, tuple_list):
        """
        Calculate the number of primitives for each angular momentum quantum number.
        
        Args:
            tuple_list (list): List of tuples containing (angular_momentum, num_primitives)
            
        Returns:
            list: List of tuples with summed primitives for each unique angular momentum
        """
        # Initialize the list of tuples
        # tuple_list = [(0,12),(0,5),(1,2)]
        # tuple_list = self.nprims_shells

        # Create a dictionary to store the sums of each common angular momentum
        sums_dict = {}

        # Loop through each tuple in the list
        for t in tuple_list:
            # Get the first element of the tuple
            key = t[0]

            # Check if the element is already in the dictionary
            if key in sums_dict:
                # If it is, add the second element of the tuple to the sum
                sums_dict[key] += t[1]
            else:
                # If it isn't, add the element to the dictionary with the value of the second element of the tuple
                sums_dict[key] = t[1]

        # Initialize the list of tuples to return
        result = []

        # Loop through each key-value pair in the dictionary
        for key, value in sums_dict.items():
            # Append a tuple with the key and value to the result list
            result.append((key, value))

        # self.nprims_angmom_list = result
        return result

                            
