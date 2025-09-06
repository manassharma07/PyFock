# Mol.py
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
from . import Data
from . import Basis
import numpy as np
#Class to store molecular properties
class Mol:
    """
    Class to store molecular properties and handle molecular structure operations.
    
    This class provides functionality for creating, manipulating, and exporting molecular
    structures. It supports reading from various coordinate file formats and automatically
    generates basis set information.
    
    Attributes:
        coordfile (str): Path to coordinate file
        atoms (list): List of atoms with coordinates
        charge (int): Molecular charge
        basis (Basis): Basis set object for the molecule
        nelectrons (int): Number of electrons
        natoms (int): Number of atoms
        coords (numpy.ndarray): Atomic coordinates in Angstroms
        coordsBohrs (numpy.ndarray): Atomic coordinates in Bohr units
        atomicSpecies (list): List of atomic symbols
        Zcharges (list): List of atomic numbers
        success (bool): Whether molecule was initialized successfully
        label (str): Molecular label/description
    """
    def __init__(self, atoms=None, coordfile=None, charge=0, basis=None):
        """
        Initialize a Mol object with atomic coordinates and properties.
        
        Args:
            atoms (list, optional): List of atoms in the format [[symbol, x, y, z], ...] 
                                   where symbol can be atomic symbol (str) or atomic number (int),
                                   and x, y, z are coordinates in Angstroms.
            coordfile (str, optional): Path to coordinate file (.xyz format supported).
            charge (int, optional): Molecular charge. Defaults to 0.
            basis (dict, optional): Basis set specification. If None, defaults to STO-3G 
                                   for all atoms.
        
        Raises:
            ValueError: If neither atoms nor coordfile is provided.
            TypeError: If coordfile is not a string.
        
        Note:
            If both atoms and coordfile are provided, atoms takes precedence and 
            coordfile is ignored.
        """ 
        #If no atoms or coordfile is specified.
        if atoms is None and coordfile is None:
            print('Error: No atoms and their coords specified.\nPlease either specify .xyz/.mol/coord file or enter atoms manually.')
            print('See manual for supported file formats.')
        #If both atoms and coordfile are specified then atoms are considered and the coord file is not considered.
        elif not coordfile is None and not atoms is None:
            print('Remark: User provided both atoms as well as the coordfile.\nOnly atoms are being used.')
        elif coordfile is not None and atoms is None:
            if isinstance(coordfile, str):
                atoms = self.readCoordsFile(coordfile)
            else:
                print('Error: The coordfile argument should be a string with the name of the coord file.')

        
        #Coord file .xyz/.mol/TURBOMOLE coord file/QE input file/Orca input file
        self.coordfile = coordfile
        """str or None: Path to the coordinate file used to initialize the molecule.
        Contains the filename of the original coordinate file (.xyz, .mol, TURBOMOLE coord, etc.) 
        if the molecule was created from a file, or None if created directly from atoms list."""
        #Atoms and their coordinates
        self.atoms = atoms
        """list: List of atoms with their coordinates in the format [[symbol, x, y, z], ...] where:
        - symbol: Atomic symbol (str) or atomic number (int)
        - x, y, z: Cartesian coordinates in Angstroms (float)
        This is the raw input data used to construct the molecule."""
        #Molecular charge
        self.charge = charge
        """int: Net charge of the molecule in elementary charge units.
        Positive values indicate cationic species, negative values indicate anionic species, 
        and 0 indicates a neutral molecule. Used to calculate the total number of electrons."""
        #The basis object for the Mol object
        self.basis = None
        """Basis or None: Basis set object containing all basis function information for quantum 
        chemical calculations. Initialized as None and populated with a Basis instance that 
        handles basis set assignments for each atom in the molecule."""

        self.nelectrons = 0
        """int: Total number of electrons in the molecule.
        Calculated as the sum of all atomic numbers (nuclear charges) adjusted by the molecular charge:
        nelectrons = Σ(Z_i) - charge, where Z_i is the atomic number of atom i."""
        self.natoms = 0
        """int: Total number of atoms in the molecule.
        Incremented during atom validation and used for array dimensioning and iteration 
        over atomic properties."""
        self.coords = []
        """numpy.ndarray: Atomic coordinates in Angstroms with shape (natoms, 3).
        Each row contains the [x, y, z] coordinates of one atom. Initialized as empty list 
        and populated during atom validation to become a numpy array."""
        self.coordsBohrs = []
        """numpy.ndarray: Atomic coordinates converted to Bohr units with shape (natoms, 3).
        Calculated from self.coords using the conversion factor Data.Angs2BohrFactor. 
        Used for quantum chemical calculations requiring atomic units of length."""
        self.atomicSpecies = []
        """list of str: List of atomic symbols as strings (e.g., ['H', 'C', 'N', 'O']).
        Ordered the same as atoms appear in the molecule. Length equals natoms. 
        Used for identifying atom types and output formatting."""

        self.Zcharges = []
        """list of int: List of atomic numbers (nuclear charges) for each atom in the molecule.
        Corresponds to the number of protons in each atomic nucleus. Used for calculating 
        electronic properties and nuclear contributions to various molecular properties."""
        self.success = False
        """bool: Flag indicating whether the molecule was successfully initialized and validated.
        Set to True if all atoms have valid symbols/atomic numbers and properly formatted 
        coordinates, False otherwise. Used to prevent operations on invalid molecular data."""

        self.label = 'Generic mol'
        """str: Descriptive label or name for the molecule.
        Defaults to 'Generic mol' but can be customized. Used in output files and for 
        identification purposes when exporting molecular data to various file formats."""
        #Validate if the atoms attribute are provided correctly
        self.success = self.validateAtoms(atoms)
        if self.success:
            self.nelectrons = self.nelectrons+charge
            self.coordsBohrs = self.coords*Data.Angs2BohrFactor
            #Basis set names for the atoms.
            #Either just one basis function can be specified for all atoms
            #Or one can specify a particular basis set for a particular atom.
            #'basis' is a dictionary specifying the basis set to be used for a particular atom
            if basis is None:
                #Default basis set is sto-3g for all atoms
                self.basis = {'all':Basis.load(mol=self, basis_name='sto-3g')}#{'all','sto-3g'}
                self.basis = Basis(self, self.basis)
            else:
                self.basis = Basis(self, basis)
        

        
        

    def validateAtoms(self, atoms):
        """
        Validate atomic symbols and coordinates provided in the atoms list.
        
        This method checks that atomic symbols are valid (either as strings matching
        known element symbols or as integers representing atomic numbers), and that
        coordinates are properly formatted as numerical values.
        
        Args:
            atoms (list): List of atoms in format [[symbol, x, y, z], ...] where
                         symbol can be str (element symbol) or int (atomic number),
                         and x, y, z are numerical coordinates.
        
        Returns:
            bool: True if all atoms are valid, False otherwise.
        
        Side Effects:
            Updates the following instance attributes:
            - atomicSpecies: List of atomic symbols
            - Zcharges: List of atomic numbers
            - nelectrons: Total number of electrons
            - natoms: Number of atoms
            - coords: Numpy array of coordinates
        
        Raises:
            Prints error messages for invalid atomic symbols, coordinates, or formatting.
        """
        #Validate atomic symbols
        for i in range(len(atoms)):
            if isinstance(atoms[i][0], str):
                elementSymbols = [x.lower() for x in Data.elementSymbols]
                if atoms[i][0].lower() in elementSymbols:
                    self.atomicSpecies.append(atoms[i][0])
                    Z = elementSymbols.index(atoms[i][0].lower())
                    self.Zcharges.append(Z)
                    self.nelectrons = self.nelectrons + Z 
                    self.natoms = self.natoms + 1
                else:
                    print('Error: Unknown atomic symbols found')
                    return False
            elif isinstance(atoms[i][0], int):
                if atoms[i][0]<=118 and atoms[i][0]>=0:
                    self.atomicSpecies.append(Data.elementSymbols[atoms[i][0]])
                    Z = atoms[i][0]
                    self.Zcharges.append(Z)
                    self.nelectrons = self.nelectrons + Z 
                    self.natoms = self.natoms + 1
                else:
                    print('Error: Found atomic number out of the valid range')
                    return False
            else:
                print('Error: Found something other than string or integer for atomic symbols/numbers')
                return False
        self.coords = np.zeros([self.natoms,3])
        #Validate atomic coords
        for i in range(len(atoms)):
            if len(atoms[i])==4:
                #TODO: the following condition doesn't work
                #FIXME
                if ( isinstance(coordi, float) for coordi in atoms[i][1:4] ) or ( isinstance(coordi, int) for coordi in atoms[i][1:4]):
                    self.coords[i] = atoms[i][1:4]
                else:
                    print('Error: Coordinates of atoms should be floats/integers.')
                    return False

            else:
                print('Error: Illegal definition of coords in atoms. There should be all three x,y,z coordinates.\nSome seem to be missing.')
                return False
        
        if self.natoms==self.coords.shape[0]:
            return True
        else:
            print('Error: Some definitions of atomic symbols/coords were illegal')
            

    def get_center_of_charge(self, units='angs'):
        """
        Calculate the center of charge (weighted by atomic numbers) of the molecule.
        
        The center of charge is computed as the weighted average of atomic positions,
        where the weights are the atomic numbers (nuclear charges) of each atom.
        
        Returns:
            numpy.ndarray or None: 3D coordinates of the center of charge in Angstroms.
                                  Returns None if the molecule was not successfully initialized.
        
        Formula:
            center_of_charge = Σ(Z_i * r_i) / Σ(Z_i)
            where Z_i is the atomic number and r_i is the position of atom i.
        """
        if not self.success:
            print("Error: Mol object is not successfully initialized.")
            return None

        # Use NumPy for efficient calculations
        total_charge = np.sum(self.Zcharges)
        if units=='angs':
            coc = np.dot(self.Zcharges, self.coords) / total_charge
        else:
            coc = np.dot(self.Zcharges, self.coordsBohrs) / total_charge

        return coc
        # # Use NumPy for efficient calculations
        # total_mass = np.sum(self.Zcharges)
        # com_x = np.sum(self.Zcharges * self.coords[:, 0]) / total_mass
        # com_y = np.sum(self.Zcharges * self.coords[:, 1]) / total_mass
        # com_z = np.sum(self.Zcharges * self.coords[:, 2]) / total_mass

        # com = np.array([com_x, com_y, com_z])
        # return com
    
    def get_nuc_dip_moment(self):
        """
        Calculate the nuclear contribution to the molecular dipole moment.
        
        The nuclear dipole moment is computed as the sum of nuclear charges
        multiplied by their positions in atomic units (Bohr).
        
        Returns:
            numpy.ndarray: 3D vector representing the nuclear dipole moment
                          in atomic units (e⋅a₀), where each component corresponds
                          to x, y, z directions.
        
        Formula:
            μ_nuc = Σ(Z_i * r_i)
            where Z_i is the nuclear charge and r_i is the position in Bohr units.
        """
        charges = self.Zcharges
        coords  = self.coordsBohrs 
        nuc_dip = np.einsum('i,ix->x', charges, coords)
        return nuc_dip

    def get_elec_dip_moment(self, dipole_moment_matrix, dmat):
        """
        Calculate the electronic contribution to the molecular dipole moment.
        
        The electronic dipole moment is computed using the dipole moment integrals
        and the density matrix from quantum chemical calculations.
        
        Args:
            dipole_moment_matrix (numpy.ndarray): 3D array of dipole moment integrals
                                                 with shape (3, nbasis, nbasis), where
                                                 the first dimension corresponds to x, y, z.
            dmat (numpy.ndarray): Density matrix with shape (nbasis, nbasis).
        
        Returns:
            numpy.ndarray: 3D vector representing the electronic dipole moment
                          in atomic units (e⋅a₀), where each component corresponds
                          to x, y, z directions.
        
        Formula:
            μ_elec = Σ_μν P_μν ⟨μ|r|ν⟩
            where P_μν is the density matrix and ⟨μ|r|ν⟩ are dipole integrals.
        """
        elec_dip = np.einsum('xij,ji->x', dipole_moment_matrix, dmat).real
        return elec_dip
    
    def get_dipole_moment(self, dipole_moment_matrix, dmat):
        """
        Calculate the total molecular dipole moment.
        
        The total dipole moment is computed as the difference between nuclear
        and electronic contributions: μ_total = μ_nuclear - μ_electronic
        
        Args:
            dipole_moment_matrix (numpy.ndarray): 3D array of dipole moment integrals
                                                 with shape (3, nbasis, nbasis).
            dmat (numpy.ndarray): Density matrix from quantum chemical calculation
                                 with shape (nbasis, nbasis).
        
        Returns:
            numpy.ndarray: 3D vector representing the total molecular dipole moment
                          in atomic units (e⋅a₀). Each component corresponds to
                          x, y, z directions.
        
        Note:
            The sign convention follows: μ_total = μ_nuclear - μ_electronic
            This gives the dipole moment vector pointing from negative to positive charge.
            
        TODO: Allow user to specify output units (e.g., Debye).
        """
        nuc_dip = self.get_nuc_dip_moment()
        elec_dip = self.get_elec_dip_moment(dipole_moment_matrix, dmat)
        mol_dip = nuc_dip - elec_dip
        # TODO: allow the user to specify units (ex: DEBYE)
        return mol_dip
    
    #Export coord file in TURBOMOLE format
    def exportCoords(atomicSpecies, coords, filename='coord'):
        """
        Export molecular coordinates to TURBOMOLE format.
        
        Creates a coordinate file in TURBOMOLE format with atomic positions
        converted from Angstroms to Bohr units.
        
        Args:
            atomicSpecies (list): List of atomic symbols as strings.
            coords (numpy.ndarray): Atomic coordinates in Angstroms with shape (natoms, 3).
            filename (str, optional): Output filename. Defaults to 'coord'.
        
        File Format:
            $coord
            x_bohr y_bohr z_bohr element_symbol
            ...
            $end
        
        Note:
            This is a static method that should be called as a class method.
            Coordinates are automatically converted from Angstroms to Bohr units.
        
        Warning:
            This method contains a bug - it references undefined variable 'coord'
            instead of 'coords'. Should be fixed in implementation.
        """
        file = open(filename,'w')
        file.write('$coord\n')
        for i in range(len(atomicSpecies)):
            file.write(str(coord[i,0]*Data.Angs2BohrFactor)+'\t'+str(coord[i,1]*Data.Angs2BohrFactor)+'\t'+str(coord[i,2]*Data.Angs2BohrFactor)+'\t'+atomicSpecies[i].lower()+'\n')
        file.write('$end\n')
        file.close()

    #Export the geometry in XYZ format
    def exportXYZ(self, filename='coord', label='Generic Mol generated by CrysX (Python Library)'):
        """
        Export molecular geometry to XYZ format file.
        
        Creates a standard XYZ format file with atomic coordinates in Angstroms.
        This method uses the molecule's current atomic species and coordinates.
        
        Args:
            filename (str, optional): Base filename without extension. Defaults to 'coord'.
                                     The '.xyz' extension will be automatically added.
            label (str, optional): Comment line for the XYZ file. Defaults to 
                                  'Generic Mol generated by CrysX (Python Library)'.
        
        File Format:
            number_of_atoms
            comment_line
            element_symbol x_coord y_coord z_coord
            ...
        
        Output:
            Creates a file named '{filename}.xyz' in the current directory.
        
        Example:
            mol.exportXYZ('water', 'H2O molecule')
            # Creates 'water.xyz' with H2O coordinates
        """
        file = open(filename+'.xyz','w')
        file.write(str(self.natoms)+'\n')
        file.write(label+'\n')
        for i in range(self.natoms):
            file.write(self.atomicSpecies[i]+'\t'+str(self.coords[i,0])+'\t'+str(self.coords[i,1])+'\t'+str(self.coords[i,2])+'\n')
        file.close()   
    #Export the geometry in XYZ format
    def exportXYZs(atomicSpecies, coords, filename='coord', label='Generic Mol generated by CrysX (Python Library)'):
        """
        Export molecular geometry to XYZ format (static method version).
        
        Static method to create XYZ files from provided atomic species and coordinates,
        without requiring a Mol object instance.
        
        Args:
            atomicSpecies (list): List of atomic symbols as strings.
            coords (numpy.ndarray): Atomic coordinates in Angstroms with shape (natoms, 3).
            filename (str, optional): Base filename without extension. Defaults to 'coord'.
                                     The '.xyz' extension will be automatically added.
            label (str, optional): Comment line for the XYZ file. Defaults to 
                                  'Generic Mol generated by CrysX (Python Library)'.
        
        File Format:
            number_of_atoms
            comment_line
            element_symbol x_coord y_coord z_coord
            ...
        
        Warning:
            This method contains a bug - it references undefined variable 'coord'
            instead of 'coords'. Should be fixed in implementation.
        
        Note:
            This is a static method and should be called as a class method.
        """
        file = open(filename+'.xyz','w')
        file.write(str(len(atomicSpecies))+'\n')
        file.write(label+'\n')
        for i in range(len(atomicSpecies)):
            file.write(atomicSpecies[i]+'\t'+str(coord[i,0])+'\t'+str(coord[i,1])+'\t'+str(coord[i,2])+'\n')
        file.close()
    
    # Read geometry from TURBOOLE coords file
    def readCoordsFile(self, filename):
        """
        Read molecular coordinates from various file formats.
        
        Currently supports XYZ format files. This method serves as a dispatcher
        to format-specific reading methods based on file extension.
        
        Args:
            filename (str): Path to the coordinate file. File format is determined
                           by the file extension.
        
        Returns:
            list: List of atoms in the format [[symbol, x, y, z], ...] where
                  symbol is the atomic symbol and x, y, z are coordinates in Angstroms.
        
        Supported Formats:
            - .xyz: Standard XYZ coordinate format
        
        Future Extensions:
            - .mol: MOL file format
            - coord: TURBOMOLE coordinate format
            - Other quantum chemistry input formats
        
        Raises:
            Prints error message if file format is not supported.
        """
        if filename.endswith('.xyz'):
            atoms = self.readXYZ(filename)
        else:
            print('Only xyz files for now') 
        return atoms
    #Read geometry information from a .xyz file
    def readXYZ(self, filename):
        """
        Read molecular geometry from an XYZ format file.
        
        Parses a standard XYZ file and extracts atomic symbols and coordinates.
        The XYZ format consists of:
        - Line 1: Number of atoms
        - Line 2: Comment line (ignored)
        - Lines 3+: atomic_symbol x_coordinate y_coordinate z_coordinate
        
        Args:
            filename (str): Path to the XYZ file to read.
        
        Returns:
            list: List of atoms in format [[symbol, x, y, z], ...] where:
                  - symbol (str): Atomic symbol (e.g., 'H', 'C', 'O')
                  - x, y, z (float): Coordinates in Angstroms
        
        File Format Expected:
            number_of_atoms
            comment_line
            element_symbol x_coord y_coord z_coord
            element_symbol x_coord y_coord z_coord
            ...
        
        Error Handling:
            - Prints error if the number of atoms read doesn't match the declared count
            - Assumes coordinates are separated by whitespace
            - Strips whitespace from all input
        
        Example:
            For a file 'water.xyz':
            3
            Water molecule
            O    0.000000    0.000000    0.000000
            H    0.000000    0.000000    0.970000
            H    0.944863    0.000000   -0.242498
            
            Returns: [['O', 0.0, 0.0, 0.0], ['H', 0.0, 0.0, 0.97], ['H', 0.944863, 0.0, -0.242498]]
        """
        with open(filename, "r") as f:
            lines = f.readlines()

        if len(lines) < 2:
            raise ValueError(f"File {filename} does not contain enough lines for XYZ format")

        # Read number of atoms from first line
        try:
            natoms = int(lines[0].strip())
        except ValueError:
            raise ValueError(f"First line of {filename} must contain an integer atom count")

        # The second line is the comment (can be blank)
        comment_line = lines[1].rstrip("\n")

        atoms = []
        for i, line in enumerate(lines[2:2 + natoms], start=3):
            parts = line.split()
            if len(parts) < 4:
                raise ValueError(f"Line {i} in {filename} does not have 4 columns: {line.strip()}")
            symbol = parts[0]
            try:
                x, y, z = map(float, parts[1:4])
            except ValueError:
                raise ValueError(f"Invalid coordinates on line {i} in {filename}: {parts[1:4]}")
            atoms.append([symbol, x, y, z])

        if len(atoms) != natoms:
            raise ValueError(
                f"Expected {natoms} atoms, but found {len(atoms)} in {filename}"
            )

        return atoms
