__all__ = ["Grids"]
# Grids.py
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
from . import Data
from . import Mol
from . import Basis
import numgrid
from joblib import Parallel, delayed
#import multiprocessing
import os
from timeit import default_timer as timer


#TODO Change it to return actual cores rather than threads
#num_cores = multiprocessing.cpu_count()
num_cores = os.cpu_count()
"""Number of CPU cores to use for parallel grid generation via Joblib's threading backend."""

lebedevOrdering = {
    0  : 1   ,
    3  : 6   ,
    5  : 14  ,
    7  : 26  ,
    9  : 38  ,
    11 : 50  ,
    13 : 74  ,
    15 : 86  ,
    17 : 110 ,
    19 : 146 ,
    21 : 170 ,
    23 : 194 ,
    25 : 230 ,
    27 : 266 ,
    29 : 302 ,
    31 : 350 ,
    35 : 434 ,
    41 : 590 ,
    47 : 770 ,
    53 : 974 ,
    59 : 1202,
    65 : 1454,
    71 : 1730,
    77 : 2030,
    83 : 2354,
    89 : 2702,
    95 : 3074,
    101: 3470,
    107: 3890,
    113: 4334,
    119: 4802,
    125: 5294,
    131: 5810
}

#         Period    1   2   3   4   5   6   7         # level
Mapping = np.array([[11, 15, 17, 17, 17, 17, 17],     # 0        
                   [17, 23, 23, 23, 23, 23, 23],      # 1
                   [23, 29, 29, 29, 29, 29, 29],      # 2
                   [29, 29, 35, 35, 35, 35, 35],      # 3
                   [35, 41, 41, 41, 41, 41, 41],      # 4 This is the minimum that user can specify.
                   [41, 47, 47, 47, 47, 47, 47],      # 5
                   [47, 53, 53, 53, 53, 53, 53],      # 6
                   [53, 59, 59, 59, 59, 59, 59],      # 7
                   [59, 59, 59, 59, 59, 59, 59],      # 8  
                   [65, 65, 65, 65, 65, 65, 65]])     # 9  This is the maximum that user can specify.

def max_ang(charge, level):
    #Mapping the LEBEDEV order with level of grids
    period = int(Data.elementPeriod[charge])
    index = Mapping[level+1,period]
    return lebedevOrdering[index]


def min_ang(charge, level):
    #Mapping the LEBEDEV order with level of grids
    period = int(Data.elementPeriod[charge])
    index = Mapping[level-3,period]
    return lebedevOrdering[index]



def genGridiNewAPI(radial_precision,proton_charges,center_coordinates_bohr,basis_set_name,center_index,level,alpha_min, alpha_max):
    #This function generates the grid for a given atom (center_index)
    #This function is used to enable the generation of grids in parallel using joblib library
    #The idea being, that grids are generated atom-by-atom,
    #so it would be better to generate them parallely.
    
    # min_num_angular_points = 110#110#86#min_ang(proton_charges[center_index], level - 4)
    # max_num_angular_points = 434#302#max_ang(proton_charges[center_index], level)
    min_num_angular_points = min_ang(proton_charges[center_index], level)
    max_num_angular_points = max_ang(proton_charges[center_index], level)
    hardness = 3
    # alpha_max = [
    #                 11720.0,  # O
    #                 13.01,  # H
    #                 13.01,  # H
    #             ]
    # alpha_min = [
    #                 {0: 0.3023, 1: 0.2753, 2: 1.185},  # O
    #                 {0: 0.122, 1: 0.727},  # H
    #                 {0: 0.122, 1: 0.727},  # H
    #             ]

    # atom grid using explicit basis set parameters
    coordinates, weights = numgrid.atom_grid(
                                    alpha_min[center_index],
                                    alpha_max[center_index],
                                    radial_precision,
                                    min_num_angular_points,
                                    max_num_angular_points,
                                    proton_charges,
                                    center_index,
                                    center_coordinates_bohr,
                                    hardness
                                )
    
    # Doesn't seem to be working
    # The problem is that the REST API request to basis set exchange results in a timeout
    # coordinates, weights = numgrid.atom_grid_bse(basis_set_name,radial_precision,
    #                                 min_num_angular_points,
    #                                 max_num_angular_points,
    #                                 proton_charges,center_index,
    #                                 center_coordinates_bohr,
    #                                 hardness
    #                                 )


    coordinates = np.array(coordinates)
    x = coordinates[:,0]
    y = coordinates[:,1]
    z = coordinates[:,2]


    return x,y,z,weights

class Grids:
    """
    Class for generating molecular integration grids for DFT and other quantum chemistry calculations.

    This class uses the `numgrid` library to generate atom-centered grids composed of:
    1. A `level` indicating the fineness of the grid (from 3 to 8, with 0 reserved internally).
    2. `coords`: a (N, 3) NumPy array of Cartesian coordinates (in Bohrs) for the N grid points.
    3. `weights`: a NumPy array of length N containing the integration weights for each grid point.

    Grid density depends not only on the level but also on the basis set and the radial precision,
    making it more flexible and customizable than typical grid generation schemes.
    """

    #Class for the generation of molecular grids for numerical integration.
    #The grid has three main components.
    # 1. The grid 'level' refers to the coarsness or fineness of the grid. level = 4 (coarse) to 9 (fine)
    # Although, internally CrysX would treat 0 as the minimum limit.
    # This is to allow us to set a smaller min_num_angular_points and a larger max_num_angular_points.
    # 2. The grid 'coords' (Bohrs) refer to a (N,3) numpy array with 3D cartesian coordinates 
    # of N grid points.
    # 3. The grid 'weights' that is an N-element numpy array with the weights corresponding to the grid points.

    #In order to initialize a Grids object we need to provide the Mol object, Basis object and level (integer).
    #One very important thing to note here is that, we use the 'numgrid' library to generate grids.
    # https://github.com/dftlibs/numgrid
    #Because of the way numgrid is implemented, there are mainly three ways to control the density of grids
    # 1. Basis set: for radial grid
    # 2. LEBEDEV ORDER or min/max angular points. We take care of this using the 'level' keyword.
    # 3. Final the radial_precision.
    # So, unlike other packages where the grid size would just depend on the level specified,
    # here even the size of basis set also affect the grid size.


    def __init__(self, mol=None, basis=None, level=3, radial_precision=1.0e-13, ncores = os.cpu_count()):
        """
        Initialize a Grids object for molecular numerical integration using atom-centered grids.

        Parameters
        ----------
        mol : object
            A Mol object containing atomic coordinates (`mol.coordsBohrs`) and nuclear charges (`mol.Zcharges`).
            Required for placing atom-centered grid points.

        basis : object or None
            A Basis object specifying the basis set to use. If None, defaults to 'def2-QZVPP'.
            Affects the radial distribution of the grid.

        level : int, default=3
            Grid resolution level. Valid values are 3 (coarse) to 8 (fine).
            Internally mapped to a min/max angular point scheme used by the `numgrid` library.

        radial_precision : float, default=1.0e-13
            Precision used in radial grid generation. Lower values increase the number of radial points.

        ncores : int, optional
            Number of CPU cores to use for parallel grid generation. Defaults to the number of available cores.

        Raises
        ------
        ValueError
            If `mol` is None or `level` is outside the supported range [3, 8].

        Notes
        -----
        Uses the `numgrid` library for generating atomic-centered numerical integration grids.
        The final grid coordinates and weights are stored in the `coords` and `weights` attributes, respectively.
        """
        # For level user can enter an integer from 3 to 8.

        # To get the same energies as PySCF (level=5) upto 1e-7 au, use the following settings
        # radial_precision=1.0e-13
        # level=3
        # pruning by density with threshold = 1e-011
        # alpha_min and alpha_max corresponding to QZVP


        # As a default we can use the def2-TZVP basis set, in case no basis set is provided
        # or a custom basis set is provided, in which case we won't have a specific name to search for in BSE database.

        if basis is None:
            basis_set_name = 'def2-QZVPP'
        else:
            basis_set_name = basis.basisSet.splitlines()[0]  #TODO FIXME But this would only work if the basis set was 
                                                             #specified from CrysX lib using Basis.load()
                                                             #If a user specified basis set was given via a string
                                                             #or a file, then this would fail.
        
        if mol is None:
            print("ERROR: Can't generate grids without molecular information!")
            return

        if level<3 or level>8:
            print("ERROR: Enter a valid value for level between 3 to 8!")
            return

        self.level = level
        """The chosen grid level (integer between 3 and 8). Controls the angular resolution and density of the integration grid."""


        #Create the atomic coordinate arrays needed by numgrid
        x_coordinates_bohr = []
        y_coordinates_bohr = []
        z_coordinates_bohr = []
        center_coordinates_bohrs = []
        for i in range (mol.natoms):
            x_coordinates_bohr.append(mol.coordsBohrs[i][0])
            y_coordinates_bohr.append(mol.coordsBohrs[i][1])
            z_coordinates_bohr.append(mol.coordsBohrs[i][2])
            center_coordinates_bohrs.append((mol.coordsBohrs[i][0],mol.coordsBohrs[i][1],mol.coordsBohrs[i][2]))

        
        #Get the required values
        num_centers = mol.natoms
        proton_charges = mol.Zcharges

        #Now start the grid stuff

        
        #Create arrays to store the returned grid points and their weights
        #These will later be turned into numpy arrays
        x_grid = []
        y_grid = []
        z_grid = []
        w_grid = []


        print('Grid generation using '+str(ncores)+' threads for Joblib.')
        output = Parallel(n_jobs=ncores, backend='threading', require='sharedmem')(delayed(genGridiNewAPI)(radial_precision,proton_charges,center_coordinates_bohrs,basis_set_name,center_index,level,basis.alpha_min, basis.alpha_max) for center_index in range(num_centers))
        num_total_grid_points = 0
        for center_index in range(num_centers):
            num_total_grid_points = num_total_grid_points + len(output[center_index][0])
            x_grid.extend(output[center_index][0])
            y_grid.extend(output[center_index][1])
            z_grid.extend(output[center_index][2])
            w_grid.extend(output[center_index][3])


        #print(num_total_grid_points)
        #Now create the numpy arrays that store the grid points and weights
        #self.coords = np.empty((len(num_total_grid_points),3))
        #self.weights = np.empty((len(num_total_grid_points)))
        self.coords = np.empty((len(x_grid),3))
        """A NumPy array of shape (N, 3), storing the Cartesian coordinates of N grid points (in Bohrs)."""
        self.weights = np.empty((len(x_grid)))
        """A NumPy array of length N, storing the quadrature weights corresponding to each grid point."""
        #Convert the Python arrays x_grid, y_grid, z_grid to the Grids.coords numpy arrays
        self.coords[:,0] = np.array(x_grid)
        self.coords[:,1] = np.array(y_grid)
        self.coords[:,2] = np.array(z_grid)
        # center_coordinates_bohrs = np.array(center_coordinates_bohrs)
        # self.coords = self.coords[(np.abs(np.subtract.outer(self.coords,center_coordinates_bohrs)) > 1e-5).all(1)]
        self.weights[:] = np.array(w_grid)
        return

        



        

