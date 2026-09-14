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
import numba
from numba import njit, prange
from . import Data
import numgrid
from joblib import Parallel, delayed
#import multiprocessing
import os


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
"""Lebedev order -> number of angular points."""

LEBEDEV_SIZES = np.array(sorted(lebedevOrdering.values()))
"""Numbers of points of the available Lebedev angular grids, ascending."""

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



def genGridiNewAPI(radial_precision,proton_charges,center_coordinates_bohr,basis_set_name,center_index,level,alpha_min, alpha_max, angular_points=None, hardness=3):
    #This function generates the grid for a given atom (center_index)
    #This function is used to enable the generation of grids in parallel using joblib library
    #The idea being, that grids are generated atom-by-atom,
    #so it would be better to generate them parallely.
    
    # min_num_angular_points = 110#110#86#min_ang(proton_charges[center_index], level - 4)
    # max_num_angular_points = 434#302#max_ang(proton_charges[center_index], level)
    if angular_points is None:
        min_num_angular_points = min_ang(proton_charges[center_index], level)
        max_num_angular_points = max_ang(proton_charges[center_index], level)
    else:
        min_num_angular_points, max_num_angular_points = int(angular_points[0]), int(angular_points[1])
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


    coordinates = np.asarray(coordinates, dtype=np.float64).reshape(-1, 3)
    weights = np.asarray(weights, dtype=np.float64)
    return coordinates, weights


# Period index of an element (H, He -> 0; Li-Ne -> 1; ...), the column of the level tables below.
_PERIOD_LIMITS = np.array([2, 10, 18, 36, 54, 86, 118])


def period_index(charge):
    """Row of the periodic table counted from 0 (H, He -> 0, Li-Ne -> 1, ...); ghost atoms (charge 0) count as period 0."""
    return int((int(charge) > _PERIOD_LIMITS).sum())


# Radial precision requested from numgrid's LMG radial grid for each level of the 'compact' preset. The LMG grid
# adds radial points until the estimated discretisation error of the basis-set exponents (def2-QZVP) falls below
# this value; these precisions give per-element radial counts close to those of the 'treutler' scheme (level 3:
# 78 for C and 56 for H against 75 and 50).
RADIAL_PRECISION = {0: 1.0e-4, 1: 1.0e-6, 2: 1.0e-7, 3: 1.0e-8, 4: 1.0e-9, 5: 1.0e-10,
                    6: 1.0e-11, 7: 1.0e-12, 8: 1.0e-13, 9: 1.0e-14}
"""numgrid radial precision of every grid level of the 'compact' preset."""

# Smallest Lebedev grid used by numgrid in the innermost shells (r < Bragg radius / 5) for each level.
MIN_ANGULAR = {0: 26, 1: 38, 2: 50, 3: 50, 4: 86, 5: 86, 6: 86, 7: 110, 8: 110, 9: 110}
"""Smallest number of angular points of every grid level of the 'compact' preset."""


def preset_angular_points(charge, level):
    """(min, max) number of Lebedev angular points of the 'compact' preset: the largest angular grid of the level
    for the element (``Mapping`` table) and the level's minimum for the innermost shells."""
    max_points = lebedevOrdering[int(Mapping[int(level), period_index(charge)])]
    return int(MIN_ANGULAR[int(level)]), int(max_points)


def box_grouping_order(atm_coords, coords, box_size=1.2, boundary_penalty=4.2):
    """Permutation that groups grid points into cubic boxes of edge ``box_size`` Bohr (spatial locality).

    The space around the molecule (bounding box of the atoms extended by ``boundary_penalty`` Bohr) is divided
    into boxes and the points are ordered box by box, keeping their original order inside a box. Nearby points
    then form the batches of the XC evaluation, so fewer basis functions contribute to each batch. Pure NumPy.
    """
    atm_coords = np.asarray(atm_coords, dtype=np.float64).reshape(-1, 3)
    coords = np.asarray(coords, dtype=np.float64).reshape(-1, 3)
    lower = atm_coords.min(axis=0) - boundary_penalty
    upper = atm_coords.max(axis=0) + boundary_penalty
    boxes = np.maximum(((upper - lower) * (1.0 / box_size)).round().astype(np.int64), 1)
    box_size = (upper - lower) / boxes
    box_ids = np.floor((coords - lower) * (1.0 / box_size)).astype(np.int64)
    box_ids[box_ids < -1] = -1
    for k in range(3):
        box_ids[box_ids[:, k] > boxes[k], k] = boxes[k]
    # linear box index with x as the slowest index (same order as sorting the (ix, iy, iz) triples)
    lin = ((box_ids[:, 0] + 1) * (boxes[1] + 2) + (box_ids[:, 1] + 1)) * (boxes[2] + 2) + (box_ids[:, 2] + 1)
    return np.argsort(lin, kind='stable')



# ---------------------------------------------------------------------------------------------------------------
# Native grids (scheme='treutler'): Treutler-Ahlrichs radial grids, Lebedev angular grids (tables from numgrid),
# angular pruning by radial regions and Becke partitioning with Treutler's atomic-size adjustment. The numbers of radial
# points and the Lebedev orders of every level are tabulated in RADIAL_POINTS_TABLE and Mapping.
# ---------------------------------------------------------------------------------------------------------------
BOHR_IN_ANGSTROM = 0.52917721092
"""Bohr radius in Angstrom used to convert the tabulated atomic radii."""

BRAGG_SLATER_RADII_ANGSTROM = np.array([
    2.00,     0.35, 1.40, 1.45, 1.05, 0.85, 0.70, 0.65, 0.60, 0.50,  # ghost, H-F
    1.50, 1.80, 1.50, 1.25, 1.10, 1.00, 1.00, 1.00, 1.80, 2.20,  # Ne-K
    1.80, 1.60, 1.40, 1.35, 1.40, 1.40, 1.40, 1.35, 1.35, 1.35,  # Ca-Cu
    1.35, 1.30, 1.25, 1.15, 1.15, 1.15, 1.90, 2.35, 2.00, 1.80,  # Zn-Y
    1.55, 1.45, 1.45, 1.35, 1.30, 1.35, 1.40, 1.60, 1.55, 1.55,  # Zr-In
    1.45, 1.45, 1.40, 1.40, 2.10, 2.60, 2.15, 1.95, 1.85, 1.85,  # Sn-Pr
    1.85, 1.85, 1.85, 1.85, 1.80, 1.75, 1.75, 1.75, 1.75, 1.75,  # Nd-Tm
    1.75, 1.75, 1.55, 1.45, 1.35, 1.35, 1.30, 1.35, 1.35, 1.35,  # Yb-Au
    1.50, 1.90, 1.80, 1.60, 1.90, 1.45, 2.10, 1.80, 2.15, 1.95,  # Hg-Ac
    1.80, 1.80, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75,  # Th-Es
    1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75,  # Fm-Ds
    1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75,  # Rg-Ubn
    1.75,
])
"""Bragg-Slater atomic radii in Angstrom indexed by the nuclear charge (J. C. Slater, J. Chem. Phys. 41, 3199
(1964); 0.35 A for H and 1.40 A for He as customary in Becke partitioning; 1.75 A beyond Ac; 2 A for ghost atoms)."""

BRAGG_SLATER_RADII = BRAGG_SLATER_RADII_ANGSTROM / BOHR_IN_ANGSTROM
"""Bragg-Slater atomic radii in Bohr indexed by the nuclear charge."""

TREUTLER_AHLRICHS_XI = np.array([
    1.0,                                                          # ghost
    0.8, 0.9,                                                     # H, He
    1.8, 1.4, 1.3, 1.1, 0.9, 0.9, 0.9, 0.9,                       # Li-Ne
    1.4, 1.3, 1.3, 1.2, 1.1, 1.0, 1.0, 1.0,                       # Na-Ar
    1.5, 1.4, 1.3, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.1, 1.1, 1.1,   # K-Zn
    1.1, 1.0, 0.9, 0.9, 0.9, 0.9,                                 # Ga-Kr
    2.0, 1.7, 1.5, 1.5, 1.35, 1.35, 1.25, 1.2, 1.25, 1.3, 1.5, 1.5,   # Rb-Cd
    1.3, 1.2, 1.2, 1.15, 1.15, 1.15,                              # In-Xe
    2.5, 2.2, 2.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5,                  # Cs, Ba, La-Eu
    1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5,                       # Gd-Lu
    1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5,                  # Hf-Hg
    1.5, 1.5, 1.5, 1.5, 1.5, 1.5,                                 # Tl-Rn
    2.5, 2.1, 3.685, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5,                # Fr, Ra, Ac-Am
    1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5,                       # Cm-Lr
])
"""Element-specific scaling parameter xi of the Treutler-Ahlrichs M4 radial grid: H-Kr from Table I of
O. Treutler and R. Ahlrichs, J. Chem. Phys. 102, 346 (1995); the values in common use for the heavier elements."""

#           Period    1   2   3   4   5   6   7        # level
RADIAL_POINTS_TABLE = np.array([[ 10, 15, 20, 30, 35, 40, 50],   # 0
                      [ 30, 40, 50, 60, 65, 70, 75],   # 1
                      [ 40, 60, 65, 75, 80, 85, 90],   # 2
                      [ 50, 75, 80, 90, 95,100,105],   # 3
                      [ 60, 90, 95,105,110,115,120],   # 4
                      [ 70,105,110,120,125,130,135],   # 5
                      [ 80,120,125,135,140,145,150],   # 6
                      [ 90,135,140,150,155,160,165],   # 7
                      [100,150,155,165,170,175,180],   # 8
                      [200,200,200,200,200,200,200]])  # 9
"""Number of radial points of the 'treutler' scheme for each grid level (rows) and period of the element (columns);
the columns follow ``period_index``."""

# Radii (in units of the Bragg-Slater radius) at which the angular pruning switches to the next angular grid,
# for H-He, Li-Ne and heavier elements.
_PRUNING_REGION_RADII = np.array([[0.25, 0.5, 1.0, 4.5],
                                [0.1667, 0.5, 0.9, 3.5],
                                [0.1, 0.4, 0.8, 2.5]])


def radial_points_for_level(charge, level):
    """Number of Treutler-Ahlrichs radial points of the 'treutler' scheme for an element at a level (0-9)."""
    return int(RADIAL_POINTS_TABLE[int(level), period_index(charge)])


def angular_points_for_level(charge, level):
    """Largest number of Lebedev angular points of the 'treutler' scheme for an element at a level (0-9)."""
    return int(lebedevOrdering[int(Mapping[int(level), period_index(charge)])])


_LEBEDEV_CACHE = {}


def lebedev_grid(num_points):
    """Lebedev angular grid with ``num_points`` points on the unit sphere, from numgrid (at run time).

    Returns ``(xyz, weights)``: an ``(num_points, 3)`` array of unit vectors and weights summing to 1.
    Results are cached.
    """
    num_points = int(num_points)
    grid = _LEBEDEV_CACHE.get(num_points)
    if grid is None:
        if num_points not in lebedevOrdering.values():
            raise ValueError('There is no Lebedev grid with ' + str(num_points) + ' points. Available: '
                             + ', '.join(str(n) for n in LEBEDEV_SIZES))
        xyz, weights = numgrid.angular_grid(num_points)
        xyz = np.ascontiguousarray(np.asarray(xyz, dtype=np.float64).reshape(-1, 3))
        weights = np.ascontiguousarray(np.asarray(weights, dtype=np.float64))
        if abs(weights.sum() - 1.0) > 1e-10:
            raise RuntimeError('numgrid returned Lebedev weights that do not sum to 1 for ' + str(num_points) + ' points.')
        grid = (xyz, weights)
        _LEBEDEV_CACHE[num_points] = grid
    return grid


def treutler_m4_radial_grid(num_points, charge=0, xi=None):
    """Treutler-Ahlrichs M4 radial quadrature (J. Chem. Phys. 102, 346 (1995)) with ``num_points`` points.

    Gauss-Chebyshev nodes of the second kind x_i = cos(i pi/(n+1)) are mapped to
    r = xi/ln2 (1 + x)^0.6 ln(2/(1 - x)); the weights include the Jacobian dr/dx. Returns ``(r, dr)`` in
    ascending order of r, such that ``sum(f(r) * 4 pi r^2 * dr)`` integrates ``f`` over space. ``xi`` defaults
    to the element-specific value of the paper (``TREUTLER_AHLRICHS_XI[charge]``).
    """
    n = int(num_points)
    if xi is None:
        charge = int(charge)
        xi = float(TREUTLER_AHLRICHS_XI[charge]) if 0 <= charge < TREUTLER_AHLRICHS_XI.shape[0] else 1.0
    alpha = 0.6
    theta = np.arange(1, n + 1, dtype=np.float64) * (np.pi / (n + 1))
    x = np.cos(theta)
    log_term = np.log(2.0 / (1.0 - x))
    r = xi / np.log(2.0) * (1.0 + x) ** alpha * log_term
    drdx = xi / np.log(2.0) * (alpha * (1.0 + x) ** (alpha - 1.0) * log_term + (1.0 + x) ** alpha / (1.0 - x))
    dr = np.pi / (n + 1) * np.sin(theta) * drdx   # Gauss-Chebyshev (second kind) weight pi/(n+1) sin^2(theta)/sqrt(1-x^2)
    return r[::-1].copy(), dr[::-1].copy()


def region_angular_sizes(charge, rads, n_ang, radii=BRAGG_SLATER_RADII):
    """Angular pruning by radial regions: the number of Lebedev points of every radial shell of an atom.

    The two innermost regions use 50 and 86 points, the valence region the full ``n_ang`` points and the outer
    region the next smaller Lebedev grid; the region boundaries are fixed multiples of the Bragg-Slater radius
    (0.25, 0.5, 1.0, 4.5 for H-He; 0.1667, 0.5, 0.9, 3.5 for Li-Ne; 0.1, 0.4, 0.8, 2.5 beyond).
    """
    rads = np.asarray(rads, dtype=np.float64)
    charge = int(charge)
    n_ang = int(n_ang)
    sizes = LEBEDEV_SIZES[LEBEDEV_SIZES >= 38]
    if n_ang < 50:
        return np.repeat(n_ang, rads.shape[0])
    if n_ang == 50:
        per_region = np.array([50, 74, 74, 74, 50])
    else:
        idx = np.where(sizes == n_ang)[0]
        if idx.shape[0] == 0:
            raise ValueError('n_ang=' + str(n_ang) + ' is not a Lebedev grid size. Available: '
                             + ', '.join(str(n) for n in LEBEDEV_SIZES))
        smaller = int(sizes[int(idx[0]) - 1])
        per_region = np.array([50, 86, smaller, n_ang, smaller])
    r_atom = radii[charge] + 1e-200
    boundaries = _PRUNING_REGION_RADII[0 if charge <= 2 else 1 if charge <= 10 else 2]
    region = ((rads / r_atom).reshape(-1, 1) > boundaries).sum(axis=1)
    return per_region[region]


_ATOMIC_GRID_CACHE = {}


def single_atom_grid(charge, level=3, pruning='regions', n_rad=None, n_ang=None):
    """Single-atom grid of the 'treutler' scheme centred at the origin.

    Returns ``(coords, vol)``: the ``(N, 3)`` point coordinates relative to the nucleus (Bohr) and the volume
    element ``4 pi r^2 dr w_ang`` of every point. ``n_rad``/``n_ang`` override the defaults of the level.
    ``pruning`` is ``'regions'``, ``None`` (every shell gets ``n_ang`` points) or a callable
    ``pruning(charge, r, n_ang)`` returning the angular-grid size of every shell. The points are ordered by
    ascending angular-grid size and, within it, in groups of twelve radial shells with the angular index
    outermost.
    """
    charge = int(charge)
    if n_rad is None:
        n_rad = radial_points_for_level(charge, level)
    if n_ang is None:
        n_ang = angular_points_for_level(charge, level)
    n_rad = int(n_rad)
    n_ang = int(n_ang)
    cacheable = pruning is None or isinstance(pruning, str)
    key = (charge, n_rad, n_ang, pruning)
    if cacheable and key in _ATOMIC_GRID_CACHE:
        return _ATOMIC_GRID_CACHE[key]

    r, dr = treutler_m4_radial_grid(n_rad, charge)
    rad_weight = 4 * np.pi * r ** 2 * dr
    if pruning is None:
        angs = np.repeat(n_ang, n_rad)
    elif isinstance(pruning, str):
        if pruning.lower() != 'regions':
            raise ValueError("Unknown pruning scheme '" + str(pruning) + "'. Available: 'regions' or None.")
        angs = region_angular_sizes(charge, r, n_ang)
    elif callable(pruning):
        angs = np.asarray(pruning(charge, r, n_ang), dtype=np.int64)
    else:
        raise ValueError("pruning must be 'regions', None or a callable.")

    coords = []
    vol = []
    for n in sorted(set(angs.tolist())):
        xyz, w = lebedev_grid(n)
        idx = np.where(angs == n)[0]
        for i0 in range(0, idx.shape[0], 12):
            sub = idx[i0:i0 + 12]
            coords.append(np.einsum('i,jk->jik', r[sub], xyz).reshape(-1, 3))
            vol.append(np.einsum('i,j->ji', rad_weight[sub], w).ravel())
    coords = np.ascontiguousarray(np.vstack(coords))
    vol = np.ascontiguousarray(np.hstack(vol))
    if cacheable:
        _ATOMIC_GRID_CACHE[key] = (coords, vol)
    return coords, vol


def size_adjustment_table(charges, scheme='treutler', radii=BRAGG_SLATER_RADII):
    """Atomic-size adjustment a_ij of the Becke partitioning for all atom pairs.

    Becke's adjustment (J. Chem. Phys. 88, 2547 (1988)) a_ij = (chi - 1/chi)/4 with chi = R_i/R_j, limited to
    [-1/2, 1/2]; ``scheme='treutler'`` uses chi = sqrt(R_i/R_j) as proposed by Treutler and Ahlrichs,
    ``'becke'`` the radii themselves. Returns ``None`` for ``scheme=None`` (no adjustment), otherwise an
    ``(natoms, natoms)`` array with a_ij = -a_ji.
    """
    if scheme is None:
        return None
    scheme = str(scheme).lower()
    charges = np.asarray(charges, dtype=np.int64)
    if scheme == 'treutler':
        rad = np.sqrt(radii[charges])
    elif scheme == 'becke':
        rad = radii[charges].astype(np.float64)
    else:
        raise ValueError("Unknown size_adjustment scheme '" + str(scheme) + "'. Available: 'treutler', 'becke' or None.")
    chi = rad.reshape(-1, 1) / rad.reshape(1, -1)   # chi_ij = R_i / R_j
    a = 0.25 * (1.0 / chi - chi)
    a[a < -0.5] = -0.5
    a[a > 0.5] = 0.5
    return np.ascontiguousarray(a)


@njit(parallel=True, cache=True, nogil=True)
def _becke_partition_kernel(coords, atom_idx, atm_coords, inv_dist, a_table, use_adjust, blocksize, out):
    # Becke's fuzzy-cell weight of every grid point for the atom it belongs to:
    #   P_A(r) / sum_B P_B(r),  P_A = prod_{B != A} s(nu_AB),  s(nu) = (1 - f(f(f(nu))))/2,  f(x) = (3x - x^3)/2,
    #   nu_AB = mu_AB + a_AB (1 - mu_AB^2),  mu_AB = (|r - R_A| - |r - R_B|) / |R_A - R_B|.
    # All atom pairs are evaluated for every point (no screening).
    npts = coords.shape[0]
    natm = atm_coords.shape[0]
    nblocks = (npts + blocksize - 1) // blocksize
    for ib in prange(nblocks):
        p0 = ib * blocksize
        p1 = min(p0 + blocksize, npts)
        d = np.empty(natm)
        P = np.empty(natm)
        for p in range(p0, p1):
            x = coords[p, 0]
            y = coords[p, 1]
            z = coords[p, 2]
            for k in range(natm):
                dx = x - atm_coords[k, 0]
                dy = y - atm_coords[k, 1]
                dz = z - atm_coords[k, 2]
                d[k] = np.sqrt(dx * dx + dy * dy + dz * dz)
                P[k] = 1.0
            for i in range(natm):
                di = d[i]
                for j in range(i):
                    g = (di - d[j]) * inv_dist[i, j]
                    if use_adjust:
                        g += a_table[i, j] * (1.0 - g * g)
                    g = (3.0 - g * g) * g * 0.5
                    g = (3.0 - g * g) * g * 0.5
                    g = (3.0 - g * g) * g * 0.5
                    P[i] *= 0.5 * (1.0 - g)
                    P[j] *= 0.5 * (1.0 + g)
            s = 0.0
            for k in range(natm):
                s += P[k]
            out[p] = P[atom_idx[p]] / s


def becke_partition_weights(coords, atom_idx, atm_coords, a_table=None, blocksize=128):
    """Becke partitioning factors P_A(r)/sum_B P_B(r) of grid points belonging to the atoms ``atom_idx``.

    ``coords`` (N, 3) and ``atm_coords`` (natoms, 3) in Bohr; ``a_table`` is the (natoms, natoms) atomic-size
    adjustment from :func:`size_adjustment_table` or ``None``. Multiplying the single-atom volume elements by
    these factors gives the molecular quadrature weights. Runs on the Numba threads set by the caller.
    """
    coords = np.ascontiguousarray(coords, dtype=np.float64)
    atm_coords = np.ascontiguousarray(atm_coords, dtype=np.float64)
    atom_idx = np.ascontiguousarray(atom_idx, dtype=np.int64)
    natm = atm_coords.shape[0]
    out = np.ones(coords.shape[0], dtype=np.float64)
    if natm < 2 or coords.shape[0] == 0:
        return out
    diff = atm_coords[:, None, :] - atm_coords[None, :, :]
    dist = np.sqrt(np.einsum('ijk,ijk->ij', diff, diff))
    inv_dist = np.zeros((natm, natm))
    off = ~np.eye(natm, dtype=bool)
    inv_dist[off] = 1.0 / dist[off]
    if a_table is None:
        a_table = np.zeros((natm, natm))
        use_adjust = False
    else:
        a_table = np.ascontiguousarray(a_table, dtype=np.float64)
        use_adjust = True
    _becke_partition_kernel(coords, atom_idx, atm_coords, inv_dist, a_table, use_adjust, int(blocksize), out)
    return out


class Grids:
    """
    Class for generating molecular integration grids for DFT and other quantum chemistry calculations.

    A ``Grids`` object holds:
    1. ``level``: the fineness of the grid, 0 (coarsest) to 9 (finest).
    2. ``coords``: a (N, 3) NumPy array of Cartesian coordinates (in Bohrs) of the N grid points.
    3. ``weights``: a NumPy array of length N with the integration weights of the points.
    4. ``atom_idx``: a NumPy integer array of length N with the index of the atom each point belongs to.

    Two schemes are available (``scheme``):

    ``'treutler'`` (default)
        Native grids: Treutler-Ahlrichs M4 radial grids with the element-specific scaling of the original
        paper, Lebedev angular grids (tables provided by numgrid at run time), angular pruning by regions of
        the Bragg-Slater radius and Becke partitioning with Treutler's atomic-size adjustment. The numbers of
        radial points and the Lebedev orders of every level are tabulated per period of the element.
        The partitioning runs in a parallel Numba kernel, or on the GPU with ``use_gpu=True``.

    ``'numgrid'``
        The grids of the numgrid library: LMG radial grids built from the smallest and largest exponents of a
        basis set (def2-QZVP by default), Lebedev angular grids pruned inside one fifth of the Bragg radius,
        numgrid's Becke partitioning. ``preset='compact'`` chooses the radial precision and the angular grids
        of the level (grids of a similar size to the 'treutler' grids, comparable accuracy at level 3 for light
        elements), ``preset='dense'`` gives the previous PyFock grids (radial precision 1e-13, up to 590
        angular points at level 3). See docs/xc_grids.md for the comparison.
    """

    def __init__(self, mol=None, basis=None, level=3, radial_precision=None, ncores=os.cpu_count(),
                 scheme=None, preset=None, angular_points=None, hardness=3, sort=True, verbose=True,
                 pruning='regions', size_adjustment='treutler', points_per_element=None, use_gpu=False):
        """
        Generate the molecular integration grid.

        Parameters
        ----------
        mol : Mol
            Molecule (atomic coordinates ``mol.coordsBohrs`` and nuclear charges ``mol.Zcharges``).
        basis : Basis or None
            ``scheme='numgrid'`` only: its smallest and largest exponents define the radial extent of the
            grid of every atom; None builds a def2-QZVP basis. Ignored by ``scheme='treutler'``.
        level : int, default=3
            Grid level, 0 (coarsest) to 9 (finest) for both schemes (the numgrid ``'dense'`` preset supports
            levels 3 to 8).
        radial_precision : float or None
            ``scheme='numgrid'`` only: precision requested from the LMG radial grid; None takes the value of
            the level from the preset (``RADIAL_PRECISION``; 1e-13 for ``'dense'``).
        ncores : int, optional
            Number of threads (Numba kernel of the partitioning, or the per-atom numgrid calls).
        scheme : str or None
            ``'treutler'`` or ``'numgrid'``. None (default) selects ``'numgrid'`` when one of the numgrid
            options ``preset``, ``radial_precision`` or ``angular_points`` is given and ``'treutler'`` otherwise.
        preset : str or None
            ``scheme='numgrid'`` only: ``'compact'`` (default) or ``'dense'``.
        angular_points : tuple or None
            ``scheme='numgrid'`` only: ``(min, max)`` number of Lebedev angular points for all atoms.
        hardness : int, default=3
            ``scheme='numgrid'`` only: number of iterations of Becke's smoothing polynomial.
        sort : bool, default=True
            Group the points box by box (1.2 Bohr boxes) so that neighbouring points form
            the batches of the XC evaluation. ``is_sorted`` records whether this was done.
        verbose : bool, default=True
            Print a one-line summary of the grid.
        pruning : str, None or callable, default='regions'
            ``scheme='treutler'`` only: angular pruning, ``'regions'`` (by radial regions of the Bragg-Slater
            radius, see ``region_angular_sizes``), ``None`` (no pruning) or a
            callable ``pruning(charge, r, n_ang)`` returning the angular-grid size of every radial shell.
        size_adjustment : str or None, default='treutler'
            ``scheme='treutler'`` only: atomic-size adjustment of the Becke partitioning, ``'treutler'``,
            ``'becke'`` or ``None``.
        points_per_element : dict, optional
            ``scheme='treutler'`` only: ``{element symbol or charge: (n_rad, n_ang)}`` overrides of the number
            of radial points and of the largest Lebedev grid, e.g. ``{'C': (75, 302)}``.
        use_gpu : bool, default=False
            ``scheme='treutler'`` only: build the grid on the GPU (:mod:`pyfock.Grids_cupy`) instead of with
            the Numba CPU kernel. The points, the partitioning and the box grouping are all computed on the
            device and the finished grid is copied back, so ``coords``, ``weights`` and ``atom_idx`` are
            NumPy arrays either way: the points, their atom indices and their box order are identical to the
            CPU ones and the weights agree to the last bits (see :mod:`pyfock.Grids_cupy`). Falls back to the
            CPU with a warning when CuPy or a CUDA device is unavailable or the molecule has more atoms than
            the kernels are compiled for (``Grids_cupy.MAX_GPU_ATOMS``). The attribute ``use_gpu`` records
            what was actually used.

        Raises
        ------
        ValueError
            If ``mol`` is None, the level is outside the supported range or an option is unknown.
        """
        if mol is None:
            raise ValueError("Can't generate grids without molecular information: a Mol object is required.")
        if scheme is None:
            numgrid_options = preset is not None or radial_precision is not None or angular_points is not None
            scheme = 'numgrid' if numgrid_options else 'treutler'
        scheme = str(scheme).lower()
        if scheme not in ('treutler', 'numgrid'):
            raise ValueError("Unknown grid scheme '" + str(scheme) + "'. Available: 'treutler' (default) or 'numgrid'.")
        level = int(level)
        if preset is None:
            preset = 'compact'
        preset = str(preset).lower()
        if scheme == 'numgrid' and preset not in ('compact', 'dense'):
            raise ValueError("Unknown grids preset '" + str(preset) + "'. Available: 'compact' (default) or 'dense'.")
        if scheme == 'numgrid' and preset == 'dense':
            if level < 3 or level > 8:
                raise ValueError("Enter a valid value for level between 3 and 8 for the 'dense' preset!")
        elif level < 0 or level > 9:
            raise ValueError("Enter a valid value for level between 0 and 9!")
        if ncores is None:
            ncores = os.cpu_count()
        self.ncores = int(max(1, ncores))

        self.level = level
        """The grid level (0-9). Controls the number of radial and angular points."""
        self.scheme = scheme
        """The grid scheme, 'treutler' or 'numgrid'."""
        self.mol = mol
        """The Mol object the grid was generated for."""
        self.natoms = int(mol.natoms)
        """Number of atoms (grid centres)."""
        self.is_sorted = False
        """Whether the points are grouped box by box (see ``sort``)."""
        self.preset = preset if scheme == 'numgrid' else None
        """numgrid parameter preset ('compact' or 'dense'); None for the 'treutler' scheme."""
        self.radial_precision = None
        """Radial precision requested from numgrid's LMG radial grid; None for the 'treutler' scheme."""
        self.angular_points = {}
        """numgrid scheme: (min, max) number of Lebedev angular points per element symbol."""
        self.radial_points = {}
        """numgrid scheme: number of radial points per element symbol."""
        self.pruning = pruning if scheme == 'treutler' else None
        """'treutler' scheme: angular pruning ('regions', None or a callable)."""
        self.size_adjustment = size_adjustment if scheme == 'treutler' else None
        """'treutler' scheme: atomic-size adjustment of the partitioning ('treutler', 'becke' or None)."""
        self.element_points = {}
        """'treutler' scheme: (n_rad, n_ang) per element symbol (n_ang: largest Lebedev grid before pruning)."""
        if use_gpu and scheme != 'treutler':
            print("Warning: use_gpu is only available for the 'treutler' scheme; the " + scheme
                  + ' grid is built on the CPU.', flush=True)
        self.use_gpu = bool(use_gpu) and scheme == 'treutler'
        """Whether the grid was built on the GPU (see the ``use_gpu`` argument)."""

        atm_coords = np.ascontiguousarray(np.asarray(mol.coordsBohrs, dtype=np.float64).reshape(-1, 3))
        charges = np.asarray(mol.Zcharges, dtype=np.int64)

        sorted_on_gpu = False
        if scheme == 'treutler':
            if self.use_gpu:
                try:
                    coords, weights, atom_idx = self._build_treutler_gpu(atm_coords, charges, level, pruning,
                                                                         size_adjustment, points_per_element, sort)
                    sorted_on_gpu = sort and coords.shape[0] > 0
                except Exception as error:
                    print('Warning: the XC grid could not be built on the GPU (' + str(error)
                          + '); building it on the CPU instead.', flush=True)
                    self.use_gpu = False
                    self.element_points = {}
            if not self.use_gpu:
                coords, weights, atom_idx = self._build_treutler(atm_coords, charges, level, pruning, size_adjustment, points_per_element)
        else:
            coords, weights, atom_idx = self._build_numgrid(mol, basis, atm_coords, charges, level, preset,
                                                            radial_precision, angular_points, hardness)

        if sorted_on_gpu:
            self.is_sorted = True
        elif sort and coords.shape[0] > 0:
            perm = box_grouping_order(atm_coords, coords)
            coords = coords[perm]
            weights = weights[perm]
            atom_idx = atom_idx[perm]
            self.is_sorted = True

        self.coords = np.ascontiguousarray(coords)
        """A NumPy array of shape (N, 3), storing the Cartesian coordinates of N grid points (in Bohrs)."""
        self.weights = np.ascontiguousarray(weights)
        """A NumPy array of length N, storing the quadrature weights corresponding to each grid point."""
        self.atom_idx = np.ascontiguousarray(atom_idx)
        """A NumPy integer array of length N with the index of the atom each grid point belongs to."""

        if verbose:
            print(self.describe(), flush=True)

    # ------------------------------------------------------------------
    @staticmethod
    def _symbol(charge):
        charge = int(charge)
        return Data.elementSymbols[charge] if 0 <= charge < len(Data.elementSymbols) else str(charge)

    @staticmethod
    def _parse_points_per_element(points_per_element):
        """``points_per_element`` keyed by element symbol or charge -> ``{charge: (n_rad, n_ang)}``."""
        overrides = {}
        if points_per_element:
            for key, value in points_per_element.items():
                if isinstance(key, str):
                    symbol = key.strip().capitalize()
                    if symbol.lower() in ('ghost', 'gh', 'x'):
                        z = 0
                    elif symbol in Data.elementSymbols:
                        z = Data.elementSymbols.index(symbol)
                    else:
                        raise ValueError("Unknown element '" + str(key) + "' in points_per_element.")
                else:
                    z = int(key)
                overrides[z] = (int(value[0]), int(value[1]))
        return overrides

    def _record_element_points(self, charges, level, overrides):
        """Fill ``element_points`` with the (n_rad, n_ang) actually used for every element of the molecule."""
        for z in charges:
            z = int(z)
            symbol = self._symbol(z)
            if symbol in self.element_points:
                continue
            n_rad, n_ang = overrides.get(z, (None, None))
            self.element_points[symbol] = (n_rad if n_rad is not None else radial_points_for_level(z, level),
                                           n_ang if n_ang is not None else angular_points_for_level(z, level))

    def _build_treutler(self, atm_coords, charges, level, pruning, size_adjustment, points_per_element):
        overrides = self._parse_points_per_element(points_per_element)
        self._record_element_points(charges, level, overrides)

        coords = []
        vol = []
        atom_idx = []
        for ia in range(atm_coords.shape[0]):
            z = int(charges[ia])
            n_rad, n_ang = overrides.get(z, (None, None))
            c, v = single_atom_grid(z, level=level, pruning=pruning, n_rad=n_rad, n_ang=n_ang)
            coords.append(c + atm_coords[ia])
            vol.append(v)
            atom_idx.append(np.full(v.shape[0], ia, dtype=np.int64))
        coords = np.vstack(coords)
        vol = np.hstack(vol)
        atom_idx = np.hstack(atom_idx)

        a_table = size_adjustment_table(charges, size_adjustment)
        nthreads_before = numba.get_num_threads()
        try:
            numba.set_num_threads(max(1, min(self.ncores, numba.config.NUMBA_NUM_THREADS)))
            factors = becke_partition_weights(coords, atom_idx, atm_coords, a_table)
        finally:
            numba.set_num_threads(nthreads_before)
        return coords, vol * factors, atom_idx

    # ------------------------------------------------------------------
    def _build_treutler_gpu(self, atm_coords, charges, level, pruning, size_adjustment, points_per_element, sort):
        """The 'treutler' build of :meth:`_build_treutler` on the GPU (see :mod:`pyfock.Grids_cupy`).

        The box grouping is done on the device too when ``sort`` is set, so the returned arrays are already
        ordered. Raises ``Grids_cupy.GridsGPUError`` when the GPU cannot be used.
        """
        from . import Grids_cupy
        overrides = self._parse_points_per_element(points_per_element)
        self._record_element_points(charges, level, overrides)
        return Grids_cupy.build_treutler_grid_cupy(atm_coords, charges, level=level, pruning=pruning,
                                                   size_adjustment=size_adjustment, overrides=overrides,
                                                   sort=sort)

    # ------------------------------------------------------------------
    def _build_numgrid(self, mol, basis, atm_coords, charges, level, preset, radial_precision, angular_points, hardness):
        if basis is None:
            # The LMG radial grids need basis-set exponents; the large def2-QZVP basis gives grids that
            # extend far enough for any calculation basis and do not change with it.
            from .Basis import Basis as _Basis
            basis = _Basis(mol, {'all': _Basis.load(mol=mol, basis_name='def2-QZVP')})
        # Only used by numgrid's (disabled) basis-set-exchange lookup.
        try:
            basis_set_name = basis.basisSet.splitlines()[0]
        except Exception:
            basis_set_name = 'def2-QZVP'

        if radial_precision is None:
            radial_precision = 1.0e-13 if preset == 'dense' else RADIAL_PRECISION[level]
        self.radial_precision = float(radial_precision)

        center_coordinates_bohrs = [(float(c[0]), float(c[1]), float(c[2])) for c in atm_coords]
        proton_charges = [int(z) for z in charges]
        num_centers = atm_coords.shape[0]

        # (min, max) angular points of every atom
        if angular_points is not None:
            angular = [(int(angular_points[0]), int(angular_points[1]))] * num_centers
        elif preset == 'compact':
            angular = [preset_angular_points(z, level) for z in proton_charges]
        else:
            angular = [(min_ang(z, level), max_ang(z, level)) for z in proton_charges]
        for z, ang in zip(proton_charges, angular):
            self.angular_points.setdefault(self._symbol(z), ang)

        # Grids are generated atom-by-atom (numgrid releases the GIL and parallelises the Becke partitioning itself).
        output = Parallel(n_jobs=self.ncores, backend='threading', require='sharedmem')(
            delayed(genGridiNewAPI)(self.radial_precision, proton_charges, center_coordinates_bohrs, basis_set_name,
                                    center_index, level, basis.alpha_min, basis.alpha_max, angular[center_index], hardness)
            for center_index in range(num_centers))
        coords = np.vstack([out[0] for out in output])
        weights = np.hstack([out[1] for out in output])
        atom_idx = np.hstack([np.full(out[1].shape[0], ia, dtype=np.int64) for ia, out in enumerate(output)])
        for ia in range(num_centers):
            symbol = self._symbol(proton_charges[ia])
            if symbol not in self.radial_points:
                r, _ = numgrid.radial_grid_lmg(basis.alpha_min[ia], basis.alpha_max[ia], self.radial_precision, proton_charges[ia])
                self.radial_points[symbol] = len(r)
        return coords, weights, atom_idx

    # ------------------------------------------------------------------
    def describe(self):
        """One-line description of the grid (scheme, level, points per element, total number of points)."""
        if self.scheme == 'treutler':
            prune_text = ('region-wise angular pruning' if isinstance(self.pruning, str) else
                          'no pruning' if self.pruning is None else 'custom pruning')
            adjust = str(self.size_adjustment).lower()
            adjust_text = ('Treutler' if adjust == 'treutler' else 'Becke' if adjust == 'becke' else 'no') + ' atomic-size adjustment'
            sizes = ', '.join(sym + ' (' + str(n[0]) + ', ' + str(n[1]) + ')' for sym, n in self.element_points.items())
            text = ('Grids: Treutler-Ahlrichs radial + Lebedev angular grids, ' + prune_text
                    + ', Becke partitioning with ' + adjust_text + ' (' + ('GPU' if self.use_gpu else 'CPU')
                    + '); level ' + str(self.level) + '; (n_rad, n_ang) per element: ' + sizes)
        else:
            per_element = ', '.join(sym + ' (' + str(self.radial_points.get(sym, '?')) + ', ' + str(ang[0]) + '-' + str(ang[1]) + ')'
                                    for sym, ang in self.angular_points.items())
            text = ('Grids (numgrid): preset ' + self.preset + ', level ' + str(self.level) + ', radial precision '
                    + '%.0e' % self.radial_precision + '; radial points and min-max angular points per element: '
                    + per_element)
        return text + '; ' + str(self.coords.shape[0]) + ' points.'

    @staticmethod
    def sort_by_boxes(coords, weights, atm_coords, *arrays, box_size=1.2):
        """Group grid points box by box (see :func:`box_grouping_order`).

        Returns the reordered ``coords``, ``weights`` and any additional per-point ``arrays``.
        """
        perm = box_grouping_order(atm_coords, coords, box_size=box_size)
        result = [np.ascontiguousarray(np.asarray(coords)[perm]), np.ascontiguousarray(np.asarray(weights)[perm])]
        for array in arrays:
            result.append(np.ascontiguousarray(np.asarray(array)[perm]))
        return tuple(result)

    def sort(self):
        """Group the points of this grid box by box in place (no-op if already sorted)."""
        if self.is_sorted or self.coords.shape[0] == 0:
            return self
        atm_coords = np.asarray(self.mol.coordsBohrs, dtype=np.float64).reshape(-1, 3)
        self.coords, self.weights, self.atom_idx = Grids.sort_by_boxes(self.coords, self.weights, atm_coords, self.atom_idx)
        self.is_sorted = True
        return self

    def prune_by_mask(self, keep):
        """Keep only the points where the boolean array ``keep`` is True (coords, weights and atom_idx)."""
        keep = np.asarray(keep, dtype=bool)
        self.coords = np.ascontiguousarray(self.coords[keep])
        self.weights = np.ascontiguousarray(self.weights[keep])
        if getattr(self, 'atom_idx', None) is not None and self.atom_idx.shape[0] == keep.shape[0]:
            self.atom_idx = np.ascontiguousarray(self.atom_idx[keep])
        return self

    @property
    def size(self):
        """Number of grid points."""
        return int(self.coords.shape[0])
