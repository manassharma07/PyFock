# DATA.py
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
#Class to store element properties
class Data:
    """
    A container class holding atomic and basis set metadata used in calculations using PyFock.

    This includes element symbols, names, covalent and atomic radii, atomic masses, shell definitions,
    and other constants and mappings relevant for electronic structure calculations.

    Note: Index 0 in all element-related lists corresponds to a ghost atom (used in some basis set techniques).
    """
    #IN ALL THE DATA THE FIRST ELEMENT CORRESPONDS TO GHOST ATOM
    elementSymbols = ["Ghost","H","He","Li","Be","B","C","N","O","F","Ne","Na","Mg","Al","Si","P","S","Cl","Ar","K","Ca","Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn","Ga","Ge","As","Se","Br","Kr","Rb","Sr","Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd","In","Sn","Sb","Te","I","Xe","Cs","Ba","La","Ce","Pr","Nd","Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu","Hf","Ta","W","Re","Os","Ir","Pt","Au","Hg","Tl","Pb","Bi","Po","At","Rn","Fr","Ra","Ac","Th","Pa","U","Np","Pu","Am","Cm","Bk","Cf","Es","Fm","Md","No","Lr","Rf","Db","Sg","Bh","Hs","Mt","Ds","Rg","Cn","Nh","Fl","Mc","Lv","Ts","Og"]
    """List of chemical element symbols. Index 0 is reserved for the ghost atom."""
    elementName = ["Ghost","Hydrogen","Helium","Lithium","Beryllium","Boron","Carbon","Nitrogen","Oxygen","Fluorine","Neon","Sodium","Magnesium","Aluminum","Silicon","Phosphorus","Sulfur","Chlorine","Argon","Potassium","Calcium","Scandium","Titanium","Vanadium","Chromium","Manganese","Iron","Cobalt","Nickel","Copper","Zinc","Gallium","Germanium","Arsenic","Selenium","Bromine","Krypton","Rubidium","Strontium","Yttrium","Zirconium","Niobium","Molybdenum","Technetium","Ruthenium","Rhodium","Palladium","Silver","Cadmium","Indium","Tin","Antimony","Tellurium","Iodine","Xenon","Cesium","Barium","Lanthanum","Cerium","Praseodymium","Neodymium","Promethium","Samarium","Europium","Gadolinium","Terbium","Dysprosium","Holmium","Erbium","Thulium","Ytterbium","Lutetium","Hafnium","Tantalum","Tungsten","Rhenium","Osmium","Iridium","Platinum","Gold","Mercury","Thallium","Lead","Bismuth","Polonium","Astatine","Radon","Francium","Radium","Actinium","Thorium","Protactinium","Uranium","Neptunium","Plutonium","Americium","Curium","Berkelium","Californium","Einsteinium","Fermium","Mendelevium","Nobelium","Lawrencium","Rutherfordium","Dubnium","Seaborgium","Bohrium","Hassium","Meitnerium","Darmstadtium","Roentgenium","Copernicium","Nihonium","Flerovium","Moscovium","Livermorium","Tennessine","Oganesson"]
    """Full names of the elements corresponding to `elementSymbols`."""
    covalentRadius = ["0","0.37","0.32","1.34","0.9","0.82","0.77","0.75","0.73","0.71","0.69","1.54","1.3","1.18","1.11","1.06","1.02","0.99","0.97","1.96","1.74","1.44","1.36","1.25","1.27","1.39","1.25","1.26","1.21","1.38","1.31","1.26","1.22","1.19","1.16","1.14","1.1","2.11","1.92","1.62","1.48","1.37","1.45","1.56","1.26","1.35","1.31","1.53","1.48","1.44","1.41","1.38","1.35","1.33","1.3","2.25","1.98","1.69","N/A","N/A","N/A","N/A","N/A","N/A","N/A","N/A","N/A","N/A","N/A","N/A","N/A","1.6","1.5","1.38","1.46","1.59","1.28","1.37","1.28","1.44","1.49","1.48","1.47","1.46","N/A","N/A","1.45","N/A","N/A","N/A","N/A","N/A","N/A","N/A","N/A","N/A","N/A","N/A","N/A","N/A","N/A","N/A","N/A","N/A","N/A","N/A","N/A","N/A","N/A","N/A","N/A","N/A","N/A","N/A","N/A","N/A","N/A","N/A","N/A"]
    """Covalent radii of elements in Ångströms. Some entries may be "N/A" if unknown or undefined."""
    massNumber = ["0","1.007944","4.0026022","6.9412","9.0121823","10.8117","12.01078","14.00672","15.99943","18.99840325","20.17976","22.989769282","24.30506","26.98153868","28.08553","30.9737622","32.0655","35.4532","39.9481","39.09831","40.0784","44.9559126","47.8671","50.94151","51.99616","54.9380455","55.8452","58.9331955","58.69344","63.5463","65.382","69.7231","72.641","74.921602","78.963","79.9041","83.7982","85.46783","87.621","88.905852","91.2242","92.906382","95.962","[98]","101.072","102.905502","106.421","107.86822","112.4118","114.8183","118.7107","121.7601","127.603","126.904473","131.2936","132.90545192","137.3277","138.905477","140.1161","140.907652","144.2423","[145]","150.362","151.9641","157.253","158.925352","162.5001","164.930322","167.2593","168.934212","173.0545","174.96681","178.492","180.947882","183.841","186.2071","190.233","192.2173","195.0849","196.9665694","200.592","204.38332","207.21","208.980401","[209]","[210]","[222]","[223]","[226]","[227]","232.038062","231.035882","238.028913","[237]","[244]","[243]","[247]","[247]","[251]","[252]","[257]","[258]","[259]","[262]","[267]","[268]","[271]","[272]","[270]","[276]","[281]","[280]","[285]","[284]","[289]","[288]","[293]","[294]","[294]"]
    """Atomic masses of elements in unified atomic mass units (u). Values in brackets indicate unstable isotopes."""
    atomicRadius = ["0.4","0.53","0.31","1.67","1.12","0.87","0.67","0.56","0.73","0.42","0.38","1.9","1.45","1.18","1.11","0.98","0.88","0.79","0.71","2.43","1.94","1.84","1.76","1.71","1.66","1.61","1.56","1.52","1.49","1.45","1.42","1.36","1.25","1.14","1.03","0.94","0.88","2.65","2.19","2.12","2.06","1.98","1.9","1.83","1.78","1.73","1.69","1.65","1.61","1.56","1.45","1.33","1.23","1.15","1.08","2.98","2.53","1.95","1.85","2.47","2.06","2.05","2.38","2.31","2.33","2.25","2.28","2.26","2.26","2.22","2.22","2.17","2.08","2","1.93","1.88","1.85","1.8","1.77","1.74","1.71","1.56","1.54","1.43","1.35","1.27","1.2","N/A","N/A","1.95","1.8","1.8","1.75","1.75","1.75","1.75","N/A","N/A","N/A","N/A","N/A","N/A","N/A","N/A","N/A","N/A","N/A","N/A","N/A","N/A","N/A","N/A","N/A","N/A","N/A","N/A","N/A","N/A","N/A"]
    """Atomic radii in Ångströms, representing empirical or van der Waals sizes. "N/A" indicates missing data."""
    CPKcolorRGB = [[255,255,255],[255,255,255],[217,255,255],[204,128,255],[194,255,0],[255,181,181],[144,144,144],[48,80,248],[255,13,13],[144,224,80],[179,227,245],[171,92,242],[138,255,0],[191,166,166],[240,200,160],[255,128,0],[255,255,48],[31,240,31],[128,209,227],[143,64,212],[61,255,0],[230,230,230],[191,194,199],[166,166,171],[138,153,199],[156,122,199],[224,102,51],[240,144,160],[80,208,80],[200,128,51],[125,128,176],[194,143,143],[102,143,143],[189,128,227],[255,161,0],[166,41,41],[92,184,209],[112,46,176],[0,255,0],[148,255,255],[148,224,224],[115,194,201],[84,181,181],[59,158,158],[36,143,143],[10,125,140],[0,105,133],[192,192,192],[255,217,143],[166,117,115],[102,128,128],[158,99,181],[212,122,0],[148,0,148],[66,158,176],[87,23,143],[0,201,0],[112,212,255],[255,255,199],[217,255,199],[199,255,199],[163,255,199],[143,255,199],[97,255,199],[69,255,199],[48,255,199],[31,255,199],[0,255,156],[0,230,117],[0,212,82],[0,191,56],[0,171,36],[77,194,255],[77,166,255],[33,148,214],[38,125,171],[38,102,150],[23,84,135],[208,208,224],[255,209,35],[184,184,208],[166,84,77],[87,89,97],[158,79,181],[171,92,0],[117,79,69],[66,130,150],[66,0,102],[0,125,0],[112,171,250],[0,186,255],[0,161,255],[0,143,255],[0,128,255],[0,107,255],[84,92,242],[120,92,227],[138,79,227],[161,54,212],[179,31,212],[179,31,186],[179,13,166],[189,13,135],[199,0,102],[204,0,89],[209,0,79],[217,0,69],[224,0,56],[230,0,46],[235,0,38]]
    """RGB values used for coloring atoms in molecular visualizations (e.g., CPK coloring scheme)."""
    au2eVFactor = 27.211324570273
    """Conversion factor from au (Hartree units) to eV (1 au = 27.211324570273 eV)."""
    Bohr2AngsFactor = 0.52917721092
    """Conversion factor from Bohr to Ångström (1 Bohr = 0.52917721092 Å)."""
    Angs2BohrFactor = 1.88972612456 # Including more digits than currently used somehow increases the error drastically?!    #Previosuly used value was 1.889725989
    """Conversion factor from Ångström to Bohr (1 Å = 1.88972612456 Bohr)."""
    shell_dict = {'s':1, 'p':2, 'd':3, 'f':4, 'g':5, 'h':6, 'i':7, 'j':8}
    """Mapping of orbital angular momentum shell labels to integer quantum numbers (e.g., 's' -> 1, 'p' -> 2, etc.)."""
    # PySCF/HORTON ordering (for Cartesian GTOs https://github.com/sunqm/libcint/blob/master/doc/program_ref.pdf and https://theochem.github.io/horton/2.0.1/tech_ref_gaussian_basis.html#collected-notes-on-gaussian-basis-sets)
    # shell_lmn = {'s':[0,0,0], 'px':[1,0,0], 'py':[0,1,0], 'pz':[0,0,1], 'dxx':[2,0,0], 'dxy':[1,1,0], 'dxz':[1,0,1], 'dyy':[0,2,0], 'dyz':[0,1,1], 'dzz':[0,0,2], 'fxxx':[3,0,0], 'fxxy':[2,1,0], 'fxxz':[2,0,1], 'fxyy':[1,2,0], 'fxyz':[1,1,1], 'fxzz':[1,0,2], 'fyyy':[0,3,0], 'fyyz':[0,2,1], 'fyzz':[0,1,2], 'fzzz':[0,0,3], 'gxxxx':[4,0,0], 'gxxxy':[3,1,0], 'gxxxz':[3,0,1], 'gxxyy':[2,2,0], 'gxxyz':[2,1,1], 'gxxzz':[2,0,2], 'gxyyy':[1,3,0], 'gxyyz':[1,2,1], 'gxyzz':[1,1,2], 'gxzzz':[1,0,3], 'gyyyy':[0,4,0], 'gyyyz':[0,3,1], 'gyyzz':[0,2,2], 'gyzzz':[0,1,3], 'gzzzz':[0,0,4] }
    shell_lmn = {
        's': [0, 0, 0],
        'px': [1, 0, 0],
        'py': [0, 1, 0],
        'pz': [0, 0, 1],
        'dxx': [2, 0, 0],
        'dxy': [1, 1, 0],
        'dxz': [1, 0, 1],
        'dyy': [0, 2, 0],
        'dyz': [0, 1, 1],
        'dzz': [0, 0, 2],
        'fxxx': [3, 0, 0],
        'fxxy': [2, 1, 0],
        'fxxz': [2, 0, 1],
        'fxyy': [1, 2, 0],
        'fxyz': [1, 1, 1],
        'fxzz': [1, 0, 2],
        'fyyy': [0, 3, 0],
        'fyyz': [0, 2, 1],
        'fyzz': [0, 1, 2],
        'fzzz': [0, 0, 3],
        'gxxxx': [4, 0, 0],
        'gxxxy': [3, 1, 0],
        'gxxxz': [3, 0, 1],
        'gxxyy': [2, 2, 0],
        'gxxyz': [2, 1, 1],
        'gxxzz': [2, 0, 2],
        'gxyyy': [1, 3, 0],
        'gxyyz': [1, 2, 1],
        'gxyzz': [1, 1, 2],
        'gxzzz': [1, 0, 3],
        'gyyyy': [0, 4, 0],
        'gyyyz': [0, 3, 1],
        'gyyzz': [0, 2, 2],
        'gyzzz': [0, 1, 3],
        'gzzzz': [0, 0, 4],
        'hxxxxx':[5,0,0], 
        'hxxxxy':[4,1,0], 
        'hxxxxz':[4,0,1], 
        'hxxxyy':[3,2,0], 
        'hxxxyz':[3,1,1], 
        'hxxxzz':[3,0,2],
        'hxxyyy':[2,3,0], 
        'hxxyyz':[2,2,1], 
        'hxxyzz':[2,1,2], 
        'hxxzzz':[2,0,3], 
        'hxyyyy':[1,4,0], 
        'hxyyyz':[1,3,1], 
        'hxyyzz':[1,2,2], 
        'hxyzzz':[1,1,3], 
        'hxzzzz':[1,0,4],
        'hyyyyy':[0,5,0], 
        'hyyyyz':[0,4,1], 
        'hyyyzz':[0,3,2], 
        'hyyzzz':[0,2,3], 
        'hyzzzz':[0,1,4], 
        'hzzzzz':[0,0,5],
        'ixxxxxx':[6,0,0], 
        'ixxxxxy':[5,1,0], 
        'ixxxxxz':[5,0,1], 
        'ixxxxyy':[4,2,0], 
        'ixxxxyz':[4,1,1], 
        'ixxxxzz':[4,0,2],
        'ixxxyyy':[3,3,0], 
        'ixxxyyz':[3,2,1], 
        'ixxxyzz':[3,1,2], 
        'ixxxzzz':[3,0,3], 
        'ixxyyyy':[2,4,0], 
        'ixxyyyz':[2,3,1], 
        'ixxyyzz':[2,2,2], 
        'ixxyzzz':[2,1,3], 
        'ixxzzzz':[2,0,4],
        'ixyyyyy':[1,5,0], 
        'ixyyyyz':[1,4,1], 
        'ixyyyzz':[1,3,2], 
        'ixyyzzz':[1,2,3], 
        'ixyzzzz':[1,1,4], 
        'ixzzzzz':[1,0,5],
        'iyyyyyy':[0,6,0],
        'iyyyyyz':[0,5,1],
        'iyyyyzz':[0,4,2],
        'iyyyzzz':[0,3,3],
        'iyyzzzz':[0,2,4],
        'iyzzzzz':[0,1,5],
        'izzzzzz':[0,0,6],
    }
    """Dictionary mapping Cartesian Gaussian orbital labels to their exponent tuples [lx, ly, lz] (PySCF/HORTON ordering)."""
    # TURBOMOLE Ordering
    shell_lmn_tmole = {'s':[0,0,0], 'px':[1,0,0], 'py':[0,1,0], 'pz':[0,0,1], 'dxx':[2,0,0], 'dyy':[0,2,0], 'dzz':[0,0,2], 'dxy':[1,1,0], 'dxz':[1,0,1], 'dyz':[0,1,1], 'fxxx':[3,0,0], 'fxxy':[2,1,0], 'fxxz':[2,0,1], 'fxyy':[1,2,0], 'fxyz':[1,1,1], 'fxzz':[1,0,2], 'fyyy':[0,3,0], 'fyyz':[0,2,1], 'fyzz':[0,1,2], 'fzzz':[0,0,3], 'gxxxx':[4,0,0], 'gxxxy':[3,1,0], 'gxxxz':[3,0,1], 'gxxyy':[2,2,0], 'gxxyz':[2,1,1], 'gxxzz':[2,0,2], 'gxyyy':[1,3,0], 'gxyyz':[1,2,1], 'gxyzz':[1,1,2], 'gxzzz':[1,0,3], 'gyyyy':[0,4,0], 'gyyyz':[0,3,1], 'gyyzz':[0,2,2], 'gyzzz':[0,1,3], 'gzzzz':[0,0,4] }
    """Same as `shell_lmn`, but with the Cartesian basis ordering used in TURBOMOLE."""
    shell_lmn_offset = [0,1,4,10,20,35,56,84]
    """Offsets for indexing Cartesian basis function blocks for increasing angular momentum (used for shell loops)."""
    # (l+1)*(l+2)/2
    shell_degen = [1,3,6,10,15,21,28,36] # Cartesian
    """Degeneracy (number of basis functions) of Cartesian shells as a function of angular momentum."""
    #First element is ghost element remember
    elementPeriod=["1","1","1","2","2","2","2","2","2","2","2","3","3","3","3","3","3","3","3","4","4","4","4","4","4","4","4","4","4","4","4","4","4","4","4","4","4","5","5","5","5","5","5","5","5","5","5","5","5","5","5","5","5","5","5","6","6","6","6","6","6","6","6","6","6","6","6","6","6","6","6","6","6","6","6","6","6","6","6","6","6","6","6","6","6","6","6","7","7","7","7","7","7","7","7","7","7","7","7","7","7","7","7","7","7","7","7","7","7","7","7","7","7","7","7","7","7","7","7"]
    """Period (row) of the periodic table for each element. Index 0 corresponds to the ghost atom."""
    # Ground state multiplicities (2S+1) for Z=1..86
    # Period 1:  H  He
    # Period 2:  Li Be B  C  N  O  F  Ne
    # Period 3:  Na Mg Al Si P  S  Cl Ar
    # Period 4:  K  Ca Sc Ti V  Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr
    # Period 5:  Rb Sr Y  Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I  Xe
    # Period 6:  Cs Ba La Ce-Lu (lanthanides)
    #            Hf Ta W  Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn
    GROUND_STATE_MULTIPLICITY = {
        'H': 2, 'He': 1,
        'Li': 2, 'Be': 1, 'B': 2, 'C': 3, 'N': 4, 'O': 3, 'F': 2, 'Ne': 1,
        'Na': 2, 'Mg': 1, 'Al': 2, 'Si': 3, 'P': 4, 'S': 3, 'Cl': 2, 'Ar': 1,
        'K': 2, 'Ca': 1, 'Sc': 2, 'Ti': 3, 'V': 4, 'Cr': 7, 'Mn': 6,
        'Fe': 5, 'Co': 4, 'Ni': 3, 'Cu': 2, 'Zn': 1,
        'Ga': 2, 'Ge': 3, 'As': 4, 'Se': 3, 'Br': 2, 'Kr': 1,
        'Rb': 2, 'Sr': 1, 'Y': 2, 'Zr': 3, 'Nb': 6, 'Mo': 7, 'Tc': 6,
        'Ru': 5, 'Rh': 4, 'Pd': 1, 'Ag': 2, 'Cd': 1,
        'In': 2, 'Sn': 3, 'Sb': 4, 'Te': 3, 'I': 2, 'Xe': 1,
        'Cs': 2, 'Ba': 1,
        # Lanthanides (Z=57-71)
        'La': 2, 'Ce': 3, 'Pr': 4, 'Nd': 5, 'Pm': 6, 'Sm': 7, 'Eu': 8,
        'Gd': 9, 'Tb': 6, 'Dy': 5, 'Ho': 4, 'Er': 3, 'Tm': 2, 'Yb': 1,
        # 5d transition metals and 6p block (Z=71-86)
        'Lu': 2, 'Hf': 3, 'Ta': 4, 'W': 5, 'Re': 6, 'Os': 5, 'Ir': 4,
        'Pt': 3, 'Au': 2, 'Hg': 1,
        'Tl': 2, 'Pb': 3, 'Bi': 4, 'Po': 3, 'At': 2, 'Rn': 1,
    }
    """Ground state multiplicities (2S+1) for Z=1..86"""
    # GROUND_STATE_MULTIPLICITY = {
    #     # Period 1
    #     'H': 2, 'He': 1,
    #     # Period 2
    #     'Li': 2, 'Be': 1, 'B': 2, 'C': 3, 'N': 4, 'O': 3, 'F': 2, 'Ne': 1,
    #     # Period 3
    #     'Na': 2, 'Mg': 1, 'Al': 2, 'Si': 3, 'P': 4, 'S': 3, 'Cl': 2, 'Ar': 1,
    #     # Period 4
    #     'K': 2, 'Ca': 1, 'Sc': 2, 'Ti': 3, 'V': 4, 'Cr': 7, 'Mn': 6,
    #     'Fe': 5, 'Co': 4, 'Ni': 3, 'Cu': 2, 'Zn': 1,
    #     'Ga': 2, 'Ge': 3, 'As': 4, 'Se': 3, 'Br': 2, 'Kr': 1,
    #     # Period 5
    #     'Rb': 2, 'Sr': 1, 'Y': 2, 'Zr': 3, 'Nb': 6, 'Mo': 7, 'Tc': 6,
    #     'Ru': 5, 'Rh': 4, 'Pd': 1, 'Ag': 2, 'Cd': 1,
    #     'In': 2, 'Sn': 3, 'Sb': 4, 'Te': 3, 'I': 2, 'Xe': 1,
    #     # Period 6: Cs-Ba, then lanthanides La-Lu, then Hf-Hg
    #     'Cs': 2, 'Ba': 1,
    #     # Lanthanides
    #     'La': 2,   # [Xe] 5d1 6s2,          2D3/2  -> 2S+1 = 2
    #     'Ce': 1,   # [Xe] 4f1 5d1 6s2,      1G4    -> 2S+1 = 1  (ground term 1G4)
    #     'Pr': 4,   # [Xe] 4f3 6s2,          4I9/2  -> 2S+1 = 4
    #     'Nd': 5,   # [Xe] 4f4 6s2,          5I4    -> 2S+1 = 5
    #     'Pm': 6,   # [Xe] 4f5 6s2,          6H5/2  -> 2S+1 = 6
    #     'Sm': 7,   # [Xe] 4f6 6s2,          7F0    -> 2S+1 = 7
    #     'Eu': 8,   # [Xe] 4f7 6s2,          8S7/2  -> 2S+1 = 8
    #     'Gd': 9,   # [Xe] 4f7 5d1 6s2,      9D2    -> 2S+1 = 9
    #     'Tb': 6,   # [Xe] 4f9 6s2,          6H15/2 -> 2S+1 = 6
    #     'Dy': 5,   # [Xe] 4f10 6s2,         5I8    -> 2S+1 = 5
    #     'Ho': 4,   # [Xe] 4f11 6s2,         4I15/2 -> 2S+1 = 4
    #     'Er': 3,   # [Xe] 4f12 6s2,         3H6    -> 2S+1 = 3
    #     'Tm': 2,   # [Xe] 4f13 6s2,         2F7/2  -> 2S+1 = 2
    #     'Yb': 1,   # [Xe] 4f14 6s2,         1S0    -> 2S+1 = 1
    #     'Lu': 2,   # [Xe] 4f14 5d1 6s2,     2D3/2  -> 2S+1 = 2
    #     # 5d transition metals
    #     'Hf': 3,   # [Xe] 4f14 5d2 6s2,     3F2    -> 2S+1 = 3
    #     'Ta': 4,   # [Xe] 4f14 5d3 6s2,     4F3/2  -> 2S+1 = 4
    #     'W': 5,    # [Xe] 4f14 5d4 6s2,     5D0    -> 2S+1 = 5
    #     'Re': 6,   # [Xe] 4f14 5d5 6s2,     6S5/2  -> 2S+1 = 6
    #     'Os': 5,   # [Xe] 4f14 5d6 6s2,     5D4    -> 2S+1 = 5
    #     'Ir': 4,   # [Xe] 4f14 5d7 6s2,     4F9/2  -> 2S+1 = 4
    #     'Pt': 3,   # [Xe] 4f14 5d9 6s1,     3D3    -> 2S+1 = 3
    #     'Au': 2,   # [Xe] 4f14 5d10 6s1,    2S1/2  -> 2S+1 = 2
    #     'Hg': 1,   # [Xe] 4f14 5d10 6s2,    1S0    -> 2S+1 = 1
    # }
    # """Ground state multiplicities (2S+1) for Z=1..80"""