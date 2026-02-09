# Importing packages and modules
from ase.io import read
from ase.calculators.siesta import Siesta
from ase import Atoms
from ase.units import Ry
from ase.parallel import parprint
import os

def calculate_bands(atoms, xcf='PBE', basis='DZP', shift=0.01, split=0.15, cutoff=200, kmesh=[5, 5, 5]):
    """Function to calculate band structure and PDOS of a bulk structure using SIESTA.
    Parameters:
    - atoms: ASE Atoms object representing the relaxed bulk structure.
    - xcf: Exchange-correlation functional to be used (default is 'PBE').
    - basis: Basis set to be used (default is 'DZP').
    - shift: Energy shift in Ry (default is 0.01 Ry).
    - split: Split norm for basis functions (default is 0.15).
    - cutoff: Mesh cutoff in Ry (default is 200 Ry).
    - kmesh: K-point mesh as a list (default is [5, 5, 5]).
    Returns:
    - None. The function performs band structure calculation and saves the data to a file.
    """
    cwd = os.getcwd()
    symbols = atoms.symbols

    # Calculation parameters in a dictionary
    calc_params = {
        'label': f'{symbols}',
        'xc': xcf,
        'basis_set': basis,
        'mesh_cutoff': cutoff * Ry,
        'energy_shift': shift * Ry,
        'kpts': kmesh,
        'directory': 'bulk/bandstructure/',
        'pseudo_path': cwd + '/pseudos'
    }
    # fdf arguments in a dictionary
    fdf_args = {
'PAO.BasisSize': basis,
'PAO.SplitNorm': split,
'BandLinesScale': 'ReciprocalLatticeVectors',
'%block BandLines': '''
1 0.000 0.000 0.000 \Gamma
60 0.500 0.000 0.000 X
60 0.500 0.500 0.500 R
60 0.000 0.500 0.000 M
60 0.000 0.000 0.000 \Gamma
60 0.500 0.500 0.500 R
%endblock BandLines''',
'BandLinesScale': 'ReciprocalLatticeVectors',
'%block PDOS.kgrid_Monkhorst_Pack': '''
40  0  0  0.0
 0 40  0  0.0
 0  0 40  0.0
%endblock PDOS.kgrid_Monkhorst_Pack''',
'%block Projected.DensityOfStates': '''
-20.00 15.00 0.200 500 eV
%endblock Projected.DensityOfStates'''
    }

    # Set up the Siesta calculator and attach it to the atoms object
    calc = Siesta(**calc_params, fdf_arguments=fdf_args)
    atoms.calc = calc
    atoms.get_potential_energy()






def calcPDOS(atoms, xcf='PBE', basis='DZP', cutoff=200, shift=0.01, kmesh=[10,10,10]):
    """Function to calculate projected density of states (PDOS) of a bulk structure using SIESTA.
    Parameters:
    - atoms: ASE Atoms object representing the relaxed bulk structure.
    - xcf: Exchange-correlation functional to be used (default is 'PBE').
    - basis: Basis set to be used (default is 'DZP').
    - cutoff: Mesh cutoff in Ry (default is 200 Ry).
    - shift: Energy shift in Ry (default is 0.01 Ry).
    - kmesh: K-point mesh as a list (default is [10, 10, 10]).
    Returns:
    - None. The function performs PDOS calculation and saves the data to a file.
    """
    project_root = Path.home() / "projects" / "thesis"
    # Calculation parameters in a dictionary
    calc_params = {
        'label': f'{atoms.symbols}_PDOS',
        'xc': xcf,
        'basis_set': basis,
        'mesh_cutoff': cutoff * Ry,
        'energy_shift': shift * Ry,
        'kpts': kmesh,
        'directory': 'bulk/bandstructure/',
        'pseudo_path': str(project_root / "pseudos"),
    }
    fdf_args = {
'BandLinesScale': 'ReciprocalLatticeVectors',
'%block Projected.DensityOfStates': '''
-20.00 15.00 0.200 500 eV
%endblock Projected.DensityOfStates'''
    }
    
    # Set up the Siesta calculator and attach it to the atoms object
    calc = Siesta(**calc_params,
                  fdf_arguments=fdf_args)
    atoms.calc = calc
    # Run the calculation
    atoms.get_potential_energy()