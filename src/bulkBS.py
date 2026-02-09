# Importing packages and modules
from ase.io import read
from ase.calculators.siesta import Siesta
from ase import Atoms
from ase.units import Ry
from ase.parallel import parprint
from pathlib import Path

def calcBS(atoms, xcf='PBE', basis='DZP', cutoff=200, shift=0.01, kmesh=[10,10,10]):
    """Function to calculate band structure of a bulk structure using SIESTA.
    Parameters:
    - atoms: ASE Atoms object representing the relaxed bulk structure.
    - xcf: Exchange-correlation functional to be used (default is 'PBE').
    - basis: Basis set to be used (default is 'DZP').
    - cutoff: Mesh cutoff in Ry (default is 200 Ry).
    - shift: Energy shift in Ry (default is 0.01 Ry).
    - kmesh: K-point mesh as a list (default is [10, 10, 10]).
    Returns:
    - None. The function performs band structure calculation and saves the data to a file.
    """
    project_root = Path.home() / "projects" / "thesis"
    # Calculation parameters in a dictionary
    calc_params = {
        'label': f'{atoms.symbols}_BS',
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
        '%block BandLines': '''
1 0.000 0.000 0.000 \Gamma
10 0.500 0.000 0.000 X
20 0.500 0.500 0.500 R
20 0.000 0.500 0.000 M
10 0.000 0.000 0.000 \Gamma
30 0.500 0.500 0.500 R
%endblock BandLines'''
    }

    # Set up the Siesta calculator and attach it to the atoms object
    calc = Siesta(**calc_params,
                  fdf_arguments=fdf_args)
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