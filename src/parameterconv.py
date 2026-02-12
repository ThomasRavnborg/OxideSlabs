# Libraries
from ase import Atoms
from ase.calculators.siesta import Siesta
from ase.units import Ry
import os
import sisl as si
import numpy as np
import pandas as pd
from itertools import product

def run_siesta(atoms, xcf='PBE', basis='DZP', shift=0.01, split=0.15, cutoff=200, kmesh=[5, 5, 5]):
    """Function to run Siesta calculation on atoms object.
    Parameters:
    - atoms: ASE Atoms object representing the structure to be calculated.
    - xcf: Exchange-correlation functional to be used (default is 'PBE').
    - basis: Basis set to be used (default is 'DZP').
    - shift: Energy shift in Ry (default is 0.01 Ry).
    - split: Split norm for basis functions (default is 0.15).
    - cutoff: Mesh cutoff in Ry (default is 200 Ry).
    - kmesh: K-point mesh as a list (default is [5, 5, 5]).
    Returns:
    - None. The function runs the calculation and outputs files in the specified directory.
    """
    cwd = os.getcwd()
    dir = 'results/bulk/basis/'

    # Calculation parameters in a dictionary
    calc_params = {
        'label': f'{atoms.symbols}',
        'xc': xcf,
        'basis_set': basis,
        'mesh_cutoff': cutoff * Ry,
        'energy_shift': shift * Ry,
        'kpts': kmesh,
        'directory': dir,
        'pseudo_path': cwd + '/pseudos'
    }

    # FDF arguments in a dictionary
    fdf_args = {
        'PAO.BasisSize': basis,
        'PAO.SplitNorm': split
    }
    
    # Set up the Siesta calculator and attach it to the atoms object
    calc = Siesta(**calc_params, fdf_arguments=fdf_args)
    atoms.calc = calc
    # Run the calculation
    atoms.get_potential_energy()

def get_enthalpy(formula):
    """Function to read enthalpy from Siesta output files.
    Parameters:
    - formula: Chemical formula of the material for which enthalpy is to be read.
    Returns:
    - Enthalpy value (in eV).
    """
    dir = 'results/bulk/basis/'
    # Read basis enthalpy from file
    with open(f'{dir}{formula}.BASIS_ENTHALPY', 'r') as file:
        lines = file.readlines()
    enthalpy = float(lines[0].split()[-1])
    return enthalpy

def get_maxforce(formula):
    """Function to read maximum force from Siesta output files.
    Parameters:
    - formula: Chemical formula of the material for which maximum force is to be read.
    Returns:
    - Maximum force (in eV/Å).
    """
    dir = 'results/bulk/basis/'
    sile = si.get_sile(f'{dir}{formula}.FA')
    forces = sile.read_force()
    return np.max(np.abs(forces))

def get_bandgap(formula):
    """Function to read bandgap from Siesta output files.
    Parameters:
    - formula: Chemical formula of the material for which bandgap is to be read.
    Returns:
    - Bandgap value at Gamma (in eV).
    """
    dir = 'results/bulk/basis/'
    # Read eigenvalues and Fermi energy from files
    eig = si.get_sile(f'{dir}{formula}.EIG').read_data()
    Ef = si.io.siesta.stdoutSileSiesta(f'{dir}{formula}.out').read_energy()['fermi']
    # Shift eigenvalues by Fermi energy and calculate bandgap
    eig -= Ef
    eig = eig.flatten()
    VBM = eig[eig <= 0].max()
    CBM = eig[eig > 0].min()
    Eg = CBM - VBM
    return Eg

def basis_opt(atoms, shifts, splits):
    """Function to optimize basis set parameters by running multiple Siesta calculations.
    Parameters:
    - atoms: ASE Atoms object representing the structure to be calculated.
    - shifts: List of energy shift values to be tested.
    - splits: List of split norm values to be tested.
    Returns:
    - None. The function runs multiple calculations and saves the results to a CSV file.
    """
    dir = 'results/bulk/basis/'
    rows = []
    for sh, sp in product(shifts, splits):
        try:
            run_siesta(atoms, shift=sh, split=sp)
            enthalpy = get_enthalpy(atoms.symbols)
            maxforce = get_maxforce(atoms.symbols)
            bandgap = get_bandgap(atoms.symbols)
        except Exception as e:
            enthalpy = None
            maxforce = None
            bandgap = None
        
        rows.append({
            "Shift": sh,
            "Split": sp,
            "Enthalpy": enthalpy,
            "MaxForce": maxforce,
            "Bandgap": bandgap,
        })

    df = pd.DataFrame(rows)
    df.to_csv(f'{dir}basisopt.csv')

def grid_conv(atoms, meshcuts, kpoints):
    """Function to optimize grid parameters by running multiple Siesta calculations.
    Parameters:
    - atoms: ASE Atoms object representing the structure to be calculated.
    - meshcuts: List of mesh cutoff values to be tested.
    - kpoints: List of k-point mesh values to be tested.
    Returns:
    - None. The function runs multiple calculations and saves the results to a CSV file.
    """
    dir = 'results/bulk/grid/'
    rows = []
    for mc, kp in product(meshcuts, kpoints):
        try:
            run_siesta(atoms, cutoff=mc, kmesh=[kp, kp, kp])
            enthalpy = get_enthalpy(atoms.symbols)
            maxforce = get_maxforce(atoms.symbols)
            bandgap = get_bandgap(atoms.symbols)
        except Exception as e:
            enthalpy = None
            maxforce = None
            bandgap = None

        rows.append({
            "MeshCutoff": mc,
            "KPoints": kp,
            "Enthalpy": enthalpy,
            "MaxForce": maxforce,
            "Bandgap": bandgap
        })

    df = pd.DataFrame(rows)
    df.to_csv(f'{dir}gridopt.csv')