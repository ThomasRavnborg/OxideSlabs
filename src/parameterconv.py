# Libraries
from ase import Atoms
from ase.calculators.siesta import Siesta
from ase.units import Ry
import os
import sisl as si
import numpy as np
import pandas as pd
from itertools import product

def run_siesta(perovskite, xcf='PBEsol', basis='DZP', EnergyShift=0.01, SplitNorm=0.15,
               MeshCutoff=300, kgrid=(5, 5, 5), dir='results/bulk/basis'):
    """Function to run Siesta calculation on atoms object.
    Parameters:
    - perovskite: Custom object representing the structure to be relaxed.
    - bulk: Boolean indicating whether the structure is bulk (True) or slab (False) (default is True).
    - xcf: Exchange-correlation functional to be used (default is 'PBEsol').
    - basis: Basis set to be used (default is 'DZP').
    - EnergyShift: Energy shift in Ry (default is 0.01 Ry).
    - SplitNorm: Split norm for basis functions (default is 0.15).
    - MeshCutoff: Mesh cutoff in Ry (default is 300 Ry).
    - kgrid: K-point mesh as a tuple (default is (5, 5, 5)).
    - dir: Directory where results will be saved (default is 'results/bulk/basis').
    Returns:
    - None. The function runs the calculation and outputs files in the specified directory.
    """
    cwd = os.getcwd()

    formula = perovskite.formula
    atoms = perovskite.atoms

    # Calculation parameters in a dictionary
    calc_params = {
        'label': f'{formula}',
        'xc': xcf,
        'basis_set': basis,
        'mesh_cutoff': MeshCutoff * Ry,
        'energy_shift': EnergyShift * Ry,
        'kpts': kgrid,
        'directory': dir,
        'pseudo_path': os.path.join(cwd, f'pseudos/{xcf}')
    }

    # FDF arguments in a dictionary
    fdf_args = {
        'PAO.BasisSize': basis,
        'PAO.SplitNorm': SplitNorm,
        'UseTreeTimer': True
    }
    
    # Set up the Siesta calculator and attach it to the atoms object
    calc = Siesta(**calc_params, fdf_arguments=fdf_args)
    atoms.calc = calc
    # Run the calculation
    atoms.get_potential_energy()

def get_enthalpy(formula, dir):
    """Function to read enthalpy from Siesta output files.
    Parameters:
    - formula: Chemical formula of the material for which enthalpy is to be read.
    Returns:
    - Enthalpy value (in eV).
    """
    # Read basis enthalpy from file
    with open(os.path.join(dir, f'{formula}.BASIS_ENTHALPY'), 'r') as file:
        lines = file.readlines()
    enthalpy = float(lines[0].split()[-1])
    return enthalpy

def get_maxforce(formula, dir):
    """Function to read maximum force from Siesta output files.
    Parameters:
    - formula: Chemical formula of the material for which maximum force is to be read.
    Returns:
    - Maximum force (in eV/Å).
    """
    sile = si.get_sile(os.path.join(dir, f'{formula}.FA'))
    forces = sile.read_force()
    return np.max(np.abs(forces))

def get_bandgap(formula, dir):
    """Function to read bandgap from Siesta output files.
    Parameters:
    - formula: Chemical formula of the material for which bandgap is to be read.
    Returns:
    - Bandgap value at Gamma (in eV).
    """
    # Read eigenvalues and Fermi energy from files
    eig = si.get_sile(os.path.join(dir, f'{formula}.EIG')).read_data()
    Ef = si.io.siesta.stdoutSileSiesta(os.path.join(dir, f'{formula}.out')).read_energy()['fermi']
    # Shift eigenvalues by Fermi energy and calculate bandgap
    eig -= Ef
    eig = eig.flatten()
    VBM = eig[eig <= 0].max()
    CBM = eig[eig > 0].min()
    Eg = CBM - VBM
    return Eg

def basis_opt(perovskite, shifts, splits):
    """Function to optimize basis set parameters by running multiple Siesta calculations.
    Parameters:
    - perovskite: Custom object representing the structure to be calculated.
    - shifts: List of energy shift values to be tested.
    - splits: List of split norm values to be tested.
    Returns:
    - None. The function runs multiple calculations and saves the results to a CSV file.
    """

    formula = perovskite.formula
    dir = f'resultsold/bulk/{formula}/basis'
    rows = []
    for sh, sp in product(shifts, splits):
        try:
            run_siesta(perovskite, EnergyShift=sh, SplitNorm=sp, dir=dir)
            enthalpy = get_enthalpy(formula, dir)
            maxforce = get_maxforce(formula, dir)
            #bandgap = get_bandgap(formula, dir)
        except Exception as e:
            enthalpy = None
            maxforce = None
            #bandgap = None
        
        rows.append({
            "EnergyShift": sh,
            "SplitNorm": sp,
            "Enthalpy": enthalpy,
            "MaxForce": maxforce,
            #"Bandgap": bandgap,
        })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(dir, 'basisopt.csv'))

def grid_conv(perovskite, meshcuts, kpoints):
    """Function to optimize grid parameters by running multiple Siesta calculations.
    Parameters:
    - perovskite: Custom object representing the structure to be calculated.
    - meshcuts: List of mesh cutoff values to be tested.
    - kpoints: List of k-point mesh values to be tested.
    Returns:
    - None. The function runs multiple calculations and saves the results to a CSV file.
    """

    formula = perovskite.formula
    dir = f'resultsold/bulk/{formula}/basis'
    rows = []
    for mc, kp in product(meshcuts, kpoints):
        try:
            run_siesta(perovskite, MeshCutoff=mc, kgrid=(kp, kp, kp), dir=dir)
            enthalpy = get_enthalpy(formula, dir)
            maxforce = get_maxforce(formula, dir)
            #bandgap = get_bandgap(formula, dir)
        except Exception as e:
            enthalpy = None
            maxforce = None
            #bandgap = None

        rows.append({
            "MeshCutoff": mc,
            "kgrid": (kp, kp, kp),
            "Enthalpy": enthalpy,
            "MaxForce": maxforce,
            #"Bandgap": bandgap
        })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(dir, 'gridopt.csv'))