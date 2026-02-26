# Libraries
from ase import Atoms
from ase.calculators.siesta import Siesta
from ase.units import Ry
import os
import sisl as si
import numpy as np
import pandas as pd
from itertools import product
from src.cleanfiles import cleanFiles

def run_siesta(perovskite, xcf='PBEsol', basis='DZP', EnergyShift=0.01, SplitNorm=0.15,
               MeshCutoff=300, kgrid=(5, 5, 5), dir='results/bulk/basis'):
    """Function to run a single Siesta self-consistent calculation
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
    energy = atoms.get_potential_energy()
    return energy

def get_enthalpy(formula, dir):
    """Function to read enthalpy from Siesta output files.
    Parameters:
    - formula: Chemical formula of the material for which enthalpy is to be read.
    - dir: Directory where the Siesta output files are located.
    Returns:
    - Enthalpy value (in eV).
    """
    # Read basis enthalpy from file
    with open(os.path.join(dir, f'{formula}.BASIS_ENTHALPY'), 'r') as file:
        lines = file.readlines()
    enthalpy = float(lines[0].split()[-1])
    return enthalpy

def get_total_force(formula, dir):
    """Function to read total force from Siesta output files.
    Parameters:
    - formula: Chemical formula of the material for which total force is to be read.
    - dir: Directory where the Siesta output files are located.
    Returns:
    - Total force (in eV/Å).
    """
    sile = si.get_sile(os.path.join(dir, f'{formula}.FA'))
    forces = sile.read_force()
    return np.linalg.norm(forces)

def get_bandgap(formula, dir):
    """Function to read bandgap from Siesta output files.
    Parameters:
    - formula: Chemical formula of the material for which bandgap is to be read.
    - dir: Directory where the Siesta output files are located.
    Returns:
    - Indirect bandgap (in eV).
    """
    # Read eigenvalues and Fermi level from SIESTA output files
    sile = si.get_sile(os.path.join(dir, f'{formula}.EIG'))
    eig = sile.read_data()
    Ef = sile.read_fermi_level()
    # Remove spin dimension and derermine number of bands
    eig = eig[0]
    nbands = eig.shape[1]
    # Shift by Fermi level
    eig -= Ef
    # Determine number of occupied bands (2 spins and 2 bands)
    n_occ = round(nbands/4)
    # Compute indirect gap
    VBM = np.max(eig[:, n_occ-1])
    CBM = np.min(eig[:, n_occ])
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
    dir = f'results/bulk/{formula}/basis'

    if os.path.exists(os.path.join(dir, 'basisopt.csv')):
        df = pd.read_csv(os.path.join(dir, 'basisopt.csv'))
    else:
        df = pd.DataFrame(columns=['EnergyShift', 'SplitNorm', 'Energy', 'Enthalpy'])

    for shift, split in product(shifts, splits):
        # Check if results have been obtained
        if ((df['EnergyShift'] == shift) & (df['SplitNorm'] == split)).any():
            print(f"EnergyShift={shift} Ry and SplitNorm={split} is in the DataFrame. Skipping.")
        else:
            # Get energy and enthalpy from SIESTA
            energy = run_siesta(perovskite, EnergyShift=shift, SplitNorm=split, dir=dir,
                                MeshCutoff=800, kgrid=(10, 10, 10))
            enthalpy = get_enthalpy(formula, dir)
            force = get_total_force(formula, dir)
            bandgap = get_bandgap(formula, dir)
            # Append results
            row = {
                "EnergyShift": shift,
                "SplitNorm": split,
                "Energy": energy,
                "Enthalpy": enthalpy,
                "TotalForce": force,
                "Bandgap": bandgap
            }
            # Create new dataframe
            df_new = pd.DataFrame([row])
            # Update old datafrem with new results
            df = pd.concat([df, df_new], ignore_index=True)
            # Save new results
            df.to_csv(os.path.join(dir, 'basisopt.csv'), index=False)
    # Clean directory of SIESTA calculations
    cleanFiles(directory=dir, confirm=False)

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
    dir = f'results/bulk/{formula}/grid'

    if os.path.exists(os.path.join(dir, 'gridconv.csv')):
        df = pd.read_csv(os.path.join(dir, 'gridconv.csv'))
    else:
        df = pd.DataFrame(columns=['MeshCutoff', 'kgrid', 'Energy', 'Enthalpy'])

    for mc, kp in product(meshcuts, kpoints):
        # Check if results have been obtained
        if ((df['MeshCutoff'] == mc) & (df['kgrid'] == f"({kp}, {kp}, {kp})")).any():
            print(f"MeshCutoff={mc} and kgrid=({kp}, {kp}, {kp}) is in the DataFrame. Skipping.")
        else:
            # Get energy and enthalpy from SIESTA
            energy = run_siesta(perovskite, EnergyShift=0.001, SplitNorm=0.1, dir=dir,
                                MeshCutoff=mc, kgrid=(kp, kp, kp))
            enthalpy = get_enthalpy(formula, dir)
            force = get_total_force(formula, dir)
            bandgap = get_bandgap(formula, dir)
            # Append results
            row = {
                "MeshCutoff": mc,
                "kgrid": f"({kp}, {kp}, {kp})",
                "Energy": energy,
                "Enthalpy": enthalpy,
                "TotalForce": force,
                "Bandgap": bandgap
            }
            # Create new dataframe
            df_new = pd.DataFrame([row])
            # Update old datafrem with new results
            df = pd.concat([df, df_new], ignore_index=True)
            # Save new results
            df.to_csv(os.path.join(dir, 'gridconv.csv'), index=False)
    # Clean directory of SIESTA calculations
    cleanFiles(directory=dir, confirm=False)