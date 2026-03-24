# Libraries
from ase import Atoms
from ase.calculators.siesta import Siesta
from ase.units import Ry
import os
import re
import sisl as si
import numpy as np
import pandas as pd
from itertools import product
from src.fdfcreate import generate_basis
from src.cleanfiles import cleanFiles

def run_siesta(perovskite, xcf='PBEsol', basis='DZPp',
               EnergyShift=0.01, SplitNorm=0.15,
               MeshCutoff=1000, kgrid=(10, 10, 10),
               pseudo='PBEsol', dir='results/bulk/basis'):
    """Function to run a single Siesta self-consistent calculation
    Parameters:
    - perovskite: Custom object representing the structure to be relaxed.
    - bulk: Boolean indicating whether the structure is bulk (True) or slab (False) (default is True).
    - xcf: Exchange-correlation functional to be used (default is 'PBEsol').
    - basis: Basis set to use for the calculation (default: 'DZP').
             If basis ends with (lower-case) p, a polarization orbital will be added to the A-site (Ba)
    - EnergyShift: Energy shift in Ry (default is 0.01 Ry).
    - SplitNorm: Split norm for basis functions (default is 0.15).
    - MeshCutoff: Mesh cutoff in Ry (default is 1000 Ry).
    - kgrid: K-point mesh as a tuple (default is (10, 10, 10)).
    - pseudo: Pseudopotential to be used (default is 'PBEsol').
    - dir: Directory to save the results (default is 'results/bulk/phonons').
    Returns:
    - None. The function runs the calculation and outputs files in the specified directory.
    """
    # Define current working directory and extract information from the perovskite object
    cwd = os.getcwd()
    formula = perovskite.formula
    atoms = perovskite.atoms

    # Custom basis sets ending with 'p' are generated with the same parameters as the standard basis sets
    # However, an extra polarization (d) orbital is added to the A-site during LCAO basis generation
    if basis.endswith('p'):
        basis = basis[:-1]

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
    dir_fdf = os.path.join(cwd, dir)
    # fdf arguments in a dictionary
    fdf_args = {
        '%include': os.path.join(dir_fdf, 'basis.fdf'),
        'PAO.SplitNorm': SplitNorm
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


def get_max_force(formula, dir):
    """Function to read maximum force on an atom from Siesta output files.
    Parameters:
    - formula: Chemical formula of the material for which maximum force is to be read.
    - dir: Directory where the Siesta output files are located.
    Returns:
    - Maximum force (in eV/Å).
    """
    sile = si.get_sile(os.path.join(dir, f'{formula}.FA'))
    forces = sile.read_force()
    return np.max(np.linalg.norm(forces, axis=1))


def get_meshcutoff(formula, dir):
    # Read SIESTA .out file
    with open(os.path.join(dir, f"{formula}.out")) as f:
        text = f.read()
    # Find mesh cutoff using regular expression
    match = re.search(
        r"InitMesh: Mesh cutoff \(required, used\).*?Ry", text, re.S)
    if match:
        meshcutoff = float(match.group(0).split()[-2])
    else:
        meshcutoff = None
    return meshcutoff


def basis_opt(perovskite, shifts, splits):
    """Function to optimize basis set parameters by running multiple Siesta calculations.
    Parameters:
    - perovskite: Custom object representing the structure to be relaxed.
    - shifts: List of energy shift values to be tested.
    - splits: List of split norm values to be tested.
    Returns:
    - None. The function runs multiple calculations and saves the results to a CSV file.
    """
    # Set up directory
    formula = perovskite.formula
    dir = f'results/bulk/{formula}/basis'
    # Load existing results if they exist, otherwise create an empty DataFrame
    if os.path.exists(os.path.join(dir, 'basisopt.csv')):
        df = pd.read_csv(os.path.join(dir, 'basisopt.csv'))
    else:
        df = pd.DataFrame(columns=['EnergyShift', 'SplitNorm', 'Energy', 'Enthalpy'])
    # Run over all combinations of energy shifts and split norms to optimize the basis set parameters
    for shift, split in product(shifts, splits):
        # Check if results have been obtained
        if ((df['EnergyShift'] == shift) & (df['SplitNorm'] == split)).any():
            print(f"EnergyShift={shift} Ry and SplitNorm={split} is in the DataFrame. Skipping.")
        else:
            # Create basis.fdf file for the current parameters
            generate_basis(perovskite, EnergyShift=shift, SplitNorm=split, dir=dir)
            # Get energy and enthalpy from SIESTA
            energy = run_siesta(perovskite, EnergyShift=shift, SplitNorm=split,
                                MeshCutoff=600, kgrid=(6, 6, 6), dir=dir)
            enthalpy = get_enthalpy(formula, dir)
            force = get_max_force(formula, dir)
            # Append results
            row = {
                "EnergyShift": shift,
                "SplitNorm": split,
                "Energy": energy,
                "Enthalpy": enthalpy,
                "MaxForce": force
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
    - perovskite: Custom object representing the structure to be relaxed.
    - meshcuts: List of mesh cutoff values to be tested.
    - kpoints: List of k-point mesh values to be tested.
    Returns:
    - None. The function runs multiple calculations and saves the results to a CSV file.
    """
    # Set up directory
    formula = perovskite.formula
    dir = f'results/bulk/{formula}/grid'

    # Check if a basis optimization has been performed and if the corresponding basis.fdf file exists
    if not os.path.exists(os.path.join(dir, 'basis.fdf')):
        # Create basis.fdf file for the current parameters
        generate_basis(perovskite, EnergyShift=0.01, SplitNorm=0.15, dir=dir)

    def _run_single_calculation(perovskite, mc, kp):

        formula = perovskite.formula
        dir = f'results/bulk/{formula}/grid'

        # Load existing results if they exist, otherwise create an empty DataFrame
        if os.path.exists(os.path.join(dir, 'gridconv.csv')):
            df = pd.read_csv(os.path.join(dir, 'gridconv.csv'))
        else:
            df = pd.DataFrame(columns=['MeshCutoff', 'kgrid', 'Energy', 'Enthalpy'])

        # Check if results have been obtained
        if ((df['MeshCutoff'] == mc) & (df['kgrid'] == f"({kp}, {kp}, {kp})")).any():
            print(f"MeshCutoff={mc} and kgrid=({kp}, {kp}, {kp}) is in the DataFrame. Skipping.")
        else:
            # Get energy and enthalpy from SIESTA
            energy = run_siesta(perovskite, EnergyShift=0.01, SplitNorm=0.15,
                                MeshCutoff=mc, kgrid=(kp, kp, kp), dir=dir)
            enthalpy = get_enthalpy(formula, dir)
            force = get_max_force(formula, dir)
            meshcutoff = get_meshcutoff(formula, dir)
            # Append results
            row = {
                "MeshCutoff": meshcutoff,
                "kgrid": f"({kp}, {kp}, {kp})",
                "Energy": energy,
                "Enthalpy": enthalpy,
                "MaxForce": force
            }
            # Create new dataframe
            df_new = pd.DataFrame([row])
            # Update old datafrem with new results
            df = pd.concat([df, df_new], ignore_index=True)
            # Save new results
            df.to_csv(os.path.join(dir, 'gridconv.csv'), index=False)

    # First, run over all mesh cutoffs for a fixed k-point mesh to optimize the mesh cutoff
    kp = kpoints[-1]  # Use the highest k-point mesh
    for mc in meshcuts:
        _run_single_calculation(perovskite, mc, kp)
    
    # Then, run over all k-point meshes for a fixed mesh cutoff to optimize the k-point mesh
    mc = meshcuts[-1]  # Use the highest mesh cutoff
    for kp in kpoints:
        _run_single_calculation(perovskite, mc, kp)

    # Clean directory of SIESTA calculations
    cleanFiles(directory=dir, confirm=False)