# Libraries
from ase import Atoms
from ase.calculators.siesta import Siesta
from ase.units import Ry
import os
import sisl as si
import numpy as np
import pandas as pd
from itertools import product

def calcSiesta(atoms, xcf='PBE', basis='DZP', shift=0.01, split=0.15,
               cutoff=200, kmesh=[5, 5, 5], dir='/bulk/basis/'):
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

    # Calculation parameters in a dictionary
    calc_params = {
        'label': f'{atoms.symbols}',
        'xc': xcf,
        'basis_set': basis,
        'mesh_cutoff': cutoff * Ry,
        'energy_shift': shift * Ry,
        'kpts': kmesh,
        'directory': cwd + dir,
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

def get_enthalpy(atoms, dir='bulk/basis'):
    # Read basis enthalpy from file
    with open(f'{dir}/{atoms.symbols}.BASIS_ENTHALPY', 'r') as file:
        lines = file.readlines()
    enthalpy = float(lines[0].split()[-1])
    return enthalpy

def get_maxforce(atoms, dir='bulk/basis'):
    sile = si.get_sile(f'{dir}/{atoms.symbols}.FA')
    forces = sile.read_force()
    return np.max(np.abs(forces))

def basis_optimization(atoms, shifts, splits):
    rows = []
    for sh, sp in product(shifts, splits):
        try:
            calcSiesta(atoms, shift=sh, split=sp, dir='/bulk/basis/')
            enthalpy = get_enthalpy(atoms, dir='bulk/basis/')
            maxforce = get_maxforce(atoms, dir='bulk/basis/')
        except Exception as e:
            enthalpy = None
            maxforce = None

        rows.append({
            "Shift": sh,
            "Split": sp,
            "Enthalpy": enthalpy,
            "MaxForce": maxforce,
        })

    df = pd.DataFrame(rows)
    df.to_csv(f'bulk/basis/basisopt.csv')

def optimize_grid(atoms, meshcuts, kpoints):
    rows = []
    for mc, kp in product(meshcuts, kpoints):
        try:
            calcSiesta(atoms, cutoff=mc, kmesh=[kp, kp, kp], dir='/bulk/grid/')
            enthalpy = get_enthalpy(atoms, dir='bulk/grid/')
            maxforce = get_maxforce(atoms, dir='bulk/grid/')
        except Exception as e:
            enthalpy = None
            maxforce = None

        rows.append({
            "MeshCutoff": mc,
            "KPoints": kp,
            "Enthalpy": enthalpy,
            "MaxForce": maxforce,
        })

    df = pd.DataFrame(rows)
    df.to_csv(f'bulk/grid/gridopt.csv')