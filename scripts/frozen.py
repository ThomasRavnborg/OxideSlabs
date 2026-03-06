import os
import numpy as np
import pandas as pd
import phonopy as ph
from ase.io import read
from src.structure import Perovskite
from src.frozenphonon import calculate_frozen_phonons
from src.structureoptimizer import relax_ase, relax_siesta

formula = 'BaTiO3'
id = '0064'

dir = os.path.join('results/bulk/',formula, id)

# Load parameters.json file for this formula and ID from the directory
params = pd.read_json(os.path.join(dir, 'parameters.json'), orient='index').to_dict()[0]

"""
# Load phonon data from the specified directory and formula
phonon = ph.load(os.path.join(dir, f'phonons/{formula}.yaml'))

# Calculate frozen phonons for the given phonon object and parameters, and save results in the specified directory
calculate_frozen_phonons(phonon, **params, dir=os.path.join(dir,'frozen'))
"""

# Get the list of q-point folders in the frozen phonon directory and loop through them
folders = os.listdir(dir)
for q in folders:
    # Read the energies.csv file for this q-point and extract the energies and corresponding structures
    dir_q = os.path.join(dir, q)
    df = pd.read_csv(os.path.join(dir_q, 'energies.csv'))
    Energies = df['Energy'].to_numpy()
    # Read the structures from the structures.xyz file and find the one with the lowest energy
    Structures = read(os.path.join(dir_q, 'structures.xyz@0:'))
    inx = np.argmin(Energies)
    atoms = Structures[inx]
    # Create a Perovskite object with the given formula and set its atoms to the structure with the lowest energy
    perovskite = Perovskite(formula)
    perovskite.set_atoms(atoms)
    # Before relaxing, delete the old density matrix file if it exists
    if os.path.exists(os.path.join(dir_q, f'{formula}.DM')):
        os.remove(os.path.join(dir_q, f'{formula}.DM'))
    # Relax the structure using ASE and save the results in the specified directory
    relax_ase(perovskite, **params, dir=dir_q)
    