import os
import numpy as np
import phonopy as ph
from src.frozenphonon import calculate_frozen_phonons

formula = 'BaTiO3'
id = '0064'

# Set the results directory
#dir_res = os.path.join('results/bulk/',formula,'frozen')
dir = os.path.join('results/bulk/',formula, id)

# Load phonon data from the specified directory and formula
phonon = ph.load(os.path.join(dir, f'phonons/{formula}.yaml'))

calculate_frozen_phonons(phonon, xcf='PBEsol', basis='DZP',
                         EnergyShift=0.001, SplitNorm=0.1,
                         MeshCutoff=1000, kgrid=(10, 10, 10),
                         dir=os.path.join(dir,'frozen'))