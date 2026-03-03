import numpy as np
from src.frozenphonon import run_frozen_phonons

displacements = np.linspace(-0.1, 0.1, 3) # in Angstrom

run_frozen_phonons('BaTiO3', '0064', 'G', displacements, skip=True)