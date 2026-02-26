# Import modules
from src.structure import Perovskite
from src.parameterconv import basis_opt
import numpy as np
# Define shifts and splits for calculations
shifts = np.arange(0.003, 0.013, 0.001)  # in Ry
splits = np.arange(0.1, 0.41, 0.05)
# Create perovskite atoms object
BaTiO3 = Perovskite('BaTiO3', a=4.01)
# Run basis "optimization" which outputs/updates a .csv
basis_opt(BaTiO3, shifts, splits)