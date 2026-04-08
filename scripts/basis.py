# Import modules
from src.structure import Perovskite
from src.parameterconv import basis_opt
import numpy as np
# Define shifts and splits for calculations
shifts = np.array([0.001, 0.002, 0.003, 0.005, 0.007, 0.010, 0.015, 0.020])  # in Ry
splits = np.arange(0.1, 0.51, 0.05)

# Create perovskite atoms object
BaTiO3 = Perovskite('BaTiO3')
#SrTiO3 = Perovskite('SrTiO3')
# Run basis "optimization" which outputs/updates a .csv
basis_opt(BaTiO3, shifts, splits)
#basis_opt(SrTiO3, shifts, splits)
