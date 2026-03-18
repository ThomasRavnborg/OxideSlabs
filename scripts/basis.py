# Import modules
from src.structure import Perovskite
from src.parameterconv import basis_opt
import numpy as np
# Define shifts and splits for calculations
shifts = np.arange(0.001, 0.011, 0.001)  # in Ry
splits = np.arange(0.05, 0.41, 0.05)
# Create perovskite atoms object
SrTiO3 = Perovskite('SrTiO3', a=3.9)
# Run basis "optimization" which outputs/updates a .csv
basis_opt(SrTiO3, shifts, splits)