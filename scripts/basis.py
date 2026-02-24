# Import modules
from src.structure import Perovskite
from src.parameterconv import basis_opt
import numpy as np
import pandas as pd
import os
# Define shifts and splits for calculations
shifts = np.arange(0.001, 0.016, 0.001)  # in Ry
splits = np.arange(0.1, 0.41, 0.02)
# Create perovskite atoms object
BaTiO3 = Perovskite('BaTiO3', a=3.97)
# Run basis "optimization" which outputs/updates a .csv
basis_opt(BaTiO3, shifts, splits)