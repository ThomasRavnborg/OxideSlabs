# Import modules
from src.structure import Perovskite
from src.parameterconv import grid_conv
import numpy as np
# Define shifts and splits for calculations
meshcuts = np.arange(200, 1001, 100)  # in Ry
kpoints = np.arange(5, 16, 1)  # k-point grid size
# Create perovskite atoms object
BaTiO3 = Perovskite('BaTiO3', a=4.01)
# Run basis "optimization" which outputs a .csv
grid_conv(BaTiO3, meshcuts, kpoints)