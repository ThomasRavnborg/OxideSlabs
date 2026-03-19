# Import modules
from src.structure import Perovskite
from src.parameterconv import grid_conv
import numpy as np
# Define shifts and splits for calculations
meshcuts = np.arange(200, 1201, 100)  # in Ry
kpoints = np.arange(5, 13, 1)  # k-point grid size
# Create perovskite atoms object
SrTiO3 = Perovskite('SrTiO3', a=3.9)
# Move Ti atoms by 0.1 Å in the z-direction
SrTiO3.atoms[1].position[2] += 0.1
# Run basis "optimization" which outputs a .csv
grid_conv(SrTiO3, meshcuts, kpoints)