# Import modules
from src.structureoptimizer import perovskite
from src.parameterconv import grid_conv
# Define shifts and splits for calculations
meshcuts = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]  # in Ry
kpoints = [5, 10, 15, 20]
# Create perovskite atoms object
BaTiO3 = perovskite('BaTiO3')
# Move one of the atoms to create a force
BaTiO3[1].position += [0.1, 0.0, 0.0]
# Run basis "optimization" which outputs a .csv
grid_conv(BaTiO3, meshcuts, kpoints)