# Import modules
from src.structureoptimizer import perovskite
from src.parameterconv import basis_opt
# Define shifts and splits for calculations
shifts = [0.001, 0.002, 0.004, 0.006, 0.008, 0.01, 0.02, 0.03, 0.04]  # in Ry
splits = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
# Create perovskite atoms object
BaTiO3 = perovskite('BaTiO3')
# Move one of the atoms to create a force
BaTiO3[1].position += [0.1, 0.0, 0.0]
# Run basis "optimization" which outputs a .csv
basis_opt(BaTiO3, shifts, splits)