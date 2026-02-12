from ase import Atoms
from ase.io import read, write
from src.bandscalc import calculate_bands
# Read relaxed structures
BaTiO3 = read('results/bulk/relaxsiesta/BaTiO3.xyz')
SrTiO3 = read('results/bulk/relaxsiesta/SrTiO3.xyz')
# Calculate band structure and PDOS
calculate_bands(BaTiO3, xcf='PBEsol', basis='DZP', shift=0.01, split=0.15, cutoff=1200, kmesh=[10, 10, 10])
calculate_bands(SrTiO3, xcf='PBEsol', basis='DZP', shift=0.01, split=0.15, cutoff=1200, kmesh=[10, 10, 10])