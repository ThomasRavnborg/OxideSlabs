from ase import Atoms
from ase.io import read, write
from src.bandscalc import calculate_bands
# Read relaxed structures
BaTiO3 = read('bulk/relax/BaTiO3.xyz')
#SrTiO3 = read('bulk/relax/SrTiO3.xyz')
# Calculate band structure and PDOS
calculate_bands(BaTiO3, xcf='PBEsol', basis='DZP', cutoff=1200, shift=0.008, kmesh=[10, 10, 10])
#calculate_bands(SrTiO3, xcf='PBEsol', basis='DZP', cutoff=1200, shift=0.008, kmesh=[10, 10, 10])