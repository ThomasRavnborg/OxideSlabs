from ase import Atoms
from ase.io import read, write
from src.phononcalc import calculate_phonons
# Read relaxed structures
BaTiO3 = read('results/bulk/relaxsiesta/BaTiO3.xyz')
SrTiO3 = read('results/bulk/relaxsiesta/SrTiO3.xyz')
# Calculate phonon dispersion
calculate_phonons(BaTiO3, xcf='PBEsol', basis='DZP', shift=0.01, split=0.15, cutoff=1200, kmesh=[5, 5, 5])
calculate_phonons(SrTiO3, xcf='PBEsol', basis='DZP', shift=0.01, split=0.15, cutoff=1200, kmesh=[5, 5, 5])