from ase import Atoms
from ase.io import read, write
from src.phononcalc import calculate_phonons
# Read relaxed structures
BaTiO3 = read('bulk/relax/BaTiO3.xyz')
#SrTiO3 = read('bulk/relax/SrTiO3.xyz')
# Calculate phonon dispersion
calculate_phonons(BaTiO3, xcf='PBEsol', basis='DZP', cutoff=1200, shift=0.008, kmesh=[5, 5, 5])
#calculate_phonons(SrTiO3, xcf='PBEsol', basis='DZP', cutoff=1200, shift=0.008, kmesh=[5, 5, 5])