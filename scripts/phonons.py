from ase import Atoms
from ase.io import read
from src.phononcalc import calculate_phonons
# Read relaxed structures
BaTiO3 = read('results/bulk/relaxsiesta/BaTiO3.xyz')
SrTiO3 = read('results/bulk/relaxsiesta/SrTiO3.xyz')
# Calculate phonon dispersion
calculate_phonons(BaTiO3, xcf='PBEsol', basis='DZP',
                  EnergyShift=0.01, SplitNorm=0.15,
                  MeshCutoff=1000, kgrid=(10, 10, 10))
calculate_phonons(SrTiO3, xcf='PBEsol', basis='DZP',
                  EnergyShift=0.01, SplitNorm=0.15,
                  MeshCutoff=1000, kgrid=(10, 10, 10))