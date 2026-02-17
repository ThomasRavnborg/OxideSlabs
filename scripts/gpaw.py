# Import
from ase import Atoms
from ase.io import read
from src.structureoptimizer import perovskite, relax_ase
from src.bandscalc import calculate_bands
from src.phononcalc import calculate_phonons
# Create atoms object for BaTiO3 and SrTiO3
BaTiO3 = perovskite('BaTiO3')
SrTiO3 = perovskite('SrTiO3')
# Run relaxation using GPAW for atomic positions and cell optimization
relax_ase(BaTiO3, xcf='PBEsol',
          MeshCutoff=100, kgrid=(10, 10, 10),
          mode='pw', dir='results/bulk/GPAW')
relax_ase(SrTiO3, xcf='PBEsol',
          MeshCutoff=100, kgrid=(10, 10, 10),
          mode='pw', dir='results/bulk/GPAW')
# Read relaxed structures
BaTiO3 = read('results/bulk/relaxsiesta/BaTiO3.xyz')
SrTiO3 = read('results/bulk/relaxsiesta/SrTiO3.xyz')
# Calculate band structure and PDOS
calculate_bands(BaTiO3, xcf='PBEsol',
                MeshCutoff=100, kgrid=(10, 10, 10),
                mode='pw', dir='results/bulk/GPAW')
calculate_bands(SrTiO3, xcf='PBEsol',
                MeshCutoff=100, kgrid=(10, 10, 10),
                mode='pw', dir='results/bulk/GPAW')
# Calculate phonon dispersion
calculate_phonons(BaTiO3, xcf='PBEsol', basis='DZP',
                  MeshCutoff=100, kgrid=(10, 10, 10),
                  mode='pw', dir='results/bulk/GPAW')
calculate_phonons(SrTiO3, xcf='PBEsol', basis='DZP',
                  MeshCutoff=100, kgrid=(10, 10, 10),
                  mode='pw', dir='results/bulk/GPAW')