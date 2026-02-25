from ase import Atoms
from ase.io import read
from src.bandscalc import calculate_bands
# Read relaxed structures
BaTiO3 = read('results/bulk/GPAW/BaTiO3.xyz')
# Calculate band structure and PDOS
calculate_bands(BaTiO3, xcf='PBEsol', basis='DZP',
                EnergyShift=0.01, SplitNorm=0.15,
                MeshCutoff=1000, kgrid=(10, 10, 10),
                dir='results/bulk/test_bands')