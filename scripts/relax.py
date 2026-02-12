# Import structureoptimizer module
from src.structureoptimizer import perovskite, relax_siesta
# Create atoms object for BaTiO3 and SrTiO3
BaTiO3 = perovskite('BaTiO3')
SrTiO3 = perovskite('SrTiO3')
# Run relaxation using SIESTA for atomic positions and cell optimization
relax_siesta(BaTiO3, xcf='PBEsol', basis='DZP', shift=0.01, split=0.15, cutoff=1200, kmesh=[10, 10, 10])
relax_siesta(SrTiO3, xcf='PBEsol', basis='DZP', shift=0.01, split=0.15, cutoff=1200, kmesh=[10, 10, 10])