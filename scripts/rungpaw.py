# Import
import os
from ase import Atoms
from ase.io import read
#from ase.parallel import world
from src.structure import Perovskite
from src.structureoptimizer import relax_ase
from src.bandscalc import calculate_bands
from src.phononcalc import calculate_phonons

def run(formula, task):
    """Function to run the entire workflow for a given perovskite formula.
    Parameters:
    - formula: Chemical formula of the perovskite ('ABX3').
    Returns:
    - None. The function performs the relaxation, band structure calculation, and phonon calculation, and saves the results to files.
    """

    perovskite = Perovskite(formula, N=1, bulk=False)
    bulk = perovskite.bulk
    if bulk:
        struc = 'bulk'
    else:
        struc = 'slab'

    if task == 'relax':
        # Run relaxation using GPAW for atomic positions and cell optimization
        relax_ase(perovskite, xcf='PBEsol',
                  MeshCutoff=100, kgrid=(10, 10, 10),
                  mode='pw', dir=f'results/{struc}/GPAW')
    else:
        # Read relaxed structure
        relaxed_atoms = read(f'results/{struc}/GPAW/{formula}.xyz', index=0)
        perovskite.set_atoms(relaxed_atoms)
    if task == 'bands':
        # Calculate band structure and PDOS
        calculate_bands(perovskite, xcf='PBEsol',
                        MeshCutoff=100, kgrid=(10, 10, 10),
                        mode='pw', dir=f'results/{struc}/GPAW')
    if task == 'phonons':
        # Calculate phonon dispersion
        calculate_phonons(perovskite, xcf='PBEsol',
                          MeshCutoff=100, kgrid=(10, 10, 10),
                          mode='pw', dir=f'results/{struc}/GPAW')

run('BaTiO3', 'bands')
run('BaTiO3', 'phonons')