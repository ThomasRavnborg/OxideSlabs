# Import
import os
from ase import Atoms
from ase.io import read
from ase.parallel import world
import phonopy as ph
#from ase.parallel import world
from gpaw.mpi import world

from src.structure import Perovskite
from src.structureoptimizer import relax_ase
from src.bandscalc import calculate_bands
from src.phononcalc import calculate_phonons
from src.frozenphonon import calculate_frozen_phonons

def run(formula, task):
    """Function to run the entire workflow for a given perovskite formula.
    Parameters:
    - formula: Chemical formula of the perovskite ('ABX3').
    Returns:
    - None. The function performs the relaxation, band structure calculation, and phonon calculation, and saves the results to files.
    """

    perovskite = Perovskite(formula, a=4.01, N=1, bulk=True)
    N = perovskite.ncells
    bulk = perovskite.bulk
    if bulk:
        struc = f"bulk/{formula}"
    else:
        struc = f"slab/{formula}/{N}uc"

    dir = f'results/{struc}/GPAW/{task}'
    if not os.path.exists(dir):
        os.makedirs(dir, exist_ok=True)

    if task == 'relax':
        # Run relaxation using GPAW for atomic positions and cell optimization
        relax_ase(perovskite, xcf='PBEsol',
                  MeshCutoff=100, kgrid=(10, 10, 10),
                  mode='pw', dir=dir)
    elif task == 'bands' or task == 'phonons':
        # Read relaxed structure
        relaxed_atoms = read(os.path.join(f'results/{struc}/GPAW/relax', f'{formula}.xyz'), index=0)
        perovskite.set_atoms(relaxed_atoms)
    if task == 'bands':
        # Calculate band structure and PDOS
        calculate_bands(perovskite, xcf='PBEsol',
                        MeshCutoff=100, kgrid=(10, 10, 10),
                        mode='pw', dir=dir)
    if task == 'phonons':
        # Calculate phonon dispersion
        calculate_phonons(perovskite, xcf='PBEsol',
                          MeshCutoff=100, kgrid=(10, 10, 10),
                          mode='pw', dir=dir)
    if task == 'frozen':
        # Calculate frozen phonons
        # Load phonon data from the specified directory and formula
        phonon = ph.load(os.path.join(f'results/{struc}/GPAW/phonons', f'{formula}.yaml'))
        # Run frozen phonon calculation
        calculate_frozen_phonons(phonon, dd=0.6, xcf='PBEsol',
                                 MeshCutoff=100, kgrid=(10, 10, 10),
                                 mode='pw', dir=dir)

for formula in ['BaTiO3']:
    for task in ['frozen']:
        run(formula, task)
        # Wait for all parallel processes to finish
        world.barrier()
