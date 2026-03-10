# Import
import os
from ase import Atoms
from ase.io import read
import phonopy as ph
#from ase.parallel import world
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

    perovskite = Perovskite(formula, N=1)
    bulk = perovskite.bulk
    if bulk:
        struc = 'bulk'
    else:
        struc = 'slab'

    dir = f'results/{struc}/{formula}/GPAW/{task}'
    if not os.path.exists(dir):
        os.makedirs(dir, exist_ok=True)

    if task == 'relax':
        # Run relaxation using GPAW for atomic positions and cell optimization
        relax_ase(perovskite, xcf='PBEsol',
                  MeshCutoff=100, kgrid=(10, 10, 10),
                  mode='pw', dir=dir)
    elif task == 'bands' or task == 'phonons':
        # Read relaxed structure
        relaxed_atoms = read(os.path.join(dir, f'{formula}.xyz'), index=0)
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
        phonon = ph.load(os.path.join(f'results/{struc}/{formula}/GPAW/phonons', f'{formula}.yaml'))
        # Calculate frozen phonons for the given phonon object and parameters, and save results in the specified directory
        calculate_frozen_phonons(phonon, dd=0.5, xcf='PBEsol',
                                 MeshCutoff=100, kgrid=(10, 10, 10),
                                 mode='pw', dir=dir)

run('BaTiO3', 'frozen')
#run('SrTiO3', 'frozen')