# Import
import os
from ase import Atoms
from ase.io import read
from ase.parallel import world, parprint
import phonopy as ph
#from ase.parallel import world
from gpaw.mpi import world

from src.structure import Perovskite, check_if_bulk
from src.structureoptimizer import relax_ase
from src.bandscalc import calculate_bands
from src.phononcalc import calculate_phonons
from src.frozenphonon import calculate_frozen_phonons

def run(formula, task, strain=0.0, Ncells=1, bulk=True):
    """Function to run the entire workflow for a given perovskite formula.
    Parameters:
    - formula: Chemical formula of the perovskite ('ABX3').
    - strain: Strain value for the perovskite structure.
    - Ncells: Number of unit cells in the perovskite structure.
    - bulk: Boolean indicating whether the structure is bulk or a slab.
    Returns:
    - None. The function performs the relaxation, band structure calculation, and phonon calculation, and saves the results to files.
    """

    perovskite = Perovskite(formula, N=Ncells, bulk=bulk)
    N = perovskite.ncells
    bulk = check_if_bulk(perovskite.atoms)
    if bulk:
        struc = f"bulk/{formula}"
        desc = "bulk"
    else:
        struc = f"slab/{formula}/{N}uc"
        desc = f"slab_{N}uc"

    dir = f'results/{struc}/GPAW/{strain}'
    dir_task = os.path.join(dir, task)

    # Make folder for the task if it doesn't exist
    if not os.path.exists(dir_task):
        os.makedirs(dir_task, exist_ok=True)

    if task == 'relax':
        # If strain is not 0, then load the unstrained structure, apply the specified strain
        if strain != 0.0:
            # Define directory for the unstrained structure (strain = 0.0)
            dir_ref = f'results/{struc}/GPAW/0.0'
            # Load the unstrained structure
            perovskite.set_atoms(read(os.path.join(dir_ref, 'relax', f'{formula}.xyz')))
            # Apply the specified strain
            perovskite.apply_strain(strain)
            strained = True
        else:
            strained = False

        parprint(f"Running relaxation of {desc} {formula} with GPAW", flush=True)
        # Run relaxation using GPAW for atomic positions and cell optimization
        relax_ase(perovskite, xcf='PBEsol',
                  MeshCutoff=60, kgrid=(12, 12, 12),
                  mode='pw', dir=dir_task, strained=strained)
    elif task == 'bands' or task == 'phonons':
        # Read relaxed structure
        relaxed_atoms = read(os.path.join(dir, 'relax', f'{formula}.xyz'), index=0)
        perovskite.set_atoms(relaxed_atoms)
    if task == 'bands':
        parprint(f"Running band structure calculation for {desc} {formula} with GPAW", flush=True)
        # Calculate band structure and PDOS
        calculate_bands(perovskite, xcf='PBEsol',
                        MeshCutoff=60, kgrid=(12, 12, 12),
                        mode='pw', dir=dir_task)
    if task == 'phonons':
        parprint(f"Running phonon calculation for {desc} {formula} with GPAW", flush=True)
        # Calculate phonon dispersion
        calculate_phonons(perovskite, xcf='PBEsol',
                          MeshCutoff=60, kgrid=(12, 12, 12),
                          mode='pw', dir=dir_task)
    if task == 'frozen':
        parprint(f"Running frozen phonon calculation for {desc} {formula} with GPAW", flush=True)
        # Calculate frozen phonons
        # Load phonon data from the specified directory and formula
        phonon = ph.load(os.path.join(dir, 'phonons', f'{formula}.yaml'))
        # Run frozen phonon calculation
        calculate_frozen_phonons(phonon, n_points=5, xcf='PBEsol',
                                 MeshCutoff=60, kgrid=(12, 12, 12),
                                 mode='pw', dir=dir_task, deg=False)

for formula in ['BaTiO3']:
    for Ncells in [2.5, 3.5]:
        for strain in [0.0]:
            for task in ['relax', 'bands', 'phonons']:
                run(formula, task, strain, Ncells=Ncells, bulk=False)
                # Wait for all parallel processes to finish
                world.barrier()
