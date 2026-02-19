# Import
from ase import Atoms
from ase.io import read
#from ase.parallel import world
from src.structureoptimizer import perovskite, relax_ase
from src.bandscalc import calculate_bands
from src.phononcalc import calculate_phonons

def run(formula):
    """Function to run the entire workflow for a given perovskite formula.
    Parameters:
    - formula: Chemical formula of the perovskite ('ABX3').
    Returns:
    - None. The function performs the relaxation, band structure calculation, and phonon calculation, and saves the results to files.
    """
    # Create atoms object for the given formula
    atoms = perovskite(formula)
    # Run relaxation using GPAW for atomic positions and cell optimization
    relax_ase(atoms, xcf='PBEsol',
             MeshCutoff=100, kgrid=(10, 10, 10),
             mode='pw', dir=f'results/bulk/GPAW')
    # Read relaxed structure
    relaxed_atoms = read(f'results/bulk/GPAW/{formula}.xyz', index=0)

    # Calculate band structure and PDOS
    calculate_bands(relaxed_atoms, xcf='PBEsol',
                    MeshCutoff=100, kgrid=(10, 10, 10),
                    mode='pw', dir=f'results/bulk/GPAW')
    # Calculate phonon dispersion
    calculate_phonons(relaxed_atoms, xcf='PBEsol', basis='DZP',
                      MeshCutoff=100, kgrid=(10, 10, 10),
                      mode='pw', dir=f'results/bulk/GPAW')

run('BaTiO3')