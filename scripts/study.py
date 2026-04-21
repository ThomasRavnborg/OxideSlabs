import os
from ase.io import read
from ase.parallel import parprint
import phonopy as ph
from itertools import product
from src.frozenphonon import calculate_frozen_phonons
from src.utils import SiestaProject
from src.structure import Perovskite
from src.fdfcreate import generate_basis
from src.structureoptimizer import relax_ase, relax_siesta
from src.bandscalc import calculate_bands
from src.phononcalc import calculate_phonons
from src.frozenphonon import calculate_frozen_phonons

# Define lists of parameters to iterate over
xcfs =    ['PBEsol']
basis =   ['DZPp']
shifts =  [0.01]
splits =  [0.15]
cutoffs = [1000]
grids =   [12]
strains = [0.0]
#strains = [0.0, 0.01, -0.01, 0.005, -0.005]

def run(formula, xcfs, basis, shifts, splits, cutoffs, grids, strains, runall=False):
    """Run the full workflow for all combinations of parameters."""
    # Find all combinations of parameters and store in a list of dictionaries
    combinations = list(product(xcfs, basis, shifts, splits, cutoffs, grids, strains))
    #parprint(f"Total combinations: {len(combinations)}")
    param_dicts = []
    for combo in combinations:
        param_dicts.append({
            'xcf': combo[0],
            'basis': combo[1],
            'EnergyShift': combo[2],
            'SplitNorm': combo[3],
            'MeshCutoff': combo[4],
            'kgrid': (combo[5], combo[5], combo[5]),
            'strain': combo[6]
        })

    # Create atoms object for BaTiO3 and initialize project
    perovskite = Perovskite(formula, N=1.5, bulk=False)
    project = SiestaProject(perovskite)

    # Loop over parameter combinations and prepare calculations
    for params in param_dicts:
        # Find calculation ID for this parameter set, or create a new one if it doesn't exist
        calc_id = project.prepare_calculation(params)
        # Update dataframe
        #project.update_summary(calc_id, params)
        # Check what step needs to be run for this calculation and set directory
        next_step = project.what_to_run(calc_id)
        dir = os.path.join(project.path, calc_id)
        # Create a copy of the parameters dictionary without the 'strain' key for use in the calculation functions
        params_calc = params.copy()
        params_calc.pop('strain')

        # If all steps are complete, skip to the next parameter set unless runall is True
        if next_step == "complete" and not runall:
            parprint(f"All steps complete for calculation {calc_id}. Skipping.", flush=True)
            continue

        # If basis generation needs to be run, run it and update the calculation ID and next step
        if next_step == "basis" or runall:
            # Generate basis.fdf file
            parprint(f"Generating basis.fdf for calculation {calc_id}", flush=True)
            generate_basis(perovskite.atoms, **params_calc, dir=dir)
            # Update to next step
            next_step = project.what_to_run(calc_id)

        # If relaxation needs to be run, run it and update to the next step
        if next_step == "relax" or runall:
            # If strain is not 0, then load the unstrained structure, apply the specified strain
            if params['strain'] != 0.0:
                # Get reference calculation ID for the unstrained structure (strain = 0.0) with the same other parameters
                params_ref = params.copy()
                params_ref['strain'] = 0.0
                ref_id = project.prepare_calculation(params_ref)
                dir_ref = os.path.join(project.path, ref_id)
                # Load the unstrained structure
                perovskite.set_atoms(read(os.path.join(dir_ref, 'relax', f'{formula}.xyz')))
                # Apply the specified strain
                perovskite.apply_strain(params['strain'])
                strained = True
            else:
                strained = False
            # Run relaxation
            parprint(f"Running relaxation for calculation {calc_id} with SIESTA", flush=True)
            dir_step = os.path.join(dir, 'relax')
            relax_ase(perovskite, **params_calc, strained=strained, dir=dir_step)
            # Update dataframe and move to next step
            project.update_summary(calc_id, params)
            next_step = project.what_to_run(calc_id)
        
        # Set the atoms object for the next steps based on the relaxed structure
        dir_relax = os.path.join(dir, 'relax')
        #dir_relax = 'results/bulk/GPAW'
        perovskite.set_atoms(read(os.path.join(dir_relax, f'{formula}.xyz')))
        
        # If band structure calculation needs to be run, run it and update to the next step
        if next_step == "bands" or runall:
            # Run band structure calculation
            parprint(f"Running band structure calculation for calculation {calc_id} with SIESTA", flush=True)
            dir_step = os.path.join(dir, 'bands')
            calculate_bands(perovskite, **params_calc, dir=dir_step)
            # Update dataframe and move to next step
            project.update_summary(calc_id, params)
            next_step = project.what_to_run(calc_id)

        # If phonon calculation needs to be run, run it and update to the next step
        if next_step == "phonons" or runall:
            parprint(f"Running phonon calculation for calculation {calc_id} with SIESTA", flush=True)
            dir_step = os.path.join(dir, 'phonons')
            calculate_phonons(perovskite, **params_calc, dir=dir_step)
            # Update dataframe
            project.update_summary(calc_id, params)
            next_step = project.what_to_run(calc_id)
        
        # If frozen phonon calculation needs to be run, run it and update the dataframe
        if next_step == "frozen" or runall:
            parprint(f"Running frozen phonon calculation for calculation {calc_id} with SIESTA", flush=True)
            dir_step = os.path.join(dir, 'frozen')
            # Load phonon data from the specified directory and formula
            phonon = ph.load(os.path.join(dir, 'phonons', f'{formula}.yaml'))
            # Calculate frozen phonons for the given phonon object and parameters, and save results in the specified directory
            calculate_frozen_phonons(phonon, **params_calc, dir=dir_step)
            # Update dataframe
            project.update_summary(calc_id, params)


for formula in ['BaTiO3', 'SrTiO3']:
    run(formula, xcfs, basis, shifts, splits, cutoffs, grids, strains)
