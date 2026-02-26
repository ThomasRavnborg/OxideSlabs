import os
from ase.io import read
from ase.parallel import parprint
from itertools import product
from src.utils import SiestaProject
from src.structure import Perovskite
from src.structureoptimizer import relax_ase, relax_siesta
from src.bandscalc import calculate_bands
from src.phononcalc import calculate_phonons

# Create atoms object for BaTiO3 and initialize project
perovskite = Perovskite('BaTiO3', a=4.01)
formula = perovskite.formula
project = SiestaProject(material=formula)

# Define lists of parameters to iterate over
xcfs =    ['PBEsol']
basis =   ['DZP']
pseudos = ['PBEsol']
shifts =  [0.001]
splits =  [0.1]
cutoffs = [1000]
grids =   [10]

# Find all combinations of parameters and store in a list of dictionaries
combinations = list(product(xcfs, basis, pseudos, shifts, splits, cutoffs, grids))
print(f"Total combinations: {len(combinations)}")
param_dicts = []
for combo in combinations:
    param_dicts.append({
        'xcf': combo[0],
        'basis': combo[1],
        #'pseudo': combo[2],
        'EnergyShift': combo[3],
        'SplitNorm': combo[4],
        'MeshCutoff': combo[5],
        'kgrid': (combo[6], combo[6], combo[6])
    })

# Loop over parameter combinations and prepare calculations
for params in param_dicts:
    # Find calculation ID for this parameter set, or create a new one if it doesn't exist
    calc_id = project.prepare_calculation(params)

    # Check what step needs to be run for this calculation and set directory
    next_step = project.what_to_run(calc_id)
    dir = os.path.join(project.material_path, calc_id)

    # If all steps are complete, skip to the next parameter set
    if next_step == "complete":
        parprint(f"All steps complete for calculation {calc_id}. Skipping.")

    # If relaxation needs to be run, run it and update the calculation ID and next step
    if next_step == "relax":
        # Run relaxation
        parprint(f"Running relaxation for calculation {calc_id}")
        dir_step = os.path.join(dir, next_step)
        relax_ase(perovskite, **params, dir=dir_step)
        # Update dataframe and move to next step
        calc_id = project.prepare_calculation(params)
        next_step = project.what_to_run(calc_id)
    
    # Set the atoms object for the next steps based on the relaxed structure
    dir_relax = os.path.join(dir, 'relax', f'{perovskite}.xyz')
    perovskite.set_atoms(read(os.path.join(dir, 'relax', f'{formula}.xyz')))

    # If band structure calculation needs to be run, run it and update the calculation ID and next step
    if next_step == "bands":
        # Run band structure calculation
        parprint(f"Running band structure calculation for calculation {calc_id}")
        dir_step = os.path.join(dir, next_step)
        calculate_bands(perovskite, **params, dir=dir_step, par=True)
        # Update to next step
        calc_id = project.prepare_calculation(params)
        next_step = project.what_to_run(calc_id)

    next_step = 'skip_phonons'  # Temporary: skip phonon calculations for now

    # If phonon calculation needs to be run, run it and update the calculation ID and next step
    if next_step == "phonons":
        parprint(f"Running phonon calculation for calculation {calc_id}")
        dir_step = os.path.join(dir, next_step)
        calculate_phonons(perovskite, **params, dir=dir_step, par=True)
        # Update to next step
        calc_id = project.prepare_calculation(params)
        next_step = project.what_to_run(calc_id)