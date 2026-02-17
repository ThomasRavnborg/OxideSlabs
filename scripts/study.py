import os
from ase.io import read
from itertools import product
from src.utils import SiestaProject
from src.structureoptimizer import perovskite, relax_ase, relax_siesta
from src.bandscalc import calculate_bands
from src.phononcalc import calculate_phonons

# Create atoms object for BaTiO3 and initialize project
atoms = perovskite('BaTiO3')
project = SiestaProject(material=str(atoms.symbols))

# Define lists of parameters to iterate over
xcfs =    ['PBEsol']
basis =   ['DZP']
pseudos = [2, 3, 4]
shifts =  [0.006, 0.008, 0.01]
splits =  [0.15]
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
        'pseudo': combo[2],
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

    if next_step == "complete":
        print(f"All steps complete for calculation {calc_id}. Skipping.")

    # Run the appropriate calculation based on the next step
    if next_step == "relax":
        # Run relaxation
        print(f"Running relaxation for calculation {calc_id}")
        relax_ase(atoms, **params, dir=os.path.join(dir, next_step))
        # Update dataframe and move to next step
        calc_id = project.prepare_calculation(params)
        next_step = project.what_to_run(calc_id)
    
    if next_step == "bands":
        # Run band structure calculation
        print(f"Running band structure calculation for calculation {calc_id}")
        atoms = read(os.path.join(dir, 'relax', f'{project.material}.xyz'))
        calculate_bands(atoms, **params, dir=os.path.join(dir, next_step), par=True)
        # Update to next step
        calc_id = project.prepare_calculation(params)
        next_step = project.what_to_run(calc_id)

    if next_step == "phonons":
        print(f"Running phonon calculation for calculation {calc_id}")
        atoms = read(os.path.join(dir, 'relax', f'{project.material}.xyz'))
        calculate_phonons(atoms, **params, dir=os.path.join(dir, next_step), par=True)
        # Update to next step
        calc_id = project.prepare_calculation(params)
        next_step = project.what_to_run(calc_id)