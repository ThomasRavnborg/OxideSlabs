import os
import copy as cp
import json
import random
import subprocess
from ase.io import read
#import calorine
from calorine.nep import setup_training


# Define the directory in which to create the input files for NEP training
run_dir = 'results/bulk/BaTiO3/ML'

# Read in the structures and energies
structures = read(os.path.join(run_dir, 'structures.xyz@0:'))
with open(os.path.join(run_dir, 'energies.json'), 'r') as f:
    energies = json.load(f)
# Randomly shuffle order of structures to ensure that the training and test sets are representative of the entire dataset
random.shuffle(structures)

# Shift the energies of the structures by the sum of the energies of the constituent atoms
# This ensures that the NEP will learn the relative energies of the structures rather than the absolute energies
# This is important for the transferability of the NEP to other systems containing the same elements.
# Also, keep track of the unique elements in the structures
unique_elements = set()
for atoms in structures:
    N_atoms = len(atoms)
    elements = atoms.get_chemical_symbols()
    unique_elements.update(elements)
    atoms.calc.results['energy'] -= sum(energies[element] for element in elements)
    atoms.calc.results['energy'] /= N_atoms
# Count the number of unique elements in the structures
N_elements = len(unique_elements)

# Prepare input for NEP construction
parameters = dict(version=4,
                  type=[N_elements, ' '.join(unique_elements)],
                  cutoff=[8, 4],
                  neuron=30,
                  generation=100000,
                  batch=1000000)
# Set up the input files for NEP training
setup_training(parameters, structures,
               rootdir=run_dir, overwrite=True,
               mode='kfold', n_splits=10)

# Run the NEP training via the nep executable of the GPUMD package
subprocess.run(["nep"], cwd=os.path.join(run_dir, 'nepmodel_split1'),
               check=True, text=True)