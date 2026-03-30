import os
import copy as cp
import subprocess
from ase.io import read
#import calorine
from calorine.nep import setup_training

# Read in the structures and energies
structures = read('results/bulk/BaTiO3/0082/frozen/G/mode_1/Q_1/structures.xyz@0:')
energies = {'Ba': -761.227747, 'Ti': -1604.974503, 'O': -440.177463}

# Define the directory in which to create the input files for NEP training
run_dir = 'results/bulk/BaTiO3/ML'


def shift_energy(structures, energies):

    structures_copy = cp.deepcopy(structures)

    for atoms in structures_copy:
        elements = atoms.get_chemical_symbols()
        atoms.calc.results['energy'] -= sum(energies[element] for element in elements)

    return structures_copy

structures = shift_energy(structures, energies)


elements = structures[0].get_chemical_symbols()
unique_elements = list(set(elements))
N_elements = len(unique_elements)
unique_elements_str = ' '.join(unique_elements)

# prepare input for NEP construction
parameters = dict(version=4,
                  type=[N_elements, unique_elements_str],
                  cutoff=[8, 4],
                  neuron=30,
                  generation=100000,
                  batch=1000000)

setup_training(parameters, structures, rootdir=run_dir, overwrite=True)

subprocess.run(["nep"], cwd=os.path.join(run_dir, 'nepmodel_full'),
               check=True, text=True)