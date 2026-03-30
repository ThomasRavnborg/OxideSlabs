
import os
import copy as cp
import subprocess
from ase.io import read
#import calorine
from calorine.nep import setup_training

structures = read('results/bulk/BaTiO3/0082/frozen/G/mode_1/Q_1/structures.xyz@0:')
energies = {'Ba': -761.227747, 'Ti': -1604.974503, 'O': -440.177463}

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

# prepare input for NEP construction
parameters = dict(version=4,
                  type=[N_elements, unique_elements],
                  cutoff=[8, 4],
                  neuron=30,
                  generation=100000,
                  batch=1000000)

#os.chdir(run_dir)

setup_training(parameters, structures, rootdir=run_dir, overwrite=True)

subprocess.run(["nep"], cwd=run_dir, check=True)