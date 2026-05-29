
from ase.io import read, write
import json
from src.calculators import run_siesta, relax_siesta, copy_calc_results

structures = read('results/spacegroups/structures.xyz', index=':')

with open('results/spacegroups/dft_params.json', 'r') as f:
    dft_params = json.load(f)

for structure in structures:
    if structure.calc is None:
        run_siesta(structure, **dft_params, dir='results/spacegroups')
        copy_calc_results(structure)
        write('results/spacegroups/structures.xyz', structures)

post_structures = structures[1::2]

for structure in post_structures:
    relax_siesta(structure, **dft_params, dir='results/spacegroups')
    copy_calc_results(structure)

write('results/spacegroups/structures_relaxed.xyz', post_structures)