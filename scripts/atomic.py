import os
import re
import json
from ase import Atoms
from ase.units import Ry
from ase.calculators.siesta import Siesta
from src.structure import Perovskite
from src.parameterconv import run_siesta
from src.cleanfiles import cleanFiles
from src.fdfcreate import generate_basis

def calculate_atomic_energies(formula, xcf='PBEsol', basis='DZPp',
                              EnergyShift=0.01, SplitNorm=0.15,
                              MeshCutoff=1000, kgrid=(12, 12, 12), dir=''):

    symbols = re.findall(r'[A-Z][a-z]*', formula)

    energies = {}
    for j, symbol in enumerate(symbols):
        dir_symbol = os.path.join(dir, symbol)
        atom = Atoms(symbols=symbol)
        atom.center(vacuum=5.0)
        # If basis ends with 'p' and this is not the first atom, remove the 'p' to avoid generating a polarized basis for subsequent atoms
        if j != 0 and basis.endswith('p'):
            basis = basis[:-1]
        # Generate the basis set for the test atom and save it in the specified directory
        generate_basis(atom, xcf, basis, EnergyShift, SplitNorm, dir=dir_symbol)
        # Add vacuum to the atom to ensure it is isolated for the calculation
        atom.center(vacuum=15.0)
        # Run the test calculation for the atom and store the energy in the dictionary
        energy = run_siesta(atom, xcf, basis, EnergyShift, SplitNorm,
                            MeshCutoff, kgrid, dir=dir_symbol)
        energies[symbol] = energy

    with open(os.path.join(dir, 'energies.json'), 'w') as f:
        json.dump(energies, f)

calculate_atomic_energies('BaTiO3', dir='results/bulk/BaTiO3/0082/atomic')