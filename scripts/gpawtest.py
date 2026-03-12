
import os
import numpy as np
import pandas as pd
from ase.io import write
from ase.units import Ry
from gpaw import GPAW
from gpaw.mpi import world

from src.structure import Perovskite



calc_params = {
    'xc': 'PBEsol',
    'mode': {'name': 'pw', 'ecut': 50 * Ry},
    'kpts': {'size': (5, 5, 5), 'gamma': True},
    'occupations': {'name': 'fermi-dirac','width': 0.05},
    'convergence': {'density': 1e-6}
}

dir = 'results/test'

if world.rank == 0:
    # Make directory for the current q-point if it doesn't exist
    os.makedirs(dir, exist_ok=True)

atoms = Perovskite('BaTiO3', N=1).atoms

for qpoint in ['G', 'X', 'R', 'M']:
    dir_q = os.path.join(dir, qpoint)
    if world.rank == 0:
        os.makedirs(dir_q, exist_ok=True)

    amplitudes = []
    energies = []
    images = []
    for d in np.arange(0, 0.05, 0.01):
        # Make a copy of the original atoms object
        atoms_disp = atoms.copy()
        # Displace atoms 1 in the z-direction by d Angstroms
        atoms_disp.positions[1][2] += d
        images.append(atoms_disp)
        # Create calculator
        calc = GPAW(txt=os.path.join(dir, f"test.txt"), **calc_params,)
        # Attach the calculator to the atoms
        atoms_disp.calc = calc
        # Run the calculation
        energy = atoms_disp.get_potential_energy()
        amplitudes.append(d)
        energies.append(energy)

    if world.rank == 0:
        # Save the supercell structures with displacements as an xyz file
        write(os.path.join(dir_q, 'structures.xyz'), images)
        # Save amplitudes and energies as a CSV file
        df = pd.DataFrame({
            'Amplitude': amplitudes,
            'Energy': energies
        })
        df.to_csv(os.path.join(dir_q, 'energies.csv'), index=False)
    
    # Wait for all parallel processes to finish
    world.barrier()
