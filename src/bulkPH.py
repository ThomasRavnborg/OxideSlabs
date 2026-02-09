# Importing packages and modules
import numpy as np
import os
# ASE
from ase import Atoms
from ase.io import read
from ase.units import Ry
from ase.parallel import parprint
from ase.calculators.siesta import Siesta
# Phonopy
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.interface.calculator import read_crystal_structure

def calcPhonons(atoms, xcf='PBE', basis='DZP', cutoff=200, shift=0.01, kmesh=[5,5,5]):
    """Function to calculate phonon properties of a bulk structure using Phonopy and SIESTA.
    Parameters:
    - atoms: ASE Atoms object representing the relaxed bulk structure.
    - xcf: Exchange-correlation functional to be used (default is 'PBE').
    - basis: Basis set to be used (default is 'DZP').
    - cutoff: Mesh cutoff in Ry (default is 200 Ry).
    - shift: Energy shift in Ry (default is 0.01 Ry).
    - kmesh: K-point mesh as a list (default is [5, 5, 5]).
    Returns:
    - None. The function performs phonon calculations and saves the phonon data to a .yaml file.
    """
    # Parameters
    scell_matrix = np.diag([2, 2, 2])  # Supercell size
    dd = 0.01 # displacement distance in Å
    # Get current working directory
    cwd = os.getcwd()
    # Calculation parameters in a dictionary
    calc_params = {
        'label': f'{atoms.symbols}_PH',
        'xc': xcf,
        'basis_set': basis,
        'mesh_cutoff': cutoff * Ry,
        'energy_shift': shift * Ry,
        'kpts': kmesh,
        'directory': 'bulk/phonons/',
        'pseudo_path': cwd + '/pseudos'
    }
    fdf_args = {
        "MD.TypeOfRun": "CG",
        "MD.NumCGsteps": 0,  # forces only
    }
    
    # Convert ASE Atoms to PhonopyAtoms
    unitcell = PhonopyAtoms(symbols=atoms.get_chemical_symbols(),
                            positions=atoms.get_positions(),
                            cell=atoms.get_cell(),
                            masses=atoms.get_masses())
    
    # Create Phonopy object and generate displacements
    phonon = Phonopy(unitcell, scell_matrix)
    phonon.generate_displacements(distance=dd)
    supercells = phonon.supercells_with_displacements
    parprint(f"Generated {len(supercells)} supercells with displacements.")

    # Calculate forces for displaced supercells using SIESTA
    forces = []
    # Loop over all supercells and calculate forces
    for i, sc in enumerate(supercells):
        # Print which supercell is being processed
        parprint(f"Processing supercell {i + 1}/{len(supercells)}")
        
        # Convert PhonopyAtoms to ASE Atoms for each supercell
        atoms_ase = Atoms(symbols=sc.symbols,
                          positions=sc.positions,
                          cell=sc.cell,
                          pbc=True)  # Assume periodic boundary conditions
        
        # Attach the SIESTA calculator to this ASE atoms object
        calc = Siesta(**calc_params, fdf_arguments=fdf_args)
        atoms_ase.calc = calc
        
        # Calculate forces on the displaced supercell
        force = atoms_ase.get_forces()
        
        # Append forces to the list
        forces.append(force)
    
    # Set forces in Phonopy and calculate force constants
    phonon.forces = forces
    
    # Save phonopy .yaml file
    phonon.save(f'bulk/phonons/{atoms.symbols}_phonon.yaml')