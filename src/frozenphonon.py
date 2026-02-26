# Importing packages and modules
import numpy as np
import pandas as pd
# ASE
from ase.io import read
from ase import Atoms
from ase.build import make_supercell
from ase.parallel import parprint
# Phonopy
import phonopy

from src.parameterconv import run_siesta

# Defining a function to get the energy for different displacements along a symmetry direction
def get_energy(formula='BaTiO3', mode='pw', xcf='PBE', bulk=True, N=1, dis=0, q_index=0):
    """Function to calculate the energy for a given displacement along a specific q-point in the Brillouin zone.
    Parameters:
    - atomname: Name of the material (e.g., 'BaTiO3').
    - mode: Calculation mode ('pw' for plane wave or 'lcao' for linear combination of atomic orbitals).
    - xcf: Exchange-correlation functional to be used (e.g., 'PBE', 'PBEsol').
    - bulk: Boolean indicating whether the structure is bulk (True) or slab (False).
    - N: Number of unit cells in each direction for bulk (int) or thickness of the slab in unit cells (float).
    - dis: Displacement magnitude in Angstroms along the phonon mode.
    - q_index: Index of the q-point along which to displace (0 for Γ, 1 for X, 2 for M).
    Returns:
    - Energy in eV for the given displacement.
    """
    # Defining labels and symmetry points in fractional coordinates
    q_labels = ['G', 'X', 'M']
    qpoints = [[[0.0, 0.0, 0.0],   # Γ
               [0.5, 0.0, 0.0],    # X
               [0.5, 0.5, 0.0]]]   # M
    # The specific q-point of interest
    q = qpoints[0][q_index]
    # Define supercell size
    nx, ny, nz = 1, 1, 1 # supercell size
    ncells = nx*ny*nz

    # Load .xyz and .yaml file from relaxation and phonon calculation
    if bulk == True:
        atoms = read(f'bulk/relax/{formula}_pw_{xcf}.xyz')
        phonon = phonopy.load(f'bulk/phonon/{formula}_pw_{xcf}.yaml')
    else:
        atoms = read(f'slab/relax/{formula}_pw_{xcf}_d{N}.xyz')
        phonon = phonopy.load(f'slab/phonon/{formula}_{mode}_{xcf}_d{N}.yaml')
    
    # Get the atomic masses and reshape
    m = atoms.get_masses()
    m = m[:, np.newaxis]
    # Make supercell and get masses for the supercell
    supercell = make_supercell(atoms, np.diag([nx, ny, nz]))
    m_sc = np.tile(m, (ncells, 1))
    
    # Run band structure at the symmetry points
    phonon.run_band_structure(qpoints, with_eigenvectors=True)
    # Extract frequencies and eigenvectors
    band_structure = phonon.get_band_structure_dict()
    frequencies = band_structure['frequencies'][0][q_index]
    eigenvec = band_structure['eigenvectors'][0][q_index]
    
    # Determine the mode index of the unstable mode
    mode_index = np.argmin(frequencies)
    parprint("Most unstable mode index:", mode_index)
    parprint("Frequency (THz):", frequencies[mode_index])
    # Determine mode vector and reshape from (Natoms*3,) to (Natoms, 3)
    modevec = eigenvec[:, mode_index]
    modevec = modevec.reshape(len(atoms), 3)
    
    # Expand the modevector to the entire supercell (with phase)
    modevec_sc = []
    # Loop over all unit cells in the supercell
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                R = np.array([ix, iy, iz])  # R vector for phase
                phased_disp = modevec*np.exp(2j*np.pi*np.dot(q, R))
                modevec_sc.append(phased_disp)
    modevec_sc = np.vstack(modevec_sc)
    
    # Displace atomic positions in the supercell
    supercell.positions += np.real(dis/np.sqrt(m_sc*len(atoms)) * np.real(modevec_sc))
    # Set up GPAW calculator
    calc = GPAW(**mode_params, **calc_params)
    supercell.calc = calc
    # Run calculation and get potential energy (per unit cell)
    energy = supercell.get_potential_energy()/ncells
    parprint(f"Displacement {dis:.3f} Å along {q_labels[q_index]} → Energy: {energy:.6f} eV")
    return energy

def run_frozen_phonon(atomname='BaTiO3', mode='pw', xcf='PBE', bulk=True, N=1,
                      d_vals=np.array([]), q_in=np.arange(0, 1, 1)):
    # Create a DataFrame where rows correspond to kpoints and columns correspond to cutoffs
    matrix = [[get_energy(atomname, mode, xcf, bulk, N, dis, q) for q in q_in] for dis in d_vals]
    # Convert the result into a pandas DataFrame
    #df = pd.DataFrame(matrix, index=d_vals, columns=np.array(['G', 'X', 'M']))
    df = pd.DataFrame(matrix, index=d_vals, columns=np.array(['G']))
    # Save dataframe to csv
    if bulk == True:
        df.to_csv(f'doublewellv2/{atomname}_{mode}_{xcf}_dw.csv', index=True)
    else:
        df.to_csv(f'doublewellv2/{atomname}_{mode}_{xcf}_d{N}_dw.csv', index=True)