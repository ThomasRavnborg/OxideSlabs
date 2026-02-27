# Importing packages and modules
import os
import numpy as np
import pandas as pd
from itertools import product
# ASE
from ase.io import read, write
from ase import Atoms
from ase.build import make_supercell
from ase.parallel import parprint
# Phonopy
import phonopy as ph

from src.parameterconv import run_siesta
from src.cleanfiles import cleanFiles


def get_modevector(phonon, q):
    """Function to extract the mode vector corresponding to the lowest frequency mode at a given q-point.
    Parameters:
    - phonon: Phonopy object containing the phonon calculation results.
    - q: q-point (in fractional coordinates) at which to extract the mode vector.
    Returns:
    - modevec: Mode vector corresponding to the most unstable mode at the given q-point.
    - stable: Boolean indicating whether the mode is stable (True) or unstable (False).
    """
    # Determine number of atoms in the unitcell
    N_unit = len(phonon.unitcell.symbols)

    # Run band structure at the symmetry point to get eigenvectors
    phonon.run_band_structure([[q]], with_eigenvectors=True)
    # Extract frequencies and eigenvectors
    band_structure = phonon.get_band_structure_dict()
    frequencies = np.squeeze(band_structure['frequencies'])
    eigenvecs = np.squeeze(band_structure['eigenvectors'])

    # Determine the mode index of the unstable mode
    mode_index = np.argmin(frequencies)
    # Determine if the mode is stable or unstable based on the frequency
    if frequencies[mode_index] > 0:
        stable = True
    else:
        stable = False
    # Determine mode vector and reshape from (N_atoms*3,) to (N_atoms, 3)
    modevec = eigenvecs[:, mode_index]
    modevec = modevec.reshape(N_unit, 3)
    return modevec, stable


def displace_atoms(unitcell, q, modevec, dis):
    """Function to generate a supercell and displace the atomic positions according to the mode vector and q-point.
    Parameters:
    - unitcell: ASE Atoms object containing the unit cell structure.
    - q: q-point (in fractional coordinates) at which the mode vector is defined.
    - modevec: Mode vector corresponding to the most unstable mode at the given q-point.
    - dis: Displacement amplitude (in Å) to apply to the atomic positions.
    Returns:
    - supercell: ASE Atoms object containing the supercell structure with displaced atomic positions.
    - supercell_matrix: 3x3 matrix defining the supercell transformation from the unit cell.
    """
    
    # Determine number of atoms in the unitcell
    N_unit = len(unitcell)

    # Determine supercell size required for the given q-point
    # If q[i] is 0, we can set the supercell size to 1 in that direction
    nx = int(1/q[0]) if q[0] != 0 else 1
    ny = int(1/q[1]) if q[1] != 0 else 1
    nz = int(1/q[2]) if q[2] != 0 else 1

    # ---------------------------------------
    nx, ny, nz = 2, 2, 2   # temporary, to test the code
    # ---------------------------------------

    ncells = nx*ny*nz

    # Get the atomic masses and reshape
    m = unitcell.get_masses()
    m = m[:, np.newaxis]
    # Make supercell and get masses for the supercell
    supercell_matrix = np.diag([nx, ny, nz])
    supercell = make_supercell(unitcell, supercell_matrix)
    m_sc = np.tile(m, (ncells, 1))
    
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
    supercell.positions += np.real(dis/np.sqrt(m_sc*N_unit) * np.real(modevec_sc))
    return supercell, supercell_matrix


def run_frozen_phonons(formula, id, qpoint, displacements):
    """Function to compute frozen phonon energies by running multiple Siesta calculations.
    Parameters:
    - formula: Chemical formula of the material (e.g., 'SrTiO3').
    - id: Unique identifier for the calculation (e.g., '0001').
    - qpoint: q-point at which to compute frozen phonon energies (e.g., 'G', 'X', 'R', 'M').
    - displacements: List of displacement amplitudes (in Å) to use for frozen phonon calculations.
    Returns:
    - None. The function runs multiple calculations and saves the results to a CSV and xyz file.
    """
    # Set the results directory
    #dir_res = os.path.join('results/bulk/',formula,'frozen', qpoint)
    dir_id = os.path.join('results/bulk/',formula, id)
    dir_res = os.path.join(dir_id, 'frozen', qpoint)

    # Load .xyz and .yaml file from relaxation and phonon calculation
    unitcell = read(os.path.join(dir_id, f'relax/{formula}.xyz'))
    phonon = ph.load(os.path.join(dir_id, f'phonons/{formula}.yaml'))

    # Dictionary for q-points
    q_dict = {
        'G': [0.0, 0.0, 0.0],
        'X': [0.5, 0.0, 0.0],
        'R': [0.5, 0.5, 0.5],
        'M': [0.5, 0.5, 0.0],
    }
    # Get q-point in fractional coordinates
    q = q_dict[qpoint]
    # Get mode vector and stability of the mode at the given q-point
    modevec, stable = get_modevector(phonon, q)
    if stable:
        print(f"The mode at q-point {qpoint} is stable. No frozen phonon calculations will be run.")
        return

    if os.path.exists(os.path.join(dir_res, 'frozen.csv')):
        df = pd.read_csv(os.path.join(dir_res, 'frozen.csv'))
    else:
        df = pd.DataFrame(columns=['displacement', 'Energy'])

    images = []

    for dis in displacements:
        # Get supercell with displaced atomic positions for the given displacement amplitude
        supercell, supercell_matrix = displace_atoms(unitcell, q, modevec, dis)
        images.append(supercell)
        # Check if results have been obtained for a given displacement
        if (df['displacement'] == dis).any():
            print(f"qpoint={qpoint}, displacement={dis} is in the DataFrame. Skipping.")
        else:
            # Determine the supercell size in each direction from the diagonal of the supercell matrix
            nx, ny, nz = supercell_matrix.diagonal().astype(int)
            
            # Run a single-point SIESTA calculation to get the total energy
            # Note that the k-point grid is scaled according to the supercell size
            # This means fewer k-points will be used for larger supercells
            energy = run_siesta(supercell, EnergyShift=0.001, SplitNorm=0.1,
                                MeshCutoff=1000, kgrid=(10/nx, 10/ny, 10/nz), dir=dir_res)
            # Scale energy by the number of unit cells in the supercell to get energy per unit cell
            energy = energy / (nx*ny*nz)

            # Append results
            row = {
                "displacement": dis,
                "Energy": energy
            }
            # Create new dataframe
            df_new = pd.DataFrame([row])
            # Update old datafrem with new results
            df = pd.concat([df, df_new], ignore_index=True)
            # Save new results
            df.to_csv(os.path.join(dir_res, 'frozen.csv'), index=False)

    # Save the supercell structures with displacements as an xyz file
    write(os.path.join(dir_res, 'frozen.xyz'), images)
    # Clean directory of SIESTA calculations
    cleanFiles(directory=dir_res, confirm=False)