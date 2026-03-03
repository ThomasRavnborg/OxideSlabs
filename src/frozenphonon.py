# Importing packages and modules
import os
import time
import numpy as np
import pandas as pd
from itertools import product
# ASE
from ase.io import read, write
from ase import Atoms
from ase.units import Ry
from ase.build import make_supercell
from ase.calculators.siesta import Siesta
from ase.parallel import parprint
# Phonopy
import phonopy as ph
# Custom modules
from src.parameterconv import run_siesta
from src.cleanfiles import cleanFiles

def phonon_to_atoms(phonon, cell='unit'):
    if cell == 'unit':
        cell = phonon.unitcell
    elif cell == 'super':
        cell = phonon.supercell
    atoms = Atoms(
        symbols=cell.symbols,
        scaled_positions=cell.scaled_positions,
        cell=cell.cell,
        pbc=True
    )
    return atoms

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
    tol = 1e-5 # Tolerance for considering a mode as stable (in THz)
    if frequencies[mode_index] > tol:
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
    q_inv = np.array([int(1/q_i) if q_i != 0 else 1 for q_i in q])
    nx, ny, nz = q_inv[0], q_inv[1], q_inv[2]

    # ---------------------------------------
    #nx, ny, nz = 2, 2, 2   # temporary, to test the code
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
    supercell.positions += dis * np.real(modevec_sc) / np.sqrt(m_sc*N_unit)
    return supercell, supercell_matrix


def run_frozen_phonons(formula, id, qpoint, displacements, skip=True):
    """Function to compute frozen phonon energies by running multiple Siesta calculations.
    Parameters:
    - formula: Chemical formula of the material (e.g., 'SrTiO3').
    - id: Unique identifier for the calculation (e.g., '0001').
    - qpoint: q-point at which to compute frozen phonon energies (e.g., 'G', 'X', 'R', 'M').
    - displacements: np.array of displacement amplitudes (in Å) to use for frozen phonon calculations.
    - skip: Boolean indicating whether to skip calculations if the mode at the given q-point is stable (default: True).
    Returns:
    - None. The function runs multiple calculations and saves the results to a CSV and xyz file.
    """
    # Set the results directory
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
    if stable and skip:
        print(f"The mode at q-point {qpoint} is stable. Skipping frozen phonon calculations.")
        return
    else:
        # Make directory for frozen phonon results if it doesn't exist
        os.makedirs(dir_res, exist_ok=True)
    # Check if results already exist for the given q-point and displacements, and load them if they do
    if os.path.exists(os.path.join(dir_res, 'frozen.csv')):
        df = pd.read_csv(os.path.join(dir_res, 'frozen.csv'))
    else:
        df = pd.DataFrame(columns=['displacement', 'Energy'])

    # Loop over displacements, generate supercell with displaced atomic positions, run Siesta calculation, and save results
    images = []
    t0 = time.time() # Start timer
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
            # Determine the k-point grid size for the SIESTA calculation based on the supercell size
            kx, ky, kz = max(1, 6//nx), max(1, 6//ny), max(1, 6//nz)
            
            # Run a single-point SIESTA calculation to get the total energy
            # Note that the k-point grid is scaled according to the supercell size
            # This means fewer k-points will be used for larger supercells
            energy = run_siesta(formula, supercell, EnergyShift=0.01, SplitNorm=0.15,
                                MeshCutoff=200, kgrid=(kx, ky, kz), dir=dir_res)
            # Scale energy by the number of unit cells in the supercell to get energy per unit cell
            energy = energy / (nx*ny*nz)

            # Append results
            row = {
                "displacement": dis,
                "Energy": energy
            }
            # Create new dataframe
            df_new = pd.DataFrame([row])
            # Update old dataframe with new results
            df = pd.concat([df, df_new], ignore_index=True)
            # Save new results
            df.to_csv(os.path.join(dir_res, 'frozen.csv'), index=False)
    
    t1 = time.time() # Stop timer
    # Save the supercell structures with displacements as an xyz file
    write(os.path.join(dir_res, 'frozen.xyz'), images)
    # Write the time taken for frozen phonon calculations to a file
    np.savez(os.path.join(dir_res, f"time.npz"), dt=t1-t0)
    # Clean directory of SIESTA calculations
    cleanFiles(directory=dir_res, confirm=False)




def calculate_frozen_phonons(phonon, displacements, xcf='PBEsol', basis='DZP',
                             EnergyShift=0.01, SplitNorm=0.15,
                             MeshCutoff=200, kgrid=(10, 10, 10),
                             pseudo='PBEsol', mode='lcao',
                             dir='resultsold/bulk/frozen'):
    # Get current working directory (cwd)
    cwd = os.getcwd()
    # Unitcell and formula from phonon object
    unitcell = phonon_to_atoms(phonon, cell='unit')
    formula = unitcell.symbols

    # Dictionary for q-points
    q_dict = {
        'G': [0.0, 0.0, 0.0],
        'X': [0.5, 0.0, 0.0],
        'R': [0.5, 0.5, 0.5],
        'M': [0.5, 0.5, 0.0],
    }

    qpoint = 'G' # Temporary, to test the code

    # Get q-point in fractional coordinates
    q = q_dict[qpoint]
    # Get mode vector and stability of the mode at the given q-point
    modevec, stable = get_modevector(phonon, q)
    if stable:
        print(f"The mode at q-point {qpoint} is stable. Skipping frozen phonon calculations.")
        return
    else:
        # Make directory for frozen phonon results if it doesn't exist
        os.makedirs(dir, exist_ok=True)
    # Check if results already exist for the given q-point and displacements, and load them if they do
    if os.path.exists(os.path.join(dir, 'frozen.csv')):
        df = pd.read_csv(os.path.join(dir, 'frozen.csv'))
    else:
        df = pd.DataFrame(columns=['displacement', 'Energy'])

    # In SIESTA, calculations are performed with localized atomic orbitals (LCAO)
    if mode == 'lcao':
        # Calculation parameters in a dictionary
        calc_params = {
            'label': f'{formula}',
            'xc': xcf,
            'basis_set': basis,
            'mesh_cutoff': MeshCutoff * Ry,
            'energy_shift': EnergyShift * Ry,
            'kpts': kgrid,
            'directory': dir,
            'pseudo_path': os.path.join(cwd, f'pseudos/{xcf}')
        }

        # FDF arguments in a dictionary
        fdf_args = {
            'PAO.BasisSize': basis,
            'PAO.SplitNorm': SplitNorm,
            'SCF.DM.Tolerance': 1e-6,
        }

    
    elif mode == 'pw':
        from gpaw import GPAW
        calc_params = {
            'xc': xcf,
            'basis': basis.lower(),
            'mode': {'name': 'pw', 'ecut': MeshCutoff * Ry},
            'kpts': {'size': kgrid, 'gamma': True},
            'occupations': {'name': 'fermi-dirac','width': 0.05},
            'convergence': {'density': 1e-6},
            'txt': os.path.join(dir, f"{formula}_PH.txt")
        }

    # Loop over displacements, generate supercell with displaced atomic positions, run Siesta calculation, and save results
    images = []
    t0 = time.time() # Start timer
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
            # Determine the k-point grid size for the SIESTA calculation based on the supercell size
            kx, ky, kz = max(1, 6//nx), max(1, 6//ny), max(1, 6//nz)
            
             # Set up the calculator based on the selected mode
            if mode == 'lcao':
                # Set up the Siesta calculator
                calc = Siesta(**calc_params, fdf_arguments=fdf_args)
            elif mode == 'pw':
                # Set up the GPAW calculator
                calc = GPAW(**calc_params)
            supercell.calc = calc
            # Run the calculation
            energy = supercell.get_potential_energy()
            # Scale energy by the number of unit cells in the supercell to get energy per unit cell
            energy = energy / (nx*ny*nz)

            # Append results
            row = {
                "displacement": dis,
                "Energy": energy
            }
            # Create new dataframe
            df_new = pd.DataFrame([row])
            # Update old dataframe with new results
            df = pd.concat([df, df_new], ignore_index=True)
            # Save new results
            df.to_csv(os.path.join(dir, 'frozen.csv'), index=False)
    
    t1 = time.time() # Stop timer
    # Save the supercell structures with displacements as an xyz file
    write(os.path.join(dir, 'frozen.xyz'), images)
    # Write the time taken for frozen phonon calculations to a file
    np.savez(os.path.join(dir, f"time.npz"), dt=t1-t0)
    # Clean directory of SIESTA calculations
    cleanFiles(directory=dir, confirm=False)
    