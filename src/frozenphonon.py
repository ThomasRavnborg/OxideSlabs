# Importing packages and modules
import os
import time
import numpy as np
import pandas as pd
import sisl as si
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
from phonopy import Phonopy
# Custom modules
from src.parameterconv import run_siesta
from src.cleanfiles import cleanFiles
from src.phononcalc import ase_to_phonopy, phonopy_to_ase

def phonon_to_atoms(phonon, cell='unit'):
    """Function to convert a Phonopy object to an ASE Atoms object.
    Parameters:
    - phonon: Phonopy object containing the phonon calculation results.
    - cell: String indicating whether to use the 'unit' cell or 'super' cell from the Phonopy object (default: 'unit').
    Returns:
    - atoms: ASE Atoms object containing the structure of the specified cell from the Phonopy object.
    """
    # Determine which cell to use based on the input argument
    if cell == 'unit':
        cell = phonon.unitcell
    elif cell == 'super':
        cell = phonon.supercell
    # Convert the Phonopy cell to an ASE Atoms object
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
    #print(f"Frequencies at q={q}: {frequencies}")
    eigenvecs = np.squeeze(band_structure['eigenvectors'])

    # Determine the mode index of the unstable mode
    mode_index = np.argmin(frequencies)
    print(frequencies[mode_index])
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


def get_displacement(unitcell, q, modevec):
    """Function to generate a supercell and get displacements according to the mode vector and q-point.
    Parameters:
    - unitcell: ASE Atoms object containing the unit cell structure.
    - q: q-point (in fractional coordinates) at which the mode vector is defined.
    - modevec: Mode vector corresponding to the most unstable mode at the given q-point.
    Returns:
    - modevec_sc: Mode vector expanded to the entire supercell, including the phase factor from the q-point and normalized by the square root of the atomic masses.
    - supercell: ASE Atoms object containing the supercell structure with displaced atomic positions.
    - supercell_matrix: 3x3 matrix defining the supercell transformation from the unit cell.
    """
    
    # Determine supercell size required for the given q-point
    q_inv = np.array([int(1/q_i) if q_i != 0 else 1 for q_i in q])
    nx, ny, nz = q_inv[0], q_inv[1], q_inv[2]
    ncells = nx*ny*nz

    # Get the atomic masses and reshape
    m = unitcell.get_masses()
    m = m[:, np.newaxis]
    # Make supercell and get masses for the supercell
    supercell_matrix = np.diag([nx, ny, nz])
    supercell = make_supercell(unitcell, supercell_matrix)
    N_a = len(supercell)
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
    # Take the real part of the super cell mode vectors and normalize by the square root of the atomic masses
    modevec_sc = np.real(modevec_sc) / np.sqrt(m_sc * N_a)
    #modevec_sc /= np.linalg.norm(modevec_sc)
    return modevec_sc, supercell, supercell_matrix

def calculate_frozen_phonons(phonon, dd=0.1, xcf='PBEsol', basis='DZP',
                             EnergyShift=0.01, SplitNorm=0.15,
                             MeshCutoff=1000, kgrid=(10, 10, 10),
                             pseudo='PBEsol', mode='lcao',
                             dir='resultsold/bulk/frozen'):
    """Function to perform frozen phonon calculations for a given Phonopy object and a range of displacement amplitudes.
    Parameters:
    - phonon: Phonopy object containing the phonon calculation results.
    - dd: Displacement spacing in Å(a.u.)^1/2 (default: 0.1).
    - xcf: Exchange-correlation functional to use in the calculations (default: 'PBEsol').
    - basis: Basis set to use in the calculations (default: 'DZP').
    - EnergyShift: Energy shift parameter for the SIESTA calculations (in Ry, default: 0.01 Ry).
    - SplitNorm: Split norm parameter for the SIESTA calculations (default: 0.15).
    - MeshCutoff: Mesh cutoff for the SIESTA calculations (in Ry, default: 1000 Ry).
    - kgrid: Tuple specifying the k-point grid size for the SIESTA calculations (default: (10, 10, 10)).
    - pseudo: Pseudopotential to use for the SIESTA calculations (default: 'PBEsol').
    - mode: String indicating whether to use localized atomic orbitals ('lcao') or plane waves ('pw') for the calculations (default: 'lcao').
    - dir: Directory where the results will be saved (default: 'resultsold/bulk/frozen').
    Returns:
    - None (results are saved to files in the specified directory).
    """
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

    # In SIESTA, calculations are performed with localized atomic orbitals (LCAO)
    if mode == 'lcao':
        # Calculation parameters in a dictionary
        calc_params = {
            'label': f'{formula}',
            'xc': xcf,
            'basis_set': basis,
            'mesh_cutoff': MeshCutoff * Ry,
            'energy_shift': EnergyShift * Ry,
            'pseudo_path': os.path.join(cwd, f'pseudos/{xcf}')
        }

        # FDF arguments in a dictionary
        fdf_args = {
            'PAO.BasisSize': basis,
            'PAO.SplitNorm': SplitNorm,
            'SCF.DM.Tolerance': 1e-6,
        }

    # In GPAW, calculations are performed with plane waves (PW)
    elif mode == 'pw':
        from gpaw import GPAW
        calc_params = {
            'xc': xcf,
            'basis': basis.lower(),
            'mode': {'name': 'pw', 'ecut': MeshCutoff * Ry},
            'occupations': {'name': 'fermi-dirac','width': 0.05},
            'convergence': {'density': 1e-6}
        }

    # Loop over q-points and perform frozen phonon calculations for each q-point
    for qpoint in q_dict.keys():
        # Set the results directory for the current q-point
        dir_q = os.path.join(dir, qpoint)
        q = q_dict[qpoint]
        # Get mode vector and stability of the mode at the given q-point
        modevec, stable = get_modevector(phonon, q)
        # If the mode is stable, skip the frozen phonon calculations for this q-point
        if stable:
            print(f"The mode at q-point {qpoint} is stable. Skipping frozen phonon calculations.")
        else:
            print(f"Calculating frozen phonons for q-point {qpoint}.")
            # Make directory for the current q-point if it doesn't exist
            os.makedirs(dir_q, exist_ok=True)
            
            # Generate the supercell and get the mode vector for the supercell
            modevec_sc, supercell, supercell_matrix = get_displacement(unitcell, q, modevec)
            # Determine the supercell size in each direction from the diagonal of the supercell matrix
            nx, ny, nz = supercell_matrix.diagonal().astype(int)
            ncells = nx*ny*nz
            # Determine the k-point grid size for the SIESTA calculation based on the supercell size
            kx, ky, kz = max(1, kgrid[0]//nx), max(1, kgrid[1]//ny), max(1, kgrid[2]//nz)

            amp = 0
            amplitudes = []
            images = []
            energies = []
            t0 = time.time() # Start timer
            while True:
                amplitudes.append(amp)
                # Make a copy of the supercell to displace
                supercell_disp = supercell.copy()
                # Displace atomic positions in the supercell according to the mode vector and displacement amplitude
                displacement = amp * modevec_sc
                #displacements.append(displacement)
                supercell_disp.positions += displacement
                images.append(supercell_disp)

                if mode == 'lcao':
                    # Set k-points for SIESTA calculation based on the supercell size
                    calc_params['kpts'] = (kx, ky, kz)
                    # Set up the Siesta calculator
                    calc = Siesta(directory=dir_q, **calc_params, fdf_arguments=fdf_args)
                elif mode == 'pw':
                    # Set k-points for GPAW calculation based on the supercell size
                    calc_params['kpts'] = {'size': (kx, ky, kz), 'gamma': True}
                    # Set up the GPAW calculator
                    calc = GPAW(txt=os.path.join(dir_q, f"{formula}.txt"), **calc_params)
                
                # Attach the calculator to the supercell
                supercell_disp.calc = calc
                # Run the calculation
                energy = supercell_disp.get_potential_energy()
                # Scale energy by the number of unit cells in the supercell to get energy per unit cell
                energy = energy / ncells
                energies.append(energy)
                """
                if mode == 'lcao':
                    # Read forces from the calculation output
                    sile = si.get_sile(os.path.join(dir_q, f'{formula}.FA'))
                    force = sile.read_force()
                elif mode == 'pw':
                    force = supercell_disp.get_forces()
                forces.append(force)
                """
                # Update amplitude for the next iteration
                amp += dd
                tol = 50*1e-3 # Tolerance for stopping the loop based on energy increase (in eV)
                # Stop the loop, if the energy has increased by more than the tolerance compared to the first point
                if len(energies) > 1 and energies[-1] - energies[0] > tol:
                    break
            t1 = time.time() # Stop timer
            # Save the supercell structures with displacements as an xyz file
            write(os.path.join(dir_q, 'structures.xyz'), images)
            # Save amplitudes and energies as a CSV file
            df = pd.DataFrame({
                'Amplitude': amplitudes,
                'Energy': energies
            })
            df.to_csv(os.path.join(dir_q, 'energies.csv'), index=False)
            # Save the displacements and forces for each displacement as a numpy array
            #np.savez(os.path.join(dir_q, 'forces.npz'), displacements=displacements, forces=forces)
            # Write the time taken for frozen phonon calculations to a file
            np.save(os.path.join(dir_q, f"time.npy"), t1-t0)
            if mode == 'lcao':
                # Clean directory of SIESTA calculations
                cleanFiles(directory=dir_q, confirm=False)
    





def run_frozen_phonons(formula, id, qpoint, displacements, skip=True):
    """Legacy function, currently not in use."""
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


    modevec_sc, supercell, supercell_matrix = get_displacement(unitcell, q, modevec)


    # Loop over displacements, generate supercell with displaced atomic positions, run Siesta calculation, and save results
    images = []
    t0 = time.time() # Start timer
    for dis in displacements:
        # make a copy of the supercell to displace
        supercell_disp = supercell.copy()
        # Displace atomic positions in the supercell according to the mode vector and displacement amplitude
        supercell_disp.positions += dis * modevec_sc
        images.append(supercell_disp)
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
            energy = run_siesta(formula, supercell_disp, EnergyShift=0.01, SplitNorm=0.15,
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