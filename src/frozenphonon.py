# Importing packages and modules
import os
import time
import copy as cp
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
from src.cleanfiles import cleanFiles
from src.phononASE import phonon_to_atoms
from src.calculators import copy_calc_results

# Try to import world from gpaw.mpi for parallel processing
# If not available, fall back to ase.parallel.world
try:
    from gpaw.mpi import world
except ImportError:
    from ase.parallel import world

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
    # Determine if the mode is stable or unstable based on the frequency
    tol = 1e-5 # Tolerance for considering a mode as stable (in THz)
    if frequencies[mode_index] > tol:
        stable = True
    else:
        stable = False
    # Determine mode vector corresponding to the most unstable mode at the given q-point
    modevec = eigenvecs[:, mode_index]
    
    # Fix arbitrary complex phase so that the mode vector is real and has the largest component positive
    imax = np.argmax(np.abs(modevec))
    phase = np.angle(modevec[imax])
    modevec = modevec * np.exp(-1j * phase)

    # Reshape from (N_atoms*3,) to (N_atoms, 3)
    modevec = modevec.reshape(N_unit, 3)
    #modevec = np.real(modevec)
    return modevec, stable

def get_modevectors(phonon, q, tol_deg=1e-4):
    """
    Extract the degenerate subspace of the lowest phonon mode at q.
    Returns a real orthonormal basis spanning that subspace.
    """

    N_unit = len(phonon.unitcell.symbols)

    phonon.run_band_structure([[q]], with_eigenvectors=True)
    band_structure = phonon.get_band_structure_dict()

    frequencies = np.squeeze(band_structure['frequencies'])
    eigenvecs = np.squeeze(band_structure['eigenvectors'])

    mode_index = np.argmin(frequencies)

    tol = 1e-5
    stable = frequencies[mode_index] > tol

    # --- find degenerate modes
    deg_indices = np.where(
        np.abs(frequencies - frequencies[mode_index]) < tol_deg
    )[0]

    subspace = []

    for i in deg_indices:

        vec = eigenvecs[:, i]

        # fix complex phase
        imax = np.argmax(np.abs(vec))
        phase = np.angle(vec[imax])
        vec = vec * np.exp(-1j * phase)

        subspace.append(vec.real)

    subspace = np.array(subspace).T  # shape (3N, n_deg)

    # --- orthonormalize real basis
    Q, _ = np.linalg.qr(subspace)

    # reshape into atomic form
    modes = []
    for i in range(Q.shape[1]):
        modes.append(Q[:, i].reshape(N_unit, 3))

    return modes, stable

def get_unstable_mode_groups(phonon, q, tol_neg=1e-5, tol_deg=1e-4):

    N_unit = len(phonon.unitcell.symbols)

    phonon.run_band_structure([[q]], with_eigenvectors=True)
    band_structure = phonon.get_band_structure_dict()

    frequencies = np.squeeze(band_structure['frequencies'])
    eigenvecs = np.squeeze(band_structure['eigenvectors'])

    unstable = np.where(frequencies < -tol_neg)[0]

    if len(unstable) == 0:
        return [], True

    groups = []
    used = set()

    for i in unstable:

        if i in used:
            continue

        deg = np.where(
            np.abs(frequencies - frequencies[i]) < tol_deg
        )[0]

        deg = [j for j in deg if j in unstable]

        used.update(deg)

        subspace = []

        for j in deg:

            vec = eigenvecs[:, j]

            imax = np.argmax(np.abs(vec))
            phase = np.angle(vec[imax])
            vec = vec * np.exp(-1j * phase)

            subspace.append(vec.real)

        subspace = np.array(subspace).T

        Q, _ = np.linalg.qr(subspace)

        modes = []

        for k in range(Q.shape[1]):
            modes.append(Q[:, k].reshape(N_unit, 3))

        groups.append({
            "frequency": frequencies[i],
            "modes": modes
        })

    return groups, False


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
    # Take the real part of the super cell mode vectors
    modevec_sc = np.real(modevec_sc)
    # Normalize by the square root of the atomic masses
    norm = np.sqrt(np.sum(m_sc * modevec_sc**2))
    modevec_sc /= norm
    #modevec_sc /= np.linalg.norm(modevec_sc)
    return modevec_sc, supercell, supercell_matrix


def calculate_frozen_phonons(phonon, n_points=10, xcf='PBEsol', basis='DZP',
                             EnergyShift=0.01, SplitNorm=0.15,
                             MeshCutoff=1000, kgrid=(10, 10, 10),
                             mode='lcao', bulk=True, deg=True,
                             dir='resultsold/bulk/frozen', par=True):
    """Function to perform frozen phonon calculations for a given Phonopy object and a range of displacement amplitudes.
    Arguments:
    - phonon (Phonopy): Phonopy object containing the phonon calculation results.
    - n_points (int): Number of displacement points - roughly (default: 10).
    - xcf (str): Exchange-correlation functional to use in the calculations (default: 'PBEsol').
    - basis (str): Basis set to use in the calculations (default: 'DZP').
    - EnergyShift (float): Energy shift parameter for the SIESTA calculations (in Ry, default: 0.01 Ry).
    - SplitNorm (float): Split norm parameter for the SIESTA calculations (default: 0.15).
    - MeshCutoff (float): Mesh cutoff for the SIESTA calculations (in Ry, default: 1000 Ry).
    - kgrid (tuple): Tuple specifying the k-point grid size for the SIESTA calculations (default: (10, 10, 10)).
    - mode (str): String indicating whether to use localized atomic orbitals ('lcao') or plane waves ('pw') for the calculations (default: 'lcao').
    - deg (bool): Boolean indicating whether to consider degenerate modes at the q-points (default: True).
    - dir (str): Directory where the results will be saved (default: 'resultsold/bulk/frozen').
    - par (bool): Boolean indicating whether to run calculations in parallel (default: True).
    Returns:
    - None (results are saved to files in the specified directory).
    """
    # Get current working directory (cwd)
    cwd = os.getcwd()
    # Unitcell and formula from phonon object
    unitcell = phonon_to_atoms(phonon, cell='unit')
    formula = unitcell.symbols
    #symbols = phonon.unitcell.symbols

    # Custom basis sets ending with 'p' are generated with the same parameters as the standard basis sets
    # However, an extra polarization (d) orbital is added to the A-site during LCAO basis generation
    if basis.endswith('p'):
        basis = basis[:-1]

    # Dictionary for q-points
    q_dict = {
        'G': [0.0, 0.0, 0.0],
        'X': [0.5, 0.0, 0.0],
        'R': [0.5, 0.5, 0.5],
        'M': [0.5, 0.5, 0.0],
    }

    dd_dict = {
        'G': 1/n_points,
        'X': 1.5/n_points,
        'R': 5/n_points,
        'M': 3/n_points,
    }

    if not bulk:
        # Remove 'R' from dictionaries
        q_dict.pop('R')
        dd_dict.pop('R')

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

        dir_fdf = os.path.join(cwd, os.path.dirname(dir))
        # fdf arguments in a dictionary
        fdf_args = {
            '%include': os.path.join(dir_fdf, 'basis.fdf'),
            'PAO.SplitNorm': SplitNorm,
            'SCF.DM.Tolerance': 1e-6,
        }
        if par:
            # Change diagonalization algorithm when running in parallel
            fdf_args['Diag.Algorithm'] = 'ELPA'
    
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
        # Define coordinates and displacement distance for the current q-point
        q = q_dict[qpoint]
        dd = dd_dict[qpoint]
        # Get mode vector and stability of the mode at the given q-point
        #modevec, stable = get_modevector(phonon, q)
        #modes, stable = get_modevectors(phonon, q)

        # Get groups of degenerate unstable modes at the given q-point
        groups, stable = get_unstable_mode_groups(phonon, q)
        # Count number of unstable modes at the given q-point
        n_unstable = len(groups)

        if stable:
            parprint(f"No unstable modes found at {qpoint}. Skipping frozen phonon calculation.", flush=True)
            continue
        
        parprint(f"{n_unstable} unstable mode(s) found at {qpoint}. Starting frozen phonon calculation.", flush=True)

        for g_id, group in enumerate(groups):

            modes = group["modes"]
            freq = group["frequency"]

            dir_group = os.path.join(dir_q, f"mode_{g_id+1}")

            if world.rank == 0:
                os.makedirs(dir_group, exist_ok=True)
                # Save the frequency of the mode to a text file
                with open(os.path.join(dir_group, "freq.txt"), "w") as f:
                    f.write(f"{freq:.6f} THz")

            if not deg:
                modes = [modes[0]] # Only consider the first mode if degenerate modes are not considered

            for mode_id, modevec in enumerate(modes):

                dir_mode = os.path.join(dir_group, f"Q_{mode_id+1}")

                
                if world.rank == 0:
                    try:
                        os.makedirs(dir_mode, exist_ok=False)
                    except FileExistsError:
                        parprint(f"Directory {dir_mode} already exists. Skipping calculation for this mode.", flush=True)
                        continue
                
                modevec_sc, supercell, supercell_matrix = get_displacement(unitcell, q, modevec)

                # Generate the supercell and get the mode vector for the supercell
                #modevec_sc, supercell, supercell_matrix = get_displacement(unitcell, q, modevec)
                # Determine the supercell size in each direction from the diagonal of the supercell matrix
                nx, ny, nz = supercell_matrix.diagonal().astype(int)
                ncells = nx*ny*nz
                # Determine the k-point grid size for the SIESTA calculation based on the supercell size
                kx, ky, kz = max(1, kgrid[0]//nx), max(1, kgrid[1]//ny), max(1, kgrid[2]//nz)

                # Set up the calculator for the supercell with displacements based on the specified mode (LCAO or PW)
                if mode == 'lcao':
                    # Set up the Siesta calculator
                    calc = Siesta(**calc_params, fdf_arguments=fdf_args,
                                  kpts=(kx, ky, kz), directory=dir_mode)
                elif mode == 'pw':
                    calc = GPAW(txt=os.path.join(dir_mode, f"{formula}.txt"), **calc_params,
                                kpts={'size': (kx, ky, kz), 'gamma': True}, symmetry='off')

                supercell_disp = supercell.copy()
                ref_positions = supercell.positions.copy()
                supercell_disp.calc = calc

                amp = 0
                amplitudes = []
                energies = []
                images = []
                if world.rank == 0:
                    t0 = time.time() # Start timer
                while True:
                    # Displace the atoms according to the mode vector by dd
                    supercell_disp.positions = ref_positions + amp * modevec_sc
                    
                    # Run the calculation
                    energy = supercell_disp.get_potential_energy()

                    # Scale energy by the number of unit cells in the supercell to get energy per unit cell
                    energy = energy / ncells
                    energies.append(energy)
                    # Append the supercell structure with displacements, forces and stresses to the list of images
                    img = copy_calc_results(supercell_disp)
                    images.append(img)
                    # Append amplitude and update for the next iteration
                    amplitudes.append(amp)
                    amp += dd
                    tol = 50*1e-3 # Tolerance for stopping the loop based on energy increase (in eV)
                    # Stop the loop, if the energy has increased by more than the tolerance compared to the first point
                    if len(energies) > 1 and energies[-1] - energies[0] > tol:
                        break
                
                # Save the supercell structures with displacements as an xyz file
                write(os.path.join(dir_mode, 'structures.xyz'), images)

                if world.rank == 0:
                    t1 = time.time() # Stop timer
                    # Save amplitudes and energies as a CSV file
                    df = pd.DataFrame({
                        'Amplitude': amplitudes,
                        'Energy': energies
                    })
                    df.to_csv(os.path.join(dir_mode, 'energies.csv'), index=False)
                    # Write the time taken for frozen phonon calculations to a file
                    np.save(os.path.join(dir_mode, f"time.npy"), t1-t0)
                if mode == 'lcao':
                    # Clean directory of SIESTA calculations
                    cleanFiles(directory=dir_mode, confirm=False)
                
                # Wait for all parallel processes to finish
                world.barrier()

    # After all calculations are complete, write a completion message to a text file in the main directory
    if world.rank == 0:
        with open(os.path.join(dir, "complete.txt"), "w") as f:
            f.write("Frozen phonon calculations complete.")
