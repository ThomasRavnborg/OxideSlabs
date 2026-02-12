# Importing packages and modules
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
# ASE
from ase import Atoms
from ase.io import read
from ase.units import Ry
from ase.parallel import parprint
from ase.calculators.siesta import Siesta
# GPAW
from gpaw import GPAW
# Phonopy
from phonopy import Phonopy
from phonopy.phonon.band_structure import get_band_qpoints_and_path_connections
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.interface.calculator import read_crystal_structure
# Custom modules
from src.cleanfiles import cleanFiles
from src.plotsettings import PlotSettings
PlotSettings().set_global_style()

def calculate_phonons(atoms, xcf='PBE', basis='DZP', shift=0.01, split=0.15,
                      cutoff=200, kmesh=[5, 5, 5], mode='lcao'):
    """Function to calculate phonon properties of a bulk structure using Phonopy and SIESTA.
    Parameters:
    - atoms: ASE Atoms object representing the relaxed bulk structure.
    - xcf: Exchange-correlation functional to be used (default is 'PBE').
    - basis: Basis set to be used (default is 'DZP').
    - shift: Energy shift in Ry (default is 0.01 Ry).
    - split: Split norm for basis functions (default is 0.15).
    - cutoff: Mesh cutoff in Ry (default is 200 Ry).
    - kmesh: K-point mesh as a list (default is [5, 5, 5]).
    - mode: Calculator mode to be used ('lcao' for SIESTA or 'pw' for GPAW, default is 'lcao').
    Returns:
    - None. The function performs phonon calculations and saves the phonon data to a .yaml file.
    """
    # Parameters
    scell_matrix = np.diag([2, 2, 2])  # Supercell size
    dd = 0.01 # displacement distance in Å

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

    # Get current working directory and set directory for results
    cwd = os.getcwd()
    dir = 'results/bulk/phonons/'
    symbols = atoms.symbols
    # In SIESTA, calculations are performed with localized atomic orbitals (LCAO)
    if mode == 'lcao':
        # Calculation parameters
        calc_params = {
            'label': f'{symbols}',
            'xc': xcf,
            'basis_set': basis,
            'mesh_cutoff': cutoff * Ry,
            'energy_shift': shift * Ry,
            'kpts': kmesh,
            'directory': dir,
            'pseudo_path': cwd + '/pseudos'
        }
        # fdf arguments
        fdf_args = {
            'PAO.BasisSize': basis,
            'PAO.SplitNorm': split,
            'Diag.Algorithm': 'ELPA',
            "MD.TypeOfRun": "CG",
            "MD.NumCGsteps": 0,  # forces only
        }
        # Set up the Siesta calculator
        calc = Siesta(**calc_params, fdf_arguments=fdf_args)
    
    elif mode == 'pw':
        calc_params = {
            'xc': xcf,
            'basis': basis.lower(),
            'mode': {'name': 'pw', 'ecut': cutoff * Ry},
            'kpts': {'size': kmesh, 'gamma': True},
            'occupations': {'name': 'fermi-dirac','width': 0.05},
            'convergence': {'density': 1e-6, 'forces': 1e-5},
            'txt': f"{dir}{symbols}_{mode}.txt"
        }
        # Set up the GPAW calculator
        calc = GPAW(**calc_params)

    # Calculate forces for displaced supercells
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
        
        # Attach the calculator to this ASE atoms object
        atoms_ase.calc = calc
        
        # Calculate forces on the displaced supercell
        force = atoms_ase.get_forces()
        
        # Append forces to the list
        forces.append(force)
    
    # Set forces in Phonopy and calculate force constants
    phonon.forces = forces
    # Save phonopy .yaml file
    phonon.save(f'{dir}{symbols}_{mode}.yaml')
    # Remove unnecessary files generated during the relaxation
    cleanFiles(directory=dir, confirm=False)

def order_labels(symbols, handles, labels):
    # Define a custom order for the labels
    order = ['DOS'] + symbols[0:3]
    # Removes duplicates labels
    seen = set()
    unique = []
    for h, l in zip(handles, labels):
        if l not in seen:
            unique.append((h, l))
            seen.add(l)
    # Sort labels
    unique = sorted(unique, key=lambda x: order.index(x[1]))
    sorted_handles, sorted_labels = zip(*unique)
    return list(sorted_handles), list(sorted_labels)

# Define a function that extracts the phonon dispersion data for plotting
def get_phonon_dispersion(phonon, bulk=True):
    """Function to extract phonon dispersion data for plotting.
    Parameters:
    - phonon: Phonopy object containing phonon data.
    - bulk: Boolean indicating if the system is bulk (True) or slab (False).
    Returns:
    - dist: Distances along the band path.
    - X: High symmetry point locations on the x-axis.
    - freq: Frequencies of the phonon modes.
    - labels: Labels for the high symmetry points.
    """
    # Specify band path and labels depending on bulk or slab
    if bulk == True:
        path = [[[0.0, 0.0, 0.0],[0.5, 0.0, 0.0],[0.5, 0.5, 0.5],
                [0.5, 0.5, 0.0],[0.0, 0.0, 0.0],[0.5, 0.5, 0.5]]]
        labels = ["$\\Gamma$", "X", "R", "M", "$\\Gamma$", "R"]
    else:
        path = [[[0.0, 0.0, 0.0],[0.5, 0.0, 0.0],
                 [0.5, 0.5, 0.0],[0.0, 0.0, 0.0]]]
        labels = ["$\\Gamma$", "X", "M", "$\\Gamma$"]
    # Get the band q-points and connections
    qpoints, connections = get_band_qpoints_and_path_connections(path, npoints=51)
    # Run band structure and total DOS calculation
    phonon.run_band_structure(qpoints, path_connections=connections,
                              labels=labels, with_eigenvectors=True)
    # Get the band structure data
    band_structure = phonon.get_band_structure_dict()
    # Extract frequencies and distances (x-values) for BS plot
    freq = band_structure['frequencies']
    dist = band_structure['distances']
    # Determine the high symmetry point locations on the x-axis
    X = np.append(np.array([arr[0] for arr in dist]), np.array(dist[-1][-1]))
    # Returns distances, symmetry point locations, frequencies and labels
    return (dist, X, freq, labels)

# Define a function that extracts the DOS data for plotting
def get_phonon_dos(phonon, bulk=True):
    """Function to extract phonon DOS data for plotting.
    Parameters:
    - phonon: Phonopy object containing phonon data.
    - bulk: Boolean indicating if the system is bulk (True) or slab (False).
    Returns:
    - dos: Density of states values.
    - freq: Frequency points.
    """
    # Load Phonopy object from YAML file
    if bulk == True:
        # Set mesh for bulk
        phonon.run_mesh([30, 30, 30])
    else:
        # Set mesh for slab
        phonon.run_mesh([30, 30, 1])
    # Run total DOS calculations and extract values
    phonon.run_total_dos()
    DOS = phonon.get_total_dos_dict()
    dos = DOS['total_dos']
    freq = DOS['frequency_points']
    # Returns DOS (states/THz) and frequencies (THz)
    return (dos, freq)

# Define a function that extracts the PDOS data for plotting
def get_phonon_pdos(phonon, bulk=True):
    """Function to extract phonon PDOS data for plotting.
    Parameters:
    - phonon: Phonopy object containing phonon data.
    - bulk: Boolean indicating if the system is bulk (True) or slab (False).
    Returns:
    - pdos: Projected density of states values.
    - freq: Frequency points.
    - symbols: List of atomic symbols in the unit cell.
    """
    # Load Phonopy object from YAML file
    if bulk == True:
        symbols = phonon.unitcell.symbols
        # Set symmetric mesh for bulk
        phonon.run_mesh([30, 30, 30], with_eigenvectors=True, is_mesh_symmetry=False)
    else:
        symbols = phonon.unitcell.symbols
        # Set asymmetric mesh for slab
        phonon.run_mesh([30, 30, 1], with_eigenvectors=True, is_mesh_symmetry=False)
    # Run PDOS calculations and extract values
    phonon.run_projected_dos()
    PDOS = phonon.get_projected_dos_dict()
    pdos = PDOS['projected_dos']
    freq = PDOS['frequency_points']
    # Returns PDOS (states/THz) and frequencies (THz)
    return (pdos, freq, symbols)

# Define a function that plots the dispersion and DOS together
def plot_dispersion(phonon, bulk=True):
    """Function to plot the phonon dispersion and DOS together.
    Parameters:
    - phonon: Phonopy object containing phonon data.
    - bulk: Boolean indicating if the system is bulk (True) or slab (False).
    Returns:
    - None. The function creates a plot of the phonon dispersion and DOS.
    """
    
    # Define tickmarks for the x- and y-axis
    ytickmarks = np.arange(-15, 26, 5)
    xtickmarks = np.arange(0, 7, 1)

    # Define colors
    colors = {'Ba': 'tab:blue', 'Sr': 'tab:purple',
              'Ti': 'tab:orange', 'O': 'tab:red'}
    
    # Make a simple figure where graphs are plotted
    fig = plt.figure(figsize=[6.6, 5])
    
    # Subplot 1 - Phonon dispersion
    ax1 = fig.add_axes([0, 0, 1, 1])
    # Set title
    #ax1.set_title('{} dispersion with {} using {}'.format(formula, xcf))
    # Extract phonon dispersion data
    (dist, X, freq, labels) = get_phonon_dispersion(phonon, bulk)
    # Plot vertical lines at symmetry points
    ax1.vlines(X, ytickmarks[0], ytickmarks[-1], color='0.5', lw=1)
    # Plot dashed line at 0
    ax1.axhline(y=0, color='k', linestyle=':')
    # Determine the number of segments between symmetry points and the number of modes
    n_segments = len(freq)
    n_modes = freq[0].shape[1]
    # Loop over all segments and modes and plot everything
    for i in range(n_segments):
        for j in range(n_modes):
            ax1.plot(dist[i], freq[i][:, j], color='k', lw=1.5)
    # Set x- and y-label
    ax1.set_xlabel('k-points')
    ax1.set_ylabel('Frequency, $\omega$ (THz)', fontsize=14)
    # Set x- and y-ticks
    ax1.set_xticks(X, labels)
    ax1.set_yticks(ytickmarks, ytickmarks)
    # Set x- and y-limits
    ax1.set_xlim(X[0], X[-1])
    ax1.set_ylim(ytickmarks[0], ytickmarks[-1])
    # Add minor tickmarks to the y-axis
    ax1.yaxis.set_minor_locator(AutoMinorLocator())
    # Initialize plot settings
    #PlotSettings().set_style_ax(ax)
    
    # Subplot 2 - Density of states (DOS)
    ax2 = fig.add_axes([1.05, 0, 0.4, 1])
    # Add title
    #ax2.set_title('Density of states')
    # Plot dashed line at Fermi level
    ax2.axhline(y=0, color='k', linestyle=':')
    # Extract total DOS data
    (dosx, dosy) = get_phonon_dos(phonon, bulk)
    # Plot total DOS
    ax2.plot(dosx, dosy, lw=1.5, color='k', label='DOS')
    ax2.fill_between(dosx, dosy, color='lightgray', alpha=0.5)
    # Extract PDOS data
    (pdosx, pdosy, symbols) = get_phonon_pdos(phonon, bulk)
    # Plot PDOS
    for i in range(pdosx.shape[0]):
        ax2.plot(pdosx[i], pdosy, lw=1.5,
                         color=colors[symbols[i]], label=f'{symbols[i]}')
    # Set x-label
    ax2.set_xlabel('DOS (states/THz)')
    # Get all handles and labels
    handles, labels = ax2.get_legend_handles_labels()
    # Remove duplicates and sort for the legend
    sorted_handles, sorted_labels = order_labels(symbols, handles, labels)
    # Add legend with duplicates removed and sorted labels
    ax2.legend(sorted_handles, sorted_labels, loc='best', fontsize=14)
    # Force x- and y-ticks
    ax2.set_xticks(xtickmarks, xtickmarks)
    ax2.set_yticks(ytickmarks, ytickmarks)
    # Set limits to match
    ax2.set_xlim(xtickmarks[0], xtickmarks[-1])
    ax2.set_ylim(ytickmarks[0], ytickmarks[-1])
    # Initialize plot settings
    #PlotSettings(ax2)
    # Hide y-tick labels
    ax2.set_yticklabels([])
    
    # Show figure
    plt.show()