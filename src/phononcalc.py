# Importing packages and modules
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
# ASE
from ase import Atoms
from ase.io import read
from ase.units import Ry
from ase.parallel import world
from ase.parallel import parprint
from ase.calculators.siesta import Siesta
# Phonopy
import phonopy as ph
from phonopy import Phonopy
from phonopy.phonon.band_structure import get_band_qpoints_and_path_connections
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.interface.calculator import read_crystal_structure
# Custom modules
from src.cleanfiles import cleanFiles
from src.plotsettings import PlotSettings
PlotSettings().set_global_style()

def calculate_phonons(atoms, xcf='PBEsol', basis='DZP', EnergyShift=0.01, SplitNorm=0.15,
                      MeshCutoff=200, kgrid=(10, 10, 10), pseudo=1,
                      mode='lcao', dir='results/bulk/phonons', par=False):
    """Function to calculate phonon properties of a bulk structure using Phonopy and SIESTA.
    Parameters:
    - atoms: ASE Atoms object representing the relaxed bulk structure.
    - xcf: Exchange-correlation functional to be used (default is 'PBEsol').
    - basis: Basis set to be used (default is 'DZP').
    - EnergyShift: Energy shift in Ry (default is 0.01 Ry).
    - SplitNorm: Split norm for basis functions (default is 0.15).
    - MeshCutoff: Mesh cutoff in Ry (default is 200 Ry).
    - kgrid: K-point mesh as a tuple (default is (10, 10, 10)).
    - pseudo: Integer index for selecting pseudopotential (default is 1).
    - mode: Calculator mode to be used ('lcao' for SIESTA or 'pw' for GPAW, default is 'lcao').
    - dir: Directory to save the results (default is 'results/bulk/phonons').
    - par: Whether the SIESTA calculator is parallel (default is False).
    Returns:
    - None. The function performs phonon calculations and saves the phonon data to a .yaml file.
    """
    # Parameters for phonon calculations
    N = 2  # Supercell size in each direction
    scell_matrix = np.diag([N, N, N])  # Supercell size
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
    symbols = atoms.symbols
    # In SIESTA, calculations are performed with localized atomic orbitals (LCAO)
    if mode == 'lcao':
        # Calculation parameters
        calc_params = {
            'label': f'{symbols}',
            'xc': xcf,
            'basis_set': basis,
            'mesh_cutoff': MeshCutoff * Ry,
            'energy_shift': EnergyShift * Ry,
            'kpts': tuple(x // N for x in kgrid),  # Reduce k-point grid for supercell calculations
            'directory': dir,
            'pseudo_path': os.path.join(cwd, 'pseudos', f'{pseudo}')
        }
        # fdf arguments
        fdf_args = {
            'PAO.BasisSize': basis,
            'PAO.SplitNorm': SplitNorm,
            'SCF.DM.Tolerance': 1e-6,
            "MD.TypeOfRun": "CG",
            "MD.NumCGsteps": 0,  # forces only
        }

        if par:
            # Change diagonalization algorithm when running in parallel
            fdf_args['Diag.Algorithm'] = 'ELPA'
    
    elif mode == 'pw':
        from gpaw import GPAW
        calc_params = {
            'xc': xcf,
            'basis': basis.lower(),
            'mode': {'name': 'pw', 'ecut': MeshCutoff * Ry},
            'kpts': {'size': tuple(x // N for x in kgrid), 'gamma': True},
            'occupations': {'name': 'fermi-dirac','width': 0.05},
            'convergence': {'density': 1e-6, 'forces': 1e-5},
            'txt': os.path.join(dir, f"{symbols}_PH.txt")
        }

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
        
        # Set up the calculator based on the selected mode
        if mode == 'lcao':
            # Set up the Siesta calculator
            calc = Siesta(**calc_params, fdf_arguments=fdf_args)
        elif mode == 'pw':
            # Set up the GPAW calculator
            calc = GPAW(**calc_params)

        # Attach the calculator to this ASE atoms object
        atoms_ase.calc = calc
        
        # Calculate forces on the displaced supercell
        force = atoms_ase.get_forces()
        
        # Append forces to the list
        forces.append(force)

    # Set forces in Phonopy and calculate force constants
    phonon.forces = forces
    if world.rank == 0:
        # Save phonopy .yaml file
        phonon.save(os.path.join(dir, f"{symbols}.yaml"))
    if mode == 'lcao':
        # Remove unnecessary files generated from SIESTA
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
def plot_dispersion(formula, ids=np.array([]), vals=np.array([]), root='results', pDOS=True, bulk=True):
    """Function to plot the phonon dispersion and DOS together.
    Parameters:
    - formula: Chemical formula of the material.
    - ids: Numpy array of IDs to plot.
    - vals: Numpy array of values corresponding to the IDs (e.g., different functionals or parameters).
    - root: Root directory for results.
    - mode: Calculation mode ('lcao' or 'pw').
    - pDOS: Whether to plot the projected density of states (PDOS) (default is True).
    - bulk: Boolean indicating if the system is bulk (True) or slab (False).
    Returns:
    - None. The function creates a plot of the phonon dispersion and DOS.
    """
    
    # Define tickmarks for the x- and y-axis
    ytickmarks = np.arange(-15, 26, 5)
    xtickmarks = np.arange(0, 7, 1)

    # Define colors and styles for plotting (if needed)
    colors = ["black", "blue", "red", "purple", "orange", "green"]
    #styles = ['-', '--', '-.', ':', '-', '--', '-.']

    # Make a simple figure where graphs are plotted
    fig = plt.figure(figsize=[6.6, 5])
    
    # Define two axes, one for the band structure and one for the DOS
    ax1 = fig.add_axes([0, 0, 1, 1])
    ax2 = fig.add_axes([1.05, 0, 0.4, 1])
    
    def _plot_disp(ax, phonon, val, col='k'):
        # Extract phonon dispersion data
        (dist, X, freq, labels) = get_phonon_dispersion(phonon, bulk)
        dist = np.array(dist)
        dist /= dist[-1][-1]  # Normalize distances to the total length of the path
        X /= X[-1]  # Normalize high symmetry point locations to the total length of the path
        # Plot vertical lines at symmetry points
        ax.vlines(X, ytickmarks[0], ytickmarks[-1], color='0.5', lw=1)
        # Plot dashed line at 0
        ax.axhline(y=0, color='k', linestyle=':')
        # Determine the number of segments between symmetry points and the number of modes
        n_segments = len(freq)
        n_modes = freq[0].shape[1]
        # Loop over all segments and modes and plot everything
        for i in range(n_segments):
            for j in range(n_modes):
                if i == 0 and j == 0:
                    ax.plot(dist[i], freq[i][:, j], color=col, lw=1.5, label=f'{val}')
                else:
                    ax.plot(dist[i], freq[i][:, j], color=col, lw=1.5)
        # Set x- and y-ticks
        ax.set_xticks(X, labels)
        ax.set_yticks(ytickmarks, ytickmarks)
        # Set x- and y-limits
        ax.set_xlim(X[0], X[-1])
        ax.set_ylim(ytickmarks[0], ytickmarks[-1])

    def _plot_dos(ax, phonon, val, col='k', style='-'):
        # Extract total DOS data
        (dosx, dosy) = get_phonon_dos(phonon, bulk)
        # Plot total DOS
        ax.plot(dosx, dosy, lw=1.5, color=col, label=f'{val}', linestyle=style)
        if pDOS:
            ax.fill_between(dosx, dosy, color='lightgray', alpha=0.5)

    def _plot_pdos(ax, phonon):
        atom_colors = {'Ba': 'tab:blue', 'Sr': 'tab:purple',
                       'Ti': 'tab:orange', 'O': 'tab:red'}
        # Extract PDOS data
        (pdosx, pdosy, symbols) = get_phonon_pdos(phonon, bulk)
        # Plot PDOS
        for i in range(pdosx.shape[0]):
            ax.plot(pdosx[i], pdosy, lw=1.5, color=atom_colors[symbols[i]], label=f'{symbols[i]}')
        # Get all handles and labels
        handles, labels = ax.get_legend_handles_labels()
        # Remove duplicates and sort for the legend
        sorted_handles, sorted_labels = order_labels(symbols, handles, labels)
        # Add legend with duplicates removed and sorted labels
        ax.legend(sorted_handles, sorted_labels, loc='best', fontsize=14)

    # Plot dashed line at Fermi level for both subplots
    ax1.axhline(y=0, color='k', linestyle=':')
    ax2.axhline(y=0, color='k', linestyle=':')

    
    dir = 'results/bulk/GPAW'
    phonon = ph.load(os.path.join(dir, f'{formula}.yaml'))
    # Plot phonon dispersion
    _plot_disp(ax1, phonon, 'GPAW', col=colors[0])
    # Plot total DOS
    _plot_dos(ax2, phonon, 'GPAW', col=colors[0])
    if pDOS:
        # Plot PDOS
        _plot_pdos(ax2, phonon)
    
    for i in range(len(ids)):
        # Load Phonopy object from YAML file
        dir = os.path.join(root, 'bulk/',formula, ids[i], 'phonons')
        phonon = ph.load(os.path.join(dir, f'{formula}.yaml'))
        
        # Plot phonon dispersion
        _plot_disp(ax1, phonon, vals[i], col=colors[i+1])
        # Plot total DOS
        _plot_dos(ax2, phonon, vals[i], col=colors[i+1])
        if pDOS:
            # Plot PDOS
            _plot_pdos(ax2, phonon)

    # Set x- and y-label
    ax1.set_xlabel('k-points')
    ax1.set_ylabel('Frequency, $\omega$ (THz)', fontsize=14)
    # Add minor tickmarks to the y-axis
    ax1.yaxis.set_minor_locator(AutoMinorLocator())
    

    ax2.set_xlabel('DOS (states/THz)')
    ax2.legend(loc='upper right')
    # Force x- and y-ticks
    ax2.set_xticks(xtickmarks, xtickmarks)
    ax2.set_yticks(ytickmarks, ytickmarks)
    # Set limits to match
    ax2.set_xlim(xtickmarks[0], xtickmarks[-1])
    ax2.set_ylim(ytickmarks[0], ytickmarks[-1])
    # Hide y-tick labels
    ax2.set_yticklabels([])
    
    # Show figure
    plt.show()


# Define a function that plots the dispersion and DOS together
def plot_dispersion2(formula, ids=np.array([]), vals=np.array([])):
    """Function to plot the phonon dispersion seperately
    Parameters:
    - formula: Chemical formula of the material.
    - ids: Numpy array of IDs to plot.
    - vals: Numpy array of values corresponding to the IDs (e.g., different functionals or parameters).
    Returns:
    - None. The function creates a plot of the phonon dispersion and DOS.
    """
    
    # Define tickmarks for the x- and y-axis
    E_tickmarks = np.arange(-10, 26, 5)

    # Define a list of colors for the plots (if needed)
    colors = ["black", "blue", "red", "purple", "orange", "green"]
    N = len(ids) + 1

    # Create N subplots for the band structure along x
    fig, axes = plt.subplots(1, N, figsize=(2.5*N, 5), sharey='col')

    plt.subplots_adjust(wspace=0.05)
    
    def _plot_disp(ax, phonon, val, col='k'):
        # Extract phonon dispersion data
        (dist, X, freq, labels) = get_phonon_dispersion(phonon)
        dist = np.array(dist)
        dist /= dist[-1][-1]  # Normalize distances to the total length of the path
        X /= X[-1]  # Normalize high symmetry point locations to the total length of the path

        # Set title
        ax.set_title(f"{val}")
        # Plot vertical lines at symmetry points
        ax.vlines(X, E_tickmarks[0], E_tickmarks[-1], color='0.5', lw=1)
        # Plot dashed line at 0
        ax.axhline(y=0, color='k', linestyle=':')
        # Determine the number of segments between symmetry points and the number of modes
        n_segments = len(freq)
        n_modes = freq[0].shape[1]
        # Loop over all segments and modes and plot everything
        for i in range(n_segments):
            for j in range(n_modes):
                ax.plot(dist[i], freq[i][:, j], color=col, lw=1.5)
        # Set x- and y-ticks
        ax.xaxis.set_ticks(X[0:-2])
        ax.set_xticklabels(labels[0:-2])
        ax.set_yticks(E_tickmarks, E_tickmarks.astype(int))
        # Set x- and y-limits
        ax.set_xlim(X[0], X[-2])
        ax.set_ylim(E_tickmarks[0], E_tickmarks[-1])
        # Add minor tickmarks to the y-axis
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        # Apply custom plot settings to the axes
        PlotSettings().set_style_ax(ax, style='default', minor=False)

    
    dir = 'results/bulk/GPAW'
    phonon = ph.load(os.path.join(dir, f'{formula}.yaml'))
    # Plot phonon dispersion
    _plot_disp(axes[0], phonon, 'pw', col=colors[0])
    axes[0].set_ylabel('DOS (states/THz)')
    
    # Cycle through the list of IDs and plot the dispersion
    for i in range(len(ids)):
        # Load Phonopy object from YAML file
        dir = os.path.join('results/bulk/',formula, ids[i], 'phonons')
        phonon = ph.load(os.path.join(dir, f'{formula}.yaml'))
        # Plot phonon dispersion
        _plot_disp(axes[i+1], phonon, vals[i], col=colors[i+1])
        # Remove y-tick labels for all but the first and last subplot
        if i < len(ids) - 1:
            axes[i+1].set_yticklabels([])
    
    # Move y-axis of the last subplot to the right but maintain the y-tickmarks on the left
    axes[-1].tick_params(axis='y', labelright=True, labelleft=False)
    # Show figure
    plt.show()