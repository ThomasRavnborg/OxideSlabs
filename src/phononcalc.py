# Importing packages and modules
import os
import time
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
from ase.calculators.siesta.parameters import Species, PAOBasisBlock
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

def ase_to_phonopy(atoms_ase):
    """Function to convert ASE Atoms object to PhonopyAtoms object.
    Parameters:
    - atoms_ase: ASE Atoms object representing the structure.
    Returns:
    - PhonopyAtoms object with the same structure as the input ASE Atoms.
    """
    return PhonopyAtoms(symbols=atoms_ase.get_chemical_symbols(),
                        positions=atoms_ase.get_positions(),
                        cell=atoms_ase.get_cell(),
                        masses=atoms_ase.get_masses())

def phonopy_to_ase(atoms_phonopy):
    """Function to convert PhonopyAtoms object to ASE Atoms object.
    Parameters:
    - atoms_phonopy: PhonopyAtoms object representing the structure.
    Returns:
    - ASE Atoms object with the same structure as the input PhonopyAtoms.
    """
    return Atoms(symbols=atoms_phonopy.symbols,
                 positions=atoms_phonopy.positions,
                 cell=atoms_phonopy.cell)

def calculate_phonons(perovskite, xcf='PBEsol', basis='DZP',
                      EnergyShift=0.01, SplitNorm=0.15,
                      MeshCutoff=1000, kgrid=(10, 10, 10),
                      pseudo='PBEsol', mode='lcao',
                      dir='results/bulk/phonons', par=True):
    """Function to calculate phonon properties of a structure using Phonopy and SIESTA.
    Parameters:
    - perovskite: Custom object representing the relaxed structure.
    - xcf: Exchange-correlation functional to be used (default is 'PBEsol').
    - basis: Basis set to use for the calculation (default: 'DZP').
             If basis ends with (lower-case) p, a polarization orbital will be added to the A-site (Ba)
    - EnergyShift: Energy shift in Ry (default is 0.01 Ry).
    - SplitNorm: Split norm for basis functions (default is 0.15).
    - MeshCutoff: Mesh cutoff in Ry (default is 1000 Ry).
    - kgrid: K-point mesh as a tuple (default is (10, 10, 10)).
    - pseudo: Pseudopotential to be used (default is 'PBEsol').
    - mode: Calculator mode to be used ('lcao' for SIESTA or 'pw' for GPAW, default is 'lcao').
    - dir: Directory to save the results (default is 'results/bulk/phonons').
    - par: Whether the SIESTA calculator is parallel (default is True).
    Returns:
    - None. The function performs phonon calculations and saves the phonon data to a .yaml file.
    """
    # Define current working directory and extract information from the perovskite object
    cwd = os.getcwd()
    formula = perovskite.formula
    #symbols = perovskite.symbols
    atoms = perovskite.atoms
    bulk = perovskite.bulk
    # Convert kgrid to a list to allow for modification
    kgrid = list(kgrid)

    # Custom basis sets ending with 'p' are generated with the same parameters as the standard basis sets
    # However, an extra polarization (d) orbital is added to the A-site during LCAO basis generation
    if basis.endswith('p'):
        basis = basis[:-1]


    # Parameters for phonon calculations
    dd = 0.01 # Displacement distance in Å
    Nsc = 2
    if bulk:
        scell_matrix = np.diag([Nsc, Nsc, Nsc])  # Supercell size for bulk
        kgrid = [x // Nsc for x in kgrid]    # Reduce k-point grid for supercell calculations
    else:
        scell_matrix = np.diag([Nsc, Nsc, 1])  # Supercell size for slab
        kgrid = [x // Nsc for x in kgrid]    # Reduce k-point grid for supercell calculations
        # For slab calculations, set k-point sampling to 1 in the z-direction
        kgrid[2] = 1

    # Convert ASE Atoms to PhonopyAtoms
    unitcell = ase_to_phonopy(atoms)
    
    # Create Phonopy object and generate displacements
    phonon = Phonopy(unitcell, scell_matrix)
    phonon.generate_displacements(distance=dd)
    supercells = phonon.supercells_with_displacements
    parprint(f"Generated {len(supercells)} supercells with displacements.")
    
    # In SIESTA, calculations are performed with localized atomic orbitals (LCAO)
    if mode == 'lcao':
        # Calculation parameters
        calc_params = {
            'label': f'{formula}',
            'xc': xcf,
            'basis_set': basis,
            'mesh_cutoff': MeshCutoff * Ry,
            'energy_shift': EnergyShift * Ry,
            'kpts': kgrid,
            'directory': dir,
            'pseudo_path': os.path.join(cwd, 'pseudos', f'{xcf}')
        }
        dir_fdf = os.path.join(cwd, os.path.dirname(dir))
        # fdf arguments in a dictionary
        fdf_args = {
            '%include': os.path.join(dir_fdf, 'basis.fdf'),
            'PAO.SplitNorm': SplitNorm,
            'SCF.DM.Tolerance': 1e-8,
            "MD.TypeOfRun": "CG",
            "MD.NumCGsteps": 0,  # forces only
        }
        if par:
            # Change diagonalization algorithm when running in parallel
            fdf_args['Diag.Algorithm'] = 'ELPA'
        if not bulk:
            # Add dipole correction for slab calculations to avoid spurious interactions between periodic images
            fdf_args['Slab.DipoleCorrection'] = 'T'
    
    elif mode == 'pw':
        from gpaw import GPAW
        calc_params = {
            'xc': xcf,
            'basis': basis.lower(),
            'mode': {'name': 'pw', 'ecut': MeshCutoff * Ry},
            'kpts': {'size': kgrid, 'gamma': True},
            'occupations': {'name': 'fermi-dirac','width': 0.05},
            'convergence': {'density': 1e-6, 'forces': 1e-5},
            'txt': os.path.join(dir, f"{formula}.txt")
        }
        if not bulk:
            # Add dipole correction for slab calculations to avoid spurious interactions between periodic images
            calc_params["poissonsolver"] = {"dipolelayer": "xy"}

    if world.rank == 0:
        forces = []
        t0 = time.time() # Start timer
    # Loop over all supercells and calculate forces
    for i, sc in enumerate(supercells):
        # Print which supercell is being processed
        parprint(f"Processing supercell {i + 1}/{len(supercells)}")
        
        # Convert PhonopyAtoms to ASE Atoms for each supercell
        atoms_ase = phonopy_to_ase(sc)
        
        if not bulk:
            # Remove periodicity in the z-direction for slab calculations
            atoms_ase.pbc = (True, True, False)

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
        
        if world.rank == 0:
            # Append forces to the list
            forces.append(force)
    
    if world.rank == 0:
        # Set forces in Phonopy and calculate force constants
        phonon.forces = forces
        t1 = time.time() # Stop timer
        # Save phonopy .yaml file
        phonon.save(os.path.join(dir, f"{formula}.yaml"))
        # Write the time taken for phonon calculations to a file
        np.save(os.path.join(dir, f"time.npy"), t1-t0)
    if mode == 'lcao':
        # Remove unnecessary files generated from SIESTA
        cleanFiles(directory=dir, confirm=False)
    # Wait for all parallel processes to finish
    world.barrier()

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
                [0.5, 0.5, 0.0],[0.0, 0.0, 0.0]]]
        labels = ["$\\Gamma$", "X", "R", "M", "$\\Gamma$"]
    else:
        path = [[[0.0, 0.0, 0.0],[0.5, 0.0, 0.0],
                 [0.5, 0.5, 0.0],[0.0, 0.0, 0.0]]]
        labels = ["$\\Gamma$", "X", "M", "$\\Gamma$"]
    # Get the band q-points and connections
    qpoints, connections = get_band_qpoints_and_path_connections(path, npoints=300)
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
def plot_dispersion(formula, ids=np.array([]), vals=np.array([]), bulk=True, Ncells=1, pDOS=False):
    """Function to plot the phonon dispersion and DOS together.
    Parameters:
    - phonon: Phonopy object containing phonon data.
    - pDOS: Whether to plot the projected density of states (PDOS) (default is True).
    - bulk: Boolean indicating if the system is bulk (True) or slab (False).
    Returns:
    - None. The function creates a plot of the phonon dispersion and DOS.
    """

    #atoms = phonon_to_atoms(phonon, cell='unit')
    #formula = atoms.symbols

    if bulk:
        struc = f'bulk/{formula}'
    else:
        struc = f'slab/{formula}/{Ncells}uc'


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

    def _plot_dos(ax, phonon, val, col='k'):
        # Extract total DOS data
        (dosx, dosy) = get_phonon_dos(phonon, bulk)
        # Plot total DOS
        ax.plot(dosx, dosy, lw=1.5, color=col, label=f'{val}')
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

    
    dir = f'results/{struc}/GPAW/phonons'
    phonon = ph.load(os.path.join(dir, f'{formula}.yaml'))
    # Plot phonon dispersion and total DOS for PW
    _plot_disp(ax1, phonon, 'PW', col='k')
    _plot_dos(ax2, phonon, 'PW', col='k')
    if pDOS:
        # Plot PDOS for PW
        _plot_pdos(ax2, phonon)

    for i in range(len(ids)):
        # Load Phonopy object from YAML file
        dir = os.path.join('results', struc, ids[i], 'phonons')
        phonon = ph.load(os.path.join(dir, f'{formula}.yaml'))
        # Plot phonon dispersion and total DOS for PW
        _plot_disp(ax1, phonon, vals[i], col=colors[i+1])
        _plot_dos(ax2, phonon, vals[i], col=colors[i+1])
    
    # Set x- and y-label
    ax1.set_xlabel('k-points')
    ax1.set_ylabel('Frequency, $\omega$ (THz)')
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
def plot_dispersion2(formula, ids=np.array([]), vals=np.array([]), bulk=True, Ncells=1):
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
    # Convert tickmarks to strings with i for negative numbers
    E_ticklabels = []
    for tick in E_tickmarks:
        if tick < 0:
            E_ticklabels.append(f'{abs(tick)}i')
        else:
            E_ticklabels.append(f'{tick}')

    # Define a list of colors for the plots (if needed)
    colors = ["black", "blue", "red", "purple", "orange", "green"]
    N = len(ids) + 1

    if bulk:
        struc = f'bulk/{formula}'
    else:
        struc = f'slab/{formula}/{Ncells}uc'

    # Create N subplots for the band structure along x
    fig, axes = plt.subplots(1, N, figsize=(2.5*N, 5), sharey='col')

    plt.subplots_adjust(wspace=0.05)
    
    def _plot_disp(ax, phonon, val, col='k'):
        # Extract phonon dispersion data
        (dist, X, freq, labels) = get_phonon_dispersion(phonon, bulk)
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
        ax.xaxis.set_ticks(X[0:-1])
        ax.set_xticklabels(labels[0:-1])
        ax.set_yticks(E_tickmarks, E_ticklabels)
        # Set x- and y-limits
        ax.set_xlim(X[0], X[-1])
        ax.set_ylim(E_tickmarks[0], E_tickmarks[-1])
        # Add minor tickmarks to the y-axis
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        # Apply custom plot settings to the axes
        PlotSettings().set_style_ax(ax, style='default', minor=False)

    
    dir = f'results/{struc}/GPAW/phonons'
    phonon = ph.load(os.path.join(dir, f'{formula}.yaml'))
    # Plot phonon dispersion
    _plot_disp(axes[0], phonon, 'PW', col=colors[0])
    axes[0].set_ylabel('Frequency, $\omega$ (THz)')
    
    # Cycle through the list of IDs and plot the dispersion
    for i in range(len(ids)):
        # Load Phonopy object from YAML file
        dir = os.path.join('results', struc, ids[i], 'phonons')
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