# Importing packages and modules
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
# ASE
from ase.units import Ry
from ase.parallel import world
from ase.parallel import parprint
from ase.calculators.siesta import Siesta
# Phonopy
from phonopy import Phonopy
from phonopy.phonon.band_structure import get_band_qpoints_and_path_connections
# Custom modules
from src.cleanfiles import cleanFiles
from src.structure import is_atom_bulk
from src.phononASE import is_phonon_bulk
from src.phononASE import ase_to_phonopy, phonopy_to_ase
from src.latexfig import LatexFigure


def calculate_phonons(atoms, xcf='PBEsol', basis='DZP',
                      EnergyShift=0.01, SplitNorm=0.15,
                      MeshCutoff=1000, kgrid=(10, 10, 10),
                      mode='lcao', dir='results/bulk/phonons', par=True):
    """Function to calculate phonon properties of a structure using Phonopy and SIESTA.
    Parameters:
    - atoms: ASE Atoms object representing the relaxed structure.
    - xcf: Exchange-correlation functional to be used (default is 'PBEsol').
    - basis: Basis set to use for the calculation (default: 'DZP').
             If basis ends with (lower-case) p, a polarization orbital will be added to the A-site (Ba)
    - EnergyShift: Energy shift in Ry (default is 0.01 Ry).
    - SplitNorm: Split norm for basis functions (default is 0.15).
    - MeshCutoff: Mesh cutoff in Ry (default is 1000 Ry).
    - kgrid: K-point mesh as a tuple (default is (10, 10, 10)).
    - mode: Calculator mode to be used ('lcao' for SIESTA or 'pw' for GPAW, default is 'lcao').
    - dir: Directory to save the results (default is 'results/bulk/phonons').
    - par: Whether the SIESTA calculator is parallel (default is True).
    Returns:
    - None. The function performs phonon calculations and saves the phonon data to a .yaml file.
    """
    # Define current working directory and extract information from the perovskite object
    cwd = os.getcwd()
    formula = atoms.get_chemical_formula()
    bulk = is_atom_bulk(atoms)
    # Convert kgrid to a list to allow for modification
    kgrid = list(kgrid)

    # Custom basis sets ending with 'p' are generated with the same parameters as the standard basis sets
    # However, an extra polarization (d) orbital is added to the A-site during LCAO basis generation
    if basis.endswith('p') or basis.endswith('d'):
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
    parprint(f"Generated {len(supercells)} supercells with displacements.", flush=True)
    
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
        parprint(f"Processing supercell {i + 1}/{len(supercells)}", flush=True)
        
        # Convert PhonopyAtoms to ASE Atoms for each supercell
        atoms_ase = phonopy_to_ase(sc, bulk)
        
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
    """Function to order the labels in the legend of the PDOS plot.
    Parameters:
        - symbols (list): List of atomic symbols.
        - handles (list): List of plot handles.
        - labels (list): List of label strings.
    Returns:
        - sorted_handles (list): List of sorted plot handles.
        - sorted_labels (list): List of sorted label strings.
    """
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


def get_phonon_dispersion(phonon):
    """Function to extract phonon dispersion data for plotting.
    Parameters:
        - phonon: Phonopy object containing phonon data.
    Returns:
        - dist: Distances along the band path.
        - X: High symmetry point locations on the x-axis.
        - freq: Frequencies of the phonon modes.
        - labels: Labels for the high symmetry points.
    """
    bulk = is_phonon_bulk(phonon)
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


def get_phonon_dos(phonon):
    """Function to extract phonon DOS data for plotting.
    Parameters:
        - phonon: Phonopy object containing phonon data.
    Returns:
        - dos: Density of states values.
        - freq: Frequency points.
    """
    bulk = is_phonon_bulk(phonon)
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


def get_phonon_pdos(phonon):
    """Function to extract phonon PDOS data for plotting.
    
    Parameters:
        - phonon: Phonopy object containing phonon data.
    
    Returns:
        - pdos: Projected density of states values.
        - freq: Frequency points.
        - symbols: List of atomic symbols in the unit cell.
    """
    bulk = is_phonon_bulk(phonon)
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


def plot_dispersion(phonons, labels, width=1, multiple=False):
    """Function to plot the phonon dispersion and DOS together.
    
    Parameters:
        - phonons: Phonon object or list of Phonopy objects containing phonon data.
        - labels: Label or list of labels for each phonon object.
        - width: Width of the figure as a fraction of the total width.
        - multiple: If True, plot each phonon object in a separate subplot. If False, plot all phonon objects in the same subplot.
    
    Returns:
        - None. The function creates a plot of the phonon dispersion and DOS.
    """

    def _ensure_list(obj):
        if isinstance(obj, list):
            return obj
        else:
            return [obj]
        
    # Ensure that phonons and labels are lists
    phonons = _ensure_list(phonons)
    labels = _ensure_list(labels)
    N_bands = len(phonons)

    # Check that the number of labels matches the number of phonon objects
    if N_bands != len(labels):
        raise ValueError("Number of labels must match number of phonons.")

    # Define tickmarks for the x- and y-axis
    E_tickmarks = np.arange(-10, 26, 5)
    # Convert tickmarks to strings with i for negative numbers
    E_ticklabels = []
    for tick in E_tickmarks:
        if tick < 0:
            E_ticklabels.append(f'{abs(tick)}i')
        else:
            E_ticklabels.append(f'{tick}')

    # Define tickmarks for the x- and y-axis
    dos_tickmarks = np.arange(0, 7, 1)
    # Define colors for the different phonon objects (up to 6)
    colors = ["black", "blue", "red", "purple", "orange", "green"]

    # Make a simple figure where graphs are plotted
    lf = LatexFigure()
    if multiple:
        fig, axes = lf.create(width=width, AR=1.8, subplots=(1, N_bands), minor=False, sharey='col')
    else:
        fig, axes = lf.create(width=width, AR=1, subplots=(1, 2), style='bands', minor=False,
                              sharey='col', gridspec_kw={'width_ratios': [1, 0.4]})
    
    def _plot_disp(ax, phonon, label, col='k'):
        # Extract phonon dispersion data
        (dist, X, freq, labels) = get_phonon_dispersion(phonon)
        dist = np.array(dist)
        dist /= dist[-1][-1]  # Normalize distances to the total length of the path
        X /= X[-1]  # Normalize high symmetry point locations to the total length of the path
        # Plot vertical lines at symmetry points
        ax.vlines(X, E_tickmarks[0], E_tickmarks[-1], color='0.5', lw=0.8)
        # Determine the number of segments between symmetry points and the number of modes
        n_segments = len(freq)
        n_modes = freq[0].shape[1]
        # Loop over all segments and modes and plot everything
        for i in range(n_segments):
            for j in range(n_modes):
                if i == 0 and j == 0:
                    ax.plot(dist[i], freq[i][:, j], color=col, lw=1, label=f'{label}')
                else:
                    ax.plot(dist[i], freq[i][:, j], color=col, lw=1)
        # Set x- and y-ticks
        if multiple:
            ax.set_xticks(X[0:-1], labels[0:-1])
        else:
            ax.set_xticks(X, labels)
        ax.set_yticks(E_tickmarks, E_ticklabels)
        # Set x- and y-limits
        ax.set_xlim(X[0], X[-1])
        ax.set_ylim(E_tickmarks[0], E_tickmarks[-1])

    def _plot_dos(ax, phonon, label='DOS', col='k', pDOS=False):
        # Extract total DOS data
        (dosx, dosy) = get_phonon_dos(phonon)
        # Plot total DOS
        ax.plot(dosx, dosy, lw=1, color=col, label=f'{label}')
        if pDOS:
            ax.fill_between(dosx, dosy, color='lightgray', alpha=0.5)
        
        # Force x- and y-ticks
        ax.set_xticks(dos_tickmarks, dos_tickmarks.astype(str))
        ax.set_yticks(E_tickmarks, E_ticklabels)
        # Set limits to match
        ax.set_xlim(dos_tickmarks[0], dos_tickmarks[-1])
        ax.set_ylim(E_tickmarks[0], E_tickmarks[-1])
        # Hide y-tick labels
        ax.set_yticklabels([])

    def _plot_pdos(ax, phonon):
        atom_colors = {'Ba': 'tab:blue', 'Sr': 'tab:purple',
                       'Ti': 'tab:orange', 'O': 'tab:red'}
        # Extract PDOS data
        (pdosx, pdosy, symbols) = get_phonon_pdos(phonon)
        # Plot PDOS
        for i in range(pdosx.shape[0]):
            ax.plot(pdosx[i], pdosy, lw=1, color=atom_colors[symbols[i]], label=f'{symbols[i]}')
        # Get all handles and labels
        handles, labels = ax.get_legend_handles_labels()
        # Remove duplicates and sort for the legend
        sorted_handles, sorted_labels = order_labels(symbols, handles, labels)
        # Add legend with duplicates removed and sorted labels
        ax.legend(sorted_handles, sorted_labels, loc='best', fontsize=14)

    if multiple:
        # Plot dispersion for each phonon object in a separate subplot
        axes[0].set_ylabel('Frequency (THz)')
        # Cycle through the 
        for i in range(N_bands):
            # Plot dashed line at Fermi level for all subplots
            axes[i].axhline(y=0, color='k', linestyle=':', lw=0.8)
            # Plot phonon dispersion for all subplots
            _plot_disp(axes[i], phonons[i], labels[i], col=colors[i])
            # Add title
            axes[i].set_title(labels[i])
            # Add minor tickmarks to the y-axis
            axes[i].yaxis.set_minor_locator(AutoMinorLocator())
            # Remove y-tick labels for all but the first and last subplot
            if i < N_bands-1 and i > 0:
                axes[i].set_yticklabels([])
        
        # Move y-axis of the last subplot to the right but maintain the y-tickmarks on the left
        axes[-1].tick_params(axis='y', labelright=True, labelleft=False)
        # Remove vertical spacing between subplots
        fig.set_constrained_layout_pads(wspace=0.0, w_pad=0.0)

    else:
        # Plot all phonon objects in the same subplot and plot the DOS in the second subplot
        
        # Define two axes, one for the band structure and one for the DOS
        ax1, ax2 = axes

        # Plot dashed line at Fermi level for both subplots
        ax1.axhline(y=0, color='k', linestyle=':', lw=0.8)
        ax2.axhline(y=0, color='k', linestyle=':', lw=0.8)
        for i in range(N_bands):
            # If a single phonon object is provided, also plot pDOS along with DOS
            if len(phonons) == 1:
                # Plot phonon dispersion, DOS and pDOS
                _plot_disp(ax1, phonons[i], label=labels[i])
                _plot_dos(ax2, phonons[i], pDOS=True)
                _plot_pdos(ax2, phonons[i])
            # If multiple phonon objects are provided, only plot the total DOS for each object
            else:
                # Plot phonon dispersion and DOS
                _plot_disp(ax1, phonons[i], label=labels[i], col=colors[i])
                _plot_dos(ax2, phonons[i], label=labels[i], col=colors[i])

        # Set x- and y-label
        ax1.set_xlabel('k-points')
        ax1.set_ylabel('Frequency (THz)')
        # Add minor tickmarks to the y-axis
        ax1.yaxis.set_minor_locator(AutoMinorLocator())

        # Set x-label
        ax2.set_xlabel('DOS (1/THz)')
        # Add legend to the DOS plot
        ax2.legend(loc='upper right')
        # Add minor tickmarks to the y-axis
        ax2.yaxis.set_minor_locator(AutoMinorLocator())

    
    # Show figure
    plt.show()