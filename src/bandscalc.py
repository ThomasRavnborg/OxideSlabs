# Importing packages and modules
import os
import time
import numpy as np
import sisl as si
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
# ASE
from ase.io import read
from ase.calculators.siesta import Siesta
from ase.calculators.siesta.parameters import Species, PAOBasisBlock
from ase import Atoms
from ase.units import Ry
from ase.dft.kpoints import bandpath
from ase.parallel import parprint
# Custom modules
from src.cleanfiles import cleanFiles
from src.plotsettings import PlotSettings
PlotSettings().set_global_style()

# Try to import world from gpaw.mpi for parallel processing
# If not available, fall back to ase.parallel.world
try:
    from gpaw.mpi import world
except ImportError:
    from ase.parallel import world

def calculate_bands(perovskite, xcf='PBEsol', basis='DZP',
                    EnergyShift=0.01, SplitNorm=0.15,
                    MeshCutoff=1000, kgrid=(10, 10, 10),
                    mode='lcao', dir='results/bulk/bandstructure', par=True):
    """Function to calculate band structure and PDOS of a bulk structure using SIESTA.
    Parameters:
    - perovskite: Custom object representing the relaxed bulk structure.
    - bulk: Boolean indicating whether the structure is bulk (True) or slab (False) (default is True).
    - xcf: Exchange-correlation functional to be used (default is 'PBEsol').
    - basis: Basis set to use for the calculation (default: 'DZP').
             If basis ends with (lower-case) p, a polarization orbital will be added to the A-site (Ba)
    - EnergyShift: Energy shift in Ry (default is 0.01 Ry).
    - SplitNorm: Split norm for basis functions (default is 0.15).
    - MeshCutoff: Mesh cutoff in Ry (default is 200 Ry).
    - kgrid: K-point mesh as a tuple (default is (10, 10, 10)).
    - mode: Calculator mode to be used ('lcao' for SIESTA or 'pw' for GPAW, default is 'lcao').
    - par: Whether the SIESTA calculator is parallel (default is True).
    Returns:
    - None. The function performs band structure calculation and saves the data to a file.
    """
    # Define current working directory and extract information from the perovskite object
    cwd = os.getcwd()
    formula = perovskite.formula
    #symbols = perovskite.symbols
    atoms = perovskite.atoms
    bulk = perovskite.bulk
    ncells = perovskite.ncells
    # Convert kgrid to a list to allow for modification
    kgrid = list(kgrid)

    # Custom basis sets ending with 'p' are generated with the same parameters as the standard basis sets
    # However, an extra polarization (d) orbital is added to the A-site during LCAO basis generation
    if basis.endswith('p'):
        basis = basis[:-1]

    if not bulk:
        # For slab calculations, set k-point sampling to 1 in the z-direction
        kgrid[2] = 1

    if mode == 'lcao':
        #parprint(f"Calculating band structure and PDOS for {formula} using SIESTA.")
        # Calculation parameters in a dictionary
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
            'SCF.DM.Tolerance': 1e-6,
            'BandLinesScale': 'ReciprocalLatticeVectors',
            '%block BandLines': '''
            1 0.000 0.000 0.000 \Gamma
            60 0.500 0.000 0.000 X
            60 0.500 0.500 0.500 R
            60 0.500 0.500 0.000 M
            60 0.000 0.000 0.000 \Gamma
            %endblock BandLines''',
            '%block PDOS.kgrid_Monkhorst_Pack': '''
            40  0  0  0.0
            0  40  0  0.0
            0   0 40  0.0
            %endblock PDOS.kgrid_Monkhorst_Pack''',
            '%block Projected.DensityOfStates': '''
            -20.00 15.00 0.200 500 eV
            %endblock Projected.DensityOfStates'''
        }

        if par:
            # Change diagonalization algorithm when running in parallel
            fdf_args['Diag.Algorithm'] = 'ELPA'
        
        if not bulk:
            # Change band path and k-point sampling for slab calculations
            fdf_args['%block BandLines'] = '''
            1 0.000 0.000 0.000 \Gamma
            60 0.500 0.000 0.000 X
            60 0.500 0.500 0.000 M
            60 0.000 0.000 0.000 \Gamma
            %endblock BandLines'''
            fdf_args['%block PDOS.kgrid_Monkhorst_Pack'] = '''
            40  0  0  0.0
            0  40  0  0.0
            0   0  1  0.0
            %endblock PDOS.kgrid_Monkhorst_Pack'''
            # Add dipole correction for slab calculations to avoid spurious interactions between periodic images
            fdf_args['Slab.DipoleCorrection'] = 'T'
        
        # Set up the Siesta calculator
        calc = Siesta(**calc_params, fdf_arguments=fdf_args)

    elif mode == 'pw':
        from gpaw import GPAW
        calc_params = {
            'xc': xcf,
            'basis': basis.lower(),
            'mode': {'name': 'pw', 'ecut': MeshCutoff * Ry},
            'kpts': {'size': kgrid, 'gamma': True},
            'occupations': {'name': 'fermi-dirac','width': 0.05},
            'convergence': {'density': 1e-6, 'forces': 1e-5},
            'txt': None
        }
        if not bulk:
            # Add dipole correction for slab calculations to avoid spurious interactions between periodic images
            calc_params["poissonsolver"] = {"dipolelayer": "xy"}
        # Set up the GPAW calculator
        calc = GPAW(**calc_params)
    
    # Attach the calculator to the atoms object
    atoms.calc = calc
    if world.rank == 0:
        t0 = time.time() # Start timer
    # Run band structure and DOS calculation
    atoms.get_potential_energy()

    if mode == 'pw':
        # Calculate bandstructure along a path
        if bulk:
            path = 'GXRMG'
            npoints = 240
        else:
            path = 'GXMG'
            npoints = 180
        BScalc = atoms.calc.fixed_density(
            nbands=int(32*ncells),
            symmetry='off',
            kpts={'path': path, 'npoints': npoints},
            convergence={'bands': 16},
            txt=os.path.join(dir, f"{formula}.txt"))
        
        path = bandpath(path, BScalc.atoms.cell, npoints=npoints)
        
        x, X, labels = path.get_linear_kpoint_axis()
        X = np.array([X, labels])
    
        # Find Fermi energy
        ef = BScalc.get_fermi_level()
        # These arrays give us the datapoints and off-sets to the fermi energy
        e_kn = np.array([BScalc.get_eigenvalues(kpt=k) for k in range(len(BScalc.get_ibz_k_points()))])
        e_nk = e_kn.T
        e_nk -= ef
        # The density of states (DOS) is calculated
        E, DOS = calc.get_dos(spin=0, npts=2000, width=0.2)
        # Shift energy values to fermi level
        E -= ef
        # Save the bandstructure and DOS data to files on the master process
        if world.rank == 0:
            t1 = time.time() # Stop timer
            # Save the bandstructure data to a file
            np.savez(os.path.join(dir, f"{formula}_BS.npz"), X=X, x=x, bands=e_nk)
            # Save the DOS data to a file
            np.savez(os.path.join(dir, f"{formula}_DOS.npz"), E=E, DOS=DOS)
            # Write the time taken for optimization to a file
            np.save(os.path.join(dir, f"time.npy"), t1-t0)
        # Wait for all parallel processes to finish
        world.barrier()
        
    elif mode == 'lcao':
        # Stop timer
        t1 = time.time() # Stop timer
        # Write the time taken for band structure calculations to a file
        np.save(os.path.join(dir, f"time.npy"), t1-t0)
        # Remove unnecessary files generated from SIESTA
        cleanFiles(directory=dir, confirm=False)

# Make a function that converts the texts 'Gamma' and 'G' into the actual symbols in the x-tick labels of the band structure plot
def convert_labels(labels):
    """Convert labels such as 'Gamma' and 'G' into LaTeX symbols for the x-tick labels of the band structure plot.
    Parameters:
    - labels: Numpy array of labels to be converted.
    Returns:
    - converted: Numpy array of converted labels with 'Gamma' and 'G' replaced by LaTeX symbols."""
    converted = []
    for label in labels:
        if label == 'Gamma' or label == 'G':
            converted.append(r'$\Gamma$')
        else:
            converted.append(label)
    return np.array(converted)

# Define a function that plots the bandstructure and DOS together
def plot_bands(formula, ids=np.array([]), vals=np.array([]), bulk=True, Ncells=1, width=1):
    """Function to plot bandstructure and DOS for a given formula.
    Requires that the bandstructure and PDOS have already been calculated and saved to files.
    Parameters:
    - formula: Chemical formula of the material.
    - ids: Numpy array of IDs to plot.
    - vals: Numpy array corresponding to the IDs (e.g., different functionals or parameters).
    - bulk: Boolean indicating whether the structure is bulk (True) or slab (False) (default is True).
    - Ncells: Number of unit cells
    - width: Fraction of the target width for the figure (default is 1).
    Returns:
    - None. The function reads the bandstructure and PDOS data from files and plots the results.
    """

    # Define tickmarks for the x- and y-axis
    E_tickmarks = np.arange(-6, 8, 2)
    dos_tickmarks = np.arange(0, 7, 1)

    # Define a list of colors for the plots (if needed)
    colors = ["black", "blue", "red", "purple", "orange", "green"]
    
    if bulk:
        struc = f'bulk/{formula}'
    else:
        struc = f'slab/{formula}/{Ncells}uc'

    # Create 2 subplots for the band structure and DOS, with shared y-axis
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 5),
                             sharey='col', gridspec_kw={'width_ratios': [1, 0.4]})
    ax1, ax2 = axes
    # Adjust spacing between subplots and set figure size using the custom PlotSettings class
    plt.subplots_adjust(wspace=0.08)
    #PlotSettings().set_size(fig)

    def _plot_bandstructure(ax, dir, val, col='k', mode='lcao'):
        if mode == 'lcao':
            # Read the bandstructure data from the file generated by Siesta using SISL
            sile_bands = si.get_sile(os.path.join(dir, f"{formula}.bands"))
            (X, labels), x, bands = sile_bands.read_data()
            bands = np.squeeze(bands).T

        elif mode == 'pw':
            # Read the bandstructure data from the file generated by GPAW
            data = np.load(os.path.join(dir, f"{formula}_BS.npz"), allow_pickle=True)
            (X, labels) = data['X']
            x = data['x']
            bands = data['bands']
        
        # Convert labels to use LaTeX symbols for Gamma point
        labels = convert_labels(labels)
        # Convert X, x and bands to numpy arrays of type float (if they are not already)
        X = np.array(X, dtype=float)
        labels = np.array(labels)
        x = np.array(x, dtype=float)
        bands = np.array(bands, dtype=float)
        # Shorten path to end with G
        indx = np.where(labels == r'$\Gamma$')[0][-1]
        labels = labels[:indx+1]
        X = X[:indx+1]
        # Normalize x and X to the total length of the path
        x /= X[-1]
        X /= X[-1]
        # Find lowest energy (VBM) and highest energy (CBM)
        VBM = bands[bands <= 0].max()
        CBM = bands[bands > 0].min()
        Eg = CBM - VBM
        print(f'Bandgap: {Eg:.3f} eV')
        # Shift bands to ensure Fermi level is in the center of the gap
        bands -= VBM
        bands -= Eg/2

        # Plot vertical lines at symmetry points
        ax.vlines(X, E_tickmarks[0], E_tickmarks[-1], color='0.5', lw=0.8)

        # Plot band-structures
        for j, e_k in enumerate(bands):
            if j == 0:
                ax.plot(x, e_k, color=col, label=f"{val}", lw=1)
            else:
                ax.plot(x, e_k, color=col, lw=1)

        #if mode == 'pw':
            # Remove last X point and label
            #X = X[:-1]
            #labels = labels[:-1]
        # Set x-ticks to the symmetry points and label them
        ax.xaxis.set_ticks(X)
        ax.set_xticklabels(labels)
        ax.set_xlim(0, 1)

        return CBM, VBM
    
    def _plot_dos(ax, dir, val, col='k', mode='lcao', shift=0, pDOS=False):
        atom_colors = {'Ba': 'tab:blue', 'Sr': 'tab:purple',
                       'Ti': 'tab:orange', 'O': 'tab:red'}
        if mode == 'lcao':
            # Read the PDOS data from the file generated by Siesta using SISL
            sile_PDOS = si.get_sile(os.path.join(dir, f"{formula}.PDOS"))
            geom, E, PDOS = sile_PDOS.read_data()
            orbits = geom.atoms.orbitals
            PDOS = np.squeeze(PDOS)
            DOS = PDOS.sum(axis=0)
            idx = np.cumsum(np.r_[0, orbits[:-1]])
            PDOS_atom = np.add.reduceat(PDOS, idx, axis=0)

        elif mode == 'pw':
            # Read the DOS data from the file generated by GPAW
            data = np.load(os.path.join(dir, f"{formula}_DOS.npz"), allow_pickle=True)
            E = np.array(data['E'], dtype=float)
            DOS = np.array(data['DOS'], dtype=float)
        # Plot total density of states (DOS)
        ax.plot(DOS, E - shift, color=col, label=f"{val}")
        if pDOS:
            # Plot PDOS for each atom
            for i in range(PDOS_atom.shape[0]):
                symbol = geom.atoms[i].symbol
                ax.plot(PDOS_atom[i], E - shift, color=atom_colors[symbol], alpha=0.5, lw=1)

    # Plot dashed line at Fermi level for both subplots
    ax1.axhline(y=0, color='k', linestyle=':', lw=0.8)
    ax2.axhline(y=0, color='k', linestyle=':', lw=0.8)

    dir = f'results/{struc}/GPAW/0.0/bands'
    CBM, VBM = _plot_bandstructure(ax1, dir, 'PW', mode='pw', col=colors[0])
    shift = VBM + (CBM - VBM)/2
    _plot_dos(ax2, dir, 'PW', mode='pw', shift=shift, col=colors[0])

    #dir = 'results/bulk/test_bands'
    #CBM, VBM = _plot_bandstructure(ax1, dir, 'test', col='orange')
    #shift = VBM + (CBM - VBM)/2
    #_plot_dos(ax2, dir, 'test', shift=shift, col='orange')
    
    # Cycle through the list of IDs and plot the bandstructure and DOS for each ID
    for i in range(len(ids)):
        dir = os.path.join('results', struc, ids[i], 'bands')

        # Subplot 1 - Band structure
        CBM, VBM = _plot_bandstructure(ax1, dir, vals[i], col=colors[i+1])
        shift = VBM + (CBM - VBM)/2
        # Subplot 2 - Density of states (DOS)
        _plot_dos(ax2, dir, vals[i], col=colors[i+1], shift=shift)
    
    # Set x- and y-label
    #ax1.set_xlabel('k-points')
    ax1.set_ylabel('Energy, $E-E_F$ (eV)')
    # Set x- and y-limits
    ax1.set_ylim(E_tickmarks[0], E_tickmarks[-1])
    # Add minor tickmarks to the y-axis
    ax1.yaxis.set_minor_locator(AutoMinorLocator())
    # Apply custom plot settings to the axes
    PlotSettings().set_style_ax(ax1, style='bands', minor=False)

    ax2.legend(loc='upper right')
    # Customize y-axis label and other parameters if needed
    ax2.set_xlabel('DOS (1/eV)')
    # Force x- and y-ticks
    ax2.set_xticks(dos_tickmarks, dos_tickmarks)
    ax2.set_yticks(E_tickmarks, E_tickmarks)
    # Set limits to match
    ax2.set_xlim(dos_tickmarks[0], dos_tickmarks[-1])
    ax2.set_ylim(E_tickmarks[0], E_tickmarks[-1])
    # Hide y-tick labels
    ax2.set_yticklabels([])
    # Apply custom plot settings to the axes
    PlotSettings().set_style_ax(ax2, style='bands')
    # Set figure size using the custom PlotSettings class
    PlotSettings().set_size(fig, width)
    # Show figure
    plt.show()

def plot_bands2(formula, ids=np.array([]), vals=np.array([]), bulk=True, Ncells=1, width=1):
    """Function to plot bandstructures seperately
    Requires that the bandstructure has already been calculated and saved to files.
    Parameters:
    - formula: Chemical formula of the material.
    - ids: Numpy array of IDs to plot.
    - vals: Numpy array corresponding to the IDs (e.g., different functionals or parameters).
    - bulk: Boolean indicating whether the structure is bulk (True) or slab (False) (default is True).
    - Ncells: Number of unit cells
    - width: Fraction of the target width for the figure (default is 1).
    Returns:
    - None. The function reads the bandstructure data from files and plots the results.
    """

    # Define tickmarks for the x- and y-axis
    E_tickmarks = np.arange(-6, 8, 2)
    # Define a list of colors for the plots (if needed)
    colors = ["black", "blue", "red", "purple", "orange", "green"]
    N = len(ids) + 1

    if bulk:
        struc = f'bulk/{formula}'
    else:
        struc = f'slab/{formula}/{Ncells}uc'

    # Create N subplots for the band structure along x
    fig, axes = plt.subplots(1, N, figsize=(2.5*N, 5), sharey='col')

    # Adjust spacing between subplots and set figure size using the custom PlotSettings class
    plt.subplots_adjust(wspace=0.05)
    #PlotSettings().set_size(fig)

    def _plot_bandstructure(ax, dir, val, col='k', mode='lcao'):

        if mode == 'lcao':
            # Read the bandstructure data from the file generated by Siesta using SISL
            sile_bands = si.get_sile(os.path.join(dir, f"{formula}.bands"))
            (X, labels), x, bands = sile_bands.read_data()
            bands = np.squeeze(bands).T

        elif mode == 'pw':
            # Read the bandstructure data from the file generated by GPAW
            data = np.load(os.path.join(dir, f"{formula}_BS.npz"), allow_pickle=True)
            (X, labels) = data['X']
            x = data['x']
            bands = data['bands']
        
        # Convert labels to use LaTeX symbols for Gamma point
        labels = convert_labels(labels)
        # Convert X, x and bands to numpy arrays of type float (if they are not already)
        X = np.array(X, dtype=float)
        labels = np.array(labels)
        x = np.array(x, dtype=float)
        bands = np.array(bands, dtype=float)
        # Shorten path to end with G
        indx = np.where(labels == r'$\Gamma$')[0][-1]
        labels = labels[:indx+1]
        X = X[:indx+1]
        # Normalize x and X to the total length of the path
        x /= X[-1]
        X /= X[-1]
        # Find lowest energy (VBM) and highest energy (CBM)
        VBM = bands[bands <= 0].max()
        CBM = bands[bands > 0].min()
        Eg = CBM - VBM
        print(f'Bandgap: {Eg:.3f} eV')
        # Shift bands to ensure Fermi level is in the center of the gap
        bands -= VBM
        bands -= Eg/2

        # Set title
        ax.set_title(f"{val}")
        # Plot horizontal line at Fermi level
        ax.axhline(y=0, color='k', linestyle='--', lw=0.8)

        # Plot vertical lines at symmetry points
        ax.vlines(X, E_tickmarks[0], E_tickmarks[-1], color='0.5', lw=0.8)

        # Plot band-structures
        for e_k in bands:
            ax.plot(x, e_k, color=col, lw=1)
        
        # Set y-ticks to the defined tickmarks and label them
        ax.set_yticks(E_tickmarks, E_tickmarks.astype(str))
        # Set x-ticks to the symmetry points and label them
        ax.set_xticks(X[:-1], labels[:-1])
        #ax.set_xticklabels(labels[:-1])
        # Set limits to match
        ax.set_xlim(X[0], X[-1])
        #ax.set_xlim(0, 1)
        ax.set_ylim(E_tickmarks[0], E_tickmarks[-1])
        # Add minor tickmarks to the y-axis
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        # Apply custom plot settings to the axes
        PlotSettings().set_style_ax(ax, style='default', minor=False)
    
    dir = f'results/{struc}/GPAW/0.0/bands'
    _plot_bandstructure(axes[0], dir, 'PW', col=colors[0], mode='pw')

    #dir = 'results/bulk/test_bands'
    #_plot_bandstructure(axes[0], dir, 'test', col=colors[0])
    axes[0].set_ylabel('Energy, $E-E_F$ (eV)')
    
    # Cycle through the list of IDs and plot the bandstructure and DOS for each ID
    for i in range(len(ids)):
        dir = os.path.join('results', struc, ids[i], 'bands')
        # Plot band-structureure for this ID
        _plot_bandstructure(axes[i+1], dir, vals[i], col=colors[i+1])
        # Remove y-tick labels for all but the first and last subplot
        if i < len(ids) - 1:
            axes[i+1].set_yticklabels([])
    
    # Move y-axis of the last subplot to the right but maintain the y-tickmarks on the left
    axes[-1].tick_params(axis='y', labelright=True, labelleft=False)
    # Set figure size using the custom PlotSettings class
    PlotSettings().set_size(fig, width)
    # Show figure
    plt.show()



def plot_DOS(formula, ids=np.array([]), vals=np.array([]), bulk=True, Ncells=1):
    """Function to plot total DOS and pDOS for multiple calculations
    Requires that the DOS and pDOS has already been calculated and saved to files.
    Parameters:
    - formula: Chemical formula of the material.
    - ids: Numpy array of IDs to plot.
    - vals: Numpy array corresponding to the IDs (e.g., different functionals or parameters).
    - bulk: Boolean indicating whether the structure is bulk (True) or slab (False) (default is True).
    - Ncells: Number of unit cells
    Returns:
    - None. The function reads the DOS and pDOS data from files and plots the results.
    """

    # Define tickmarks for the x- and y-axis
    E_tickmarks = np.arange(-6, 8, 2)
    dos_tickmarks = np.arange(0, 7, 1)
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

    def _plot_dos(ax, dir, val, col, mode, shift=0, pDOS=True):
        atom_colors = {'Ba': 'tab:blue', 'Sr': 'tab:purple',
                       'Ti': 'tab:orange', 'O': 'tab:red'}
        if mode == 'lcao':
            # Read the PDOS data from the file generated by Siesta using SISL
            sile_PDOS = si.get_sile(os.path.join(dir, f"{formula}.PDOS"))
            geom, E, PDOS = sile_PDOS.read_data()
            orbits = geom.atoms.orbitals
            PDOS = np.squeeze(PDOS)
            DOS = PDOS.sum(axis=0)
            idx = np.cumsum(np.r_[0, orbits[:-1]])
            PDOS_atom = np.add.reduceat(PDOS, idx, axis=0)

        elif mode == 'pw':
            # Read the DOS data from the file generated by GPAW
            data = np.load(os.path.join(dir, f"{formula}_DOS.npz"), allow_pickle=True)
            E = np.array(data['E'], dtype=float)
            DOS = np.array(data['DOS'], dtype=float)
        # Plot total density of states (DOS)
        ax.plot(DOS, E - shift, color=col, label=f"{val}")
        if pDOS:
            # Plot PDOS for each atom
            for i in range(PDOS_atom.shape[0]):
                symbol = geom.atoms[i].symbol
                ax.plot(PDOS_atom[i], E - shift, color=atom_colors[symbol], alpha=0.5)
        
        ax.legend(loc='upper right')
        # Customize y-axis label and other parameters if needed
        ax.set_xlabel('DOS (1/eV)')
        # Force x- and y-ticks
        ax.set_xticks(dos_tickmarks, dos_tickmarks.astype(str))
        ax.set_yticks(E_tickmarks, E_tickmarks.astype(str))
        # Set limits to match
        ax.set_xlim(dos_tickmarks[0], dos_tickmarks[-1])
        ax.set_ylim(E_tickmarks[0], E_tickmarks[-1])
        # Add minor tickmarks to the y-axis
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        # Apply custom plot settings to the axes
        PlotSettings().set_style_ax(ax, style='default', minor=False)

    
    dir = f'results/{struc}/GPAW/0.0/bands'
    _plot_dos(axes[0], dir, 'PW', col=colors[0], mode='pw', pDOS=False)

    #dir = 'results/bulk/test_bands'
    #_plot_bandstructure(axes[0], dir, 'test', col=colors[0])
    axes[0].set_ylabel('Energy, $E-E_F$ (eV)')
    
    # Cycle through the list of IDs and plot the bandstructure and DOS for each ID
    for i in range(len(ids)):
        dir = os.path.join('results', struc, ids[i], 'bands')
        # Plot band-structureure for this ID
        _plot_dos(axes[i+1], dir, vals[i], col=colors[i+1], mode='lcao')
        # Remove y-tick labels for all but the first and last subplot
        if i < len(ids) - 1:
            axes[i+1].set_yticklabels([])
    
    # Move y-axis of the last subplot to the right but maintain the y-tickmarks on the left
    axes[-1].tick_params(axis='y', labelright=True, labelleft=False)
    # Show figure
    plt.show()



# Define a function that plots the bandstructure and DOS together
def plot_bands_SISL(formula, dir='results/bulk/bandstructure'):
    """Legacy function to plot bandstructure and DOS for a given formula.
    Uses SISL to read the Hamiltonian and geometry, and to calculate the bandstructure and DOS.
    Parameters:
    - formula: Chemical formula of the material.
    Returns:
    - None. The function performs band structure calculation and plots the results.
    """
    # Read the Hamiltonian and geometry from the HSX file generated by Siesta
    sile_HSX = si.get_sile(os.path.join(dir, f"{formula}.HSX"))
    H = sile_HSX.read_hamiltonian()
    geom = sile_HSX.read_geometry()
    Ef = sile_HSX.read_fermi_level()
    print("Fermi level (eV):", Ef)
    
    # Define tickmarks for the x- and y-axis
    ytickmarks = np.arange(-6, 8, 2)
    xtickmarks = np.arange(0, 7, 1)
    
    # Make a simple figure where graphs are plotted
    fig = plt.figure(figsize=[6.6, 5])

    # Subplot 1 - Band structure
    ax1 = fig.add_axes([0, 0, 1, 1])
    # Calculate band-structure using SISL
    band = si.BandStructure(geom, [ [0.0, 0.0, 0.0],     # Gamma
                                    [0.5, 0.0, 0.0],     # X
                                    [0.5, 0.5, 0.5],     # R
                                    [0.5, 0.5, 0.0],     # M
                                    [0.0, 0.0, 0.0],     # Gamma
                                    [0.5, 0.5, 0.5]      # R
                                    ], 300,              # number of k-points
                                    [r'$\Gamma$','X', 'R', 'M', r'$\Gamma$', 'R'])
    xtick, xtick_label = band.lineartick()
    lk = band.lineark()
    ax1.xaxis.set_ticks(xtick)
    ax1.set_xticklabels(xtick_label)

    # Plot vertical lines at symmetry points
    ax1.vlines(xtick, ytickmarks[0], ytickmarks[-1], color='0.5', linewidth=1)
    # Plot dashed line at Fermi level
    ax1.axhline(y=0, color='k', linestyle=':')

    # Plot band-structures
    band.set_parent(H)
    eigs = band.apply.ndarray.eigh()
    ax1.plot(lk, eigs, color='k')

    # Set x- and y-label
    #ax1.set_xlabel('k-points')
    ax1.set_ylabel('Energy, $E-E_F$ (eV)')
    # Set x- and y-limits
    ax1.set_xlim(xtick[0], xtick[-1])
    ax1.set_ylim(ytickmarks[0], ytickmarks[-1])
    # Add minor tickmarks to the y-axis
    ax1.yaxis.set_minor_locator(AutoMinorLocator())
    
    # Subplot 2 - Density of states (DOS)
    ax2 = fig.add_axes([1.05, 0, 0.4, 1])
    # Calculate DOS using Monkhorst-Pack grid
    E = np.linspace(ytickmarks[0], ytickmarks[-1], 500)
    bz = si.MonkhorstPack(H, [40, 40, 40])
    dos = bz.apply.average.eigenvalue(wrap=lambda ev: ev.DOS(E))
    
    # Plot dashed line at Fermi level
    ax2.axhline(y=0, color='k', linestyle=':')
    
    # Plot total density of states
    ax2.plot(dos, E, 'k', label="DOS")
    
    # Customize y-axis label and other parameters if needed
    ax2.set_xlabel('DOS (1/eV)')
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