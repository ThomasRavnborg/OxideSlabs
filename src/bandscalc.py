# Importing packages and modules
import os
import numpy as np
import sisl as si
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
# ASE
from ase.io import read
from ase.calculators.siesta import Siesta
from ase import Atoms
from ase.units import Ry
from ase.dft.kpoints import bandpath
from ase.parallel import world
from ase.parallel import parprint
# GPAW
from gpaw import GPAW
# Custom modules
from src.cleanfiles import cleanFiles
from src.plotsettings import PlotSettings
PlotSettings().set_global_style()

def calculate_bands(atoms, bulk=True, xcf='PBEsol', basis='DZP', EnergyShift=0.01, SplitNorm=0.15,
                    MeshCutoff=200, kgrid=(10, 10, 10), pseudo=1, mode='lcao',
                    dir='results/bulk/bandstructure', par=False):
    """Function to calculate band structure and PDOS of a bulk structure using SIESTA.
    Parameters:
    - atoms: ASE Atoms object representing the relaxed bulk structure.
    - bulk: Boolean indicating whether the structure is bulk (True) or slab (False) (default is True).
    - xcf: Exchange-correlation functional to be used (default is 'PBEsol').
    - basis: Basis set to be used (default is 'DZP').
    - EnergyShift: Energy shift in Ry (default is 0.01 Ry).
    - SplitNorm: Split norm for basis functions (default is 0.15).
    - MeshCutoff: Mesh cutoff in Ry (default is 200 Ry).
    - kgrid: K-point mesh as a tuple (default is (10, 10, 10)).
    - pseudo: Integer index for selecting pseudopotential (default is 1).
    - mode: Calculator mode to be used ('lcao' for SIESTA or 'pw' for GPAW, default is 'lcao').
    - par: Whether the SIESTA calculator is parallel (default is False).
    Returns:
    - None. The function performs band structure calculation and saves the data to a file.
    """
    cwd = os.getcwd()
    symbols = atoms.symbols

    if mode == 'lcao':
        parprint(f"Calculating band structure and PDOS for {symbols} using SIESTA.")
        # Calculation parameters in a dictionary
        calc_params = {
            'label': f'{symbols}',
            'xc': xcf,
            'basis_set': basis,
            'mesh_cutoff': MeshCutoff * Ry,
            'energy_shift': EnergyShift * Ry,
            'kpts': kgrid,
            'directory': dir,
            'pseudo_path': os.path.join(cwd, 'pseudos', f'{pseudo}')
        }
        # fdf arguments in a dictionary
        fdf_args = {
            'PAO.BasisSize': basis,
            'PAO.SplitNorm': SplitNorm,
            'SCF.DM.Tolerance': 1e-6,
            'BandLinesScale': 'ReciprocalLatticeVectors',
            '%block BandLines': '''
            1 0.000 0.000 0.000 \Gamma
            60 0.500 0.000 0.000 X
            60 0.500 0.500 0.500 R
            60 0.500 0.500 0.000 M
            60 0.000 0.000 0.000 \Gamma
            60 0.500 0.500 0.500 R
            %endblock BandLines''',
            'BandLinesScale': 'ReciprocalLatticeVectors',
            '%block PDOS.kgrid_Monkhorst_Pack': '''
            40  0  0  0.0
            0 40  0  0.0
            0  0 40  0.0
            %endblock PDOS.kgrid_Monkhorst_Pack''',
            '%block Projected.DensityOfStates': '''
            -20.00 15.00 0.200 500 eV
            %endblock Projected.DensityOfStates'''
        }

        if par:
            # Change diagonalization algorithm when running in parallel
            fdf_args['Diag.Algorithm'] = 'ELPA'

        # Set up the Siesta calculator
        calc = Siesta(**calc_params, fdf_arguments=fdf_args)

    elif mode == 'pw':
        #from gpaw import GPAW
        calc_params = {
            'xc': xcf,
            'basis': basis.lower(),
            'mode': {'name': 'pw', 'ecut': MeshCutoff * Ry},
            'kpts': {'size': kgrid, 'gamma': True},
            'occupations': {'name': 'fermi-dirac','width': 0.05},
            'convergence': {'density': 1e-6, 'forces': 1e-5},
            'txt': None
        }
        # Set up the GPAW calculator
        calc = GPAW(**calc_params)

    # Attach the calculator to the atoms object and run calculation
    atoms.calc = calc
    atoms.get_potential_energy()

    if mode == 'pw':
        # Calculate bandstructure along a path (in this case 'GXRMGR')    
        BScalc = atoms.calc.fixed_density(
            nbands=32,
            symmetry='off',
            kpts={'path': 'GXRMGR', 'npoints': 300},
            convergence={'bands': 16},
            txt=os.path.join(dir, f"{symbols}_BS.txt"))
        
        path = bandpath('GXRMGR', BScalc.atoms.cell, npoints=300)
        
        x, X, labels = path.get_linear_kpoint_axis()
        X = np.array([X, labels])
    
        # Find Fermi energy
        ef = BScalc.get_fermi_level()
        # These arrays give us the datapoints and off-sets to the fermi energy
        e_kn = np.array([BScalc.get_eigenvalues(kpt=k) for k in range(len(BScalc.get_ibz_k_points()))])
        e_nk = e_kn.T
        e_nk -= ef
        # The density of states (DOS) is calculated
        E, DOS = BScalc.get_dos(spin=0, npts=2000, width=0.2)
        # Shift energy values to fermi level
        E -= ef
        # Save the bandstructure and DOS data to files on the master process
        if world.rank == 0:
            # Save the bandstructure data to a file
            np.savez(os.path.join(dir, f"{symbols}_BS.npz"), X=X, x=x, bands=e_nk)
            # Save the DOS data to a file
            np.savez(os.path.join(dir, f"{symbols}_DOS.npz"), E=E, DOS=DOS)
    elif mode == 'lcao':
        # Remove unnecessary files generated from SIESTA
        cleanFiles(directory=dir, confirm=False)

# Define a function that plots the bandstructure and DOS together
def plot_bands(formula, ids=np.array([]), vals=np.array([])):
    """Function to plot bandstructure and DOS for a given formula.
    Requires that the bandstructure and PDOS have already been calculated and saved to files.
    Parameters:
    - formula: Chemical formula of the material.
    - ids: Numpy array of IDs to plot.
    - vals: Numpy array corresponding to the IDs (e.g., different functionals or parameters).
    Returns:
    - None. The function reads the bandstructure and PDOS data from files and plots the results.
    """

    # Define tickmarks for the x- and y-axis
    ytickmarks = np.arange(-6, 8, 2)
    xtickmarks = np.arange(0, 7, 1)
    
    # Make a simple figure where graphs are plotted
    fig = plt.figure(figsize=[6.6, 5])

    # Create axes objects for the band structure and DOS subplots
    ax1 = fig.add_axes([0, 0, 1, 1])
    ax2 = fig.add_axes([1.05, 0, 0.4, 1])

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
        
        # Convert X, x and bands to numpy arrays of type float (if they are not already)
        X = np.array(X, dtype=float)
        x = np.array(x, dtype=float)
        bands = np.array(bands, dtype=float)
        # Normalize X and x to the total length of the path (the last value in X and x respectively)
        X /= X[-1]
        x /= x[-1]
        # Find lowest energy (VBM) and highest energy (CBM)
        VBM = bands[bands <= 0].max()
        CBM = bands[bands > 0].min()
        Eg = CBM - VBM
        print(f'Bandgap: {Eg:.3f} eV')
        # Shift bands to set VBM at 0 eV
        bands -= VBM

        # Set x-ticks to the symmetry points and label them
        ax.xaxis.set_ticks(X)
        ax.set_xticklabels(labels)
        # Plot vertical lines at symmetry points
        ax.vlines(X, ytickmarks[0], ytickmarks[-1], color='0.5', linewidth=1)
        ax.set_xlim(X[0], X[-1])

        # Plot band-structures
        for j, e_k in enumerate(bands):
            if j == 0:
                ax.plot(x, e_k, color=col, label=f"{val}")
            else:
                ax.plot(x, e_k, color=col)
        return VBM
    
    def _plot_dos(ax, dir, val, col='k', mode='lcao', shift=0):
        if mode == 'lcao':
            # Read the PDOS data from the file generated by Siesta using SISL
            sile_PDOS = si.get_sile(os.path.join(dir, f"{formula}.PDOS"))
            geom, E, PDOS = sile_PDOS.read_data()
            PDOS = np.squeeze(PDOS)
            DOS = PDOS.sum(axis=0)

        elif mode == 'pw':
            # Read the DOS data from the file generated by GPAW
            data = np.load(os.path.join(dir, f"{formula}_DOS.npz"), allow_pickle=True)
            E = np.array(data['E'], dtype=float)
            DOS = np.array(data['DOS'], dtype=float)
        # Plot total density of states (DOS)
        ax.plot(DOS, E - shift, color=col, label=f"{val}")

    # Plot dashed line at Fermi level for both subplots
    ax1.axhline(y=0, color='k', linestyle=':')
    ax2.axhline(y=0, color='k', linestyle=':')

    dir = 'results/bulk/GPAW'
    VBM = _plot_bandstructure(ax1, dir, 'GPAW', mode='pw')
    _plot_dos(ax2, dir, 'GPAW', mode='pw', shift=VBM)
    
    # Cycle through the list of IDs and plot the bandstructure and DOS for each ID
    for i in range(len(ids)):
        dir = os.path.join('results/bulk/',formula, ids[i], 'bands')
        # Cycle through a list of colors for the plots
        col = plt.get_cmap("viridis")(i / (len(ids) - 1))

        # Subplot 1 - Band structure
        VBM = _plot_bandstructure(ax1, dir, vals[i], col=col)

        # Subplot 2 - Density of states (DOS)
        _plot_dos(ax2, dir, vals[i], col=col, shift=VBM)
    
    # Set x- and y-label
    ax1.set_xlabel('k-points')
    ax1.set_ylabel('Energy, $E-E_F$ (eV)')
    # Set x- and y-limits
    ax1.set_ylim(ytickmarks[0], ytickmarks[-1])
    # Add minor tickmarks to the y-axis
    ax1.yaxis.set_minor_locator(AutoMinorLocator())
    # Apply custom plot settings to the axes
    PlotSettings().set_style_ax(ax1, style='bands', minor=False)

    ax2.legend(loc='upper right')
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
    # Apply custom plot settings to the axes
    PlotSettings().set_style_ax(ax2, style='bands')
    
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
    ax1.set_xlabel('k-points')
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