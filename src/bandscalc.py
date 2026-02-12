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
from ase.parallel import parprint
# Custom modules
from src.cleanfiles import cleanFiles
from src.plotsettings import PlotSettings
PlotSettings().set_global_style()

def calculate_bands(atoms, xcf='PBE', basis='DZP', shift=0.01, split=0.15, cutoff=200, kmesh=[5, 5, 5]):
    """Function to calculate band structure and PDOS of a bulk structure using SIESTA.
    Parameters:
    - atoms: ASE Atoms object representing the relaxed bulk structure.
    - xcf: Exchange-correlation functional to be used (default is 'PBE').
    - basis: Basis set to be used (default is 'DZP').
    - shift: Energy shift in Ry (default is 0.01 Ry).
    - split: Split norm for basis functions (default is 0.15).
    - cutoff: Mesh cutoff in Ry (default is 200 Ry).
    - kmesh: K-point mesh as a list (default is [5, 5, 5]).
    Returns:
    - None. The function performs band structure calculation and saves the data to a file.
    """
    cwd = os.getcwd()
    dir = 'results/bulk/bandstructure/'
    symbols = atoms.symbols

    # Calculation parameters in a dictionary
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
    # fdf arguments in a dictionary
    fdf_args = {
'PAO.BasisSize': basis,
'PAO.SplitNorm': split,
'BandLinesScale': 'ReciprocalLatticeVectors',
'%block BandLines': '''
1 0.000 0.000 0.000 \Gamma
60 0.500 0.000 0.000 X
60 0.500 0.500 0.500 R
60 0.000 0.500 0.000 M
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
    parprint(f"Calculating band structure and PDOS for {symbols}.")
    # Set up the Siesta calculator and attach it to the atoms object
    calc = Siesta(**calc_params, fdf_arguments=fdf_args)
    atoms.calc = calc
    atoms.get_potential_energy()
    # Remove unnecessary files generated during the relaxation
    cleanFiles(directory=dir, confirm=False)

# Define a function that plots the bandstructure and DOS together
def plot_bands(formula):
    """Function to plot bandstructure and DOS for a given formula.
    Requires that the bandstructure and PDOS have already been calculated and saved to files.
    Parameters:
    - formula: Chemical formula of the material.
    Returns:
    - None. The function reads the bandstructure and PDOS data from files and plots the results.
    """
    dir = 'results/bulk/bandstructure/'
    sile_HSX = si.get_sile(f'{dir}{formula}.HSX')
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
                                    ], 301,              # number of k-points
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

    sile_PDOS = si.get_sile(f'{dir}{formula}.PDOS')
    geom, E, PDOS = sile_PDOS.read_data()
    PDOS = np.squeeze(PDOS)
    ax2 = fig.add_axes([1.05, 0, 0.4, 1])
    # Add title
    #ax2.set_title('Density of states')
    # Plot dashed line at Fermi level
    ax2.axhline(y=0, color='k', linestyle=':')
    # Plot total density of states

    ax2.plot(PDOS.sum(axis=0), E, 'k', label="DOS")
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


# Define a function that plots the bandstructure and DOS together
def plotBandstructureSISL(formula):
    """Legacy function to plot bandstructure and DOS for a given formula.
    Uses SISL to read the Hamiltonian and geometry, and to calculate the bandstructure and DOS.
    Parameters:
    - formula: Chemical formula of the material.
    Returns:
    - None. The function performs band structure calculation and plots the results.
    """
    dir = 'results/bulk/relax/'
    # Read the Hamiltonian and geometry from the HSX file generated by Siesta
    sile_HSX = si.get_sile(f'{dir}{formula}.HSX')
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
                                    ], 301,              # number of k-points
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