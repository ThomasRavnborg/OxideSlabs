# Libraries
from ase import Atoms
from ase.calculators.siesta import Siesta
from ase.calculators.siesta.parameters import Species, PAOBasisBlock
from ase.units import Ry
import os
from ase.optimize import BFGS
from ase.filters import FrechetCellFilter
import sisl as si

# Function to generate perovskite structure
def perovskite(formula):
    """Function to create a perovskite structure.
    Parameters:
    - formula: Chemical formula of the perovskite ('ABX3').
    Returns:
    - ASE Atoms object representing the perovskite structure.
    """
    # Dictionary of lattice constants for different perovskites
    a = {'BaTiO3': 4.01,
         'SrTiO3': 3.905
         }
    if formula not in a:
        raise ValueError(f"Lattice constant for {formula} not found in dictionary.")
    
    sca_pos = [[0, 0, 0],
                [1/2, 1/2, 1/2],
                [1/2, 0, 1/2],
                [0, 1/2, 1/2],
                [1/2, 1/2, 0]]
    def unitCell(a):
        return [[a, 0, 0], [0, a, 0], [0, 0, a]]
    return Atoms(formula, cell=unitCell(a[formula]), pbc=True, scaled_positions=sca_pos)

def relaxASE(atoms, xcf='PBE', basis='DZP', shift=0.01, split=0.15,
               cutoff=200, kmesh=[5, 5, 5], fmax=0.005, filt=True):
    """Function to relax a bulk structure using ASE BFGS optimizer with Siesta calculator.
    Parameters:
    - atoms: ASE Atoms object representing the structure to be relaxed.
    - xcf: Exchange-correlation functional to be used (default is 'PBE').
    - basis: Basis set to be used (default is 'DZP').
    - shift: Energy shift in Ry (default is 0.01 Ry).
    - split: Split norm for basis functions (default is 0.15).
    - cutoff: Mesh cutoff in Ry (default is 200 Ry).
    - kmesh: K-point mesh as a list (default is [5, 5, 5]).
    - fmax: Maximum force criterion for convergence in eV/Å (default is 0.005 eV/Å).
    - filt: Boolean indicating whether to optimize unit cell parameters (True) or only atomic positions (False).
    Returns:
    - None. The function performs the relaxation and saves the relaxed structure to an xyz file.
    """
    cwd = os.getcwd()
    symbols = atoms.symbols

    # Calculation parameters in a dictionary
    calc_params = {
        'label': f'{symbols}',
        'xc': xcf,
        'basis_set': basis,
        'mesh_cutoff': cutoff * Ry,
        'energy_shift': shift * Ry,
        'kpts': kmesh,
        'directory': 'bulk/relax/',
        'pseudo_path': cwd + '/pseudos'
    }
    # fdf arguments in a dictionary
    fdf_args = {
        'PAO.BasisSize': basis,
        'PAO.SplitNorm': split
    }

    # Set up the Siesta calculator and attach it to the atoms object
    calc = Siesta(**calc_params, fdf_arguments=fdf_args)
    atoms.calc = calc
    
    # Set filter to allow for unit cell and atoms to be simultaneously optimized
    if filt == True:
        # Set unit cell filter to the FrechetCellFilter
        opt_conf = FrechetCellFilter(atoms)
    elif filt == False:
        # Only optimize positions of the atoms
        opt_conf = atoms

    # Use BFGS optimizer
    opt = BFGS(opt_conf, logfile='bulk/relax/relax.log', trajectory='bulk/relax/relax.traj')
    # Run the optimization until forces are smaller than fmax
    opt.run(fmax=fmax)
    # Write atoms object to file
    atoms.write(f'bulk/relax/{symbols}.xyz')


def relaxSiesta(atoms, xcf='PBE', basis='DZP', shift=0.01, split=0.15,
                cutoff=200, kmesh=[5, 5, 5], fmax=0.005, smax=0.01):
    """Function to relax a bulk structure with a single Siesta calculation.
    Parameters:
    - atoms: ASE Atoms object representing the structure to be relaxed.
    - xcf: Exchange-correlation functional to be used (default is 'PBE').
    - basis: Basis set to be used (default is 'DZP').
    - split: Split norm for basis functions (default is 0.15).
    - cutoff: Mesh cutoff in Ry (default is 200 Ry).
    - shift: Energy shift in Ry (default is 0.01 Ry).
    - kmesh: K-point mesh as a list (default is [5, 5, 5]).
    - fmax: Maximum force criterion for convergence in eV/Å (default is 0.005 eV/Å).
    - smax: Maximum stress criterion for convergence in GPa (default is 0.01 GPa).
    Returns:
    - None. The function performs the relaxation and saves the relaxed structure to an xyz file.
    """
    cwd = os.getcwd()
    symbols = atoms.symbols

    # Species information for recognizing .psf pseudopotential files
    spc = [
      Species(symbol=f'{symbols[0]}', pseudopotential=f'{symbols[0]}.psf'),
      Species(symbol=f'{symbols[1]}', pseudopotential=f'{symbols[1]}.psf'),
      Species(symbol=f'{symbols[2]}', pseudopotential=f'{symbols[2]}.psf')
    ]

    # Calculation parameters in a dictionary
    calc_params = {
        'label': f'{symbols}',
        'xc': xcf,
        'basis_set': basis,
        'mesh_cutoff': cutoff * Ry,
        'energy_shift': shift * Ry,
        'kpts': kmesh,
        'directory': 'bulk/relaxsiesta/',
        'pseudo_path': cwd + '/pseudos'
    }
    
    fdf_args = {
        'PAO.BasisSize': basis,
        'PAO.SplitNorm': split,
        'WriteMDHistory': 'T',
        'MD.TypeOfRun': 'CG',
        'Diag.Algorithm': 'ELPA',
        'MD.Steps': '100',
        'MD.VariableCell': 'T',
        'MD.MaxForceTol': f'{fmax} eV/Ang',
        'MD.MaxStressTol': f'{smax} GPa',
        'MD.TargetPressure': '0.0 GPa'
    }
    
    # Set up the Siesta calculator and attach it to the atoms object
    calc = Siesta(**calc_params, fdf_arguments=fdf_args)
    atoms.calc = calc
    # Run the single Siesta calculation
    atoms.get_potential_energy()
    # Read relaxed structure geometry from HSX file
    sile = si.get_sile(f'bulk/relaxsiesta/{symbols}.XV')
    geom = sile.read_geometry()
    # Convert into ase atoms object and save to xyz file
    atoms = geom.to.ase()
    atoms.write(f'bulk/relaxsiesta/{symbols}.xyz')