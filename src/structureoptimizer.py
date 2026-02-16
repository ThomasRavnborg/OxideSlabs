# Importing packages and modules
import os
import sisl as si
# ASE
from ase import Atoms
from ase.calculators.siesta import Siesta
from ase.calculators.siesta.parameters import Species, PAOBasisBlock
from ase.units import Ry
from ase.optimize import BFGS
from ase.filters import FrechetCellFilter
from ase.parallel import parprint
# Custom modules
from src.cleanfiles import cleanFiles

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

def relax_ase(atoms, xcf='PBEsol', basis='DZP', EnergyShift=0.01, SplitNorm=0.15,
              MeshCutoff=200, kgrid=(10, 10, 10), fmax=0.005, mode='lcao', filt=True, dir='results/bulk/relax'):
    """Function to relax a bulk structure using ASE BFGS optimizer with SIESTA or GPAW calculator.
    Parameters:
    - atoms: ASE Atoms object representing the structure to be relaxed.
    - xcf: Exchange-correlation functional to be used (default is 'PBEsol').
    - basis: Basis set to be used (default is 'DZP').
    - EnergyShift: Energy shift in Ry (default is 0.01 Ry).
    - SplitNorm: Split norm for basis functions (default is 0.15).
    - MeshCutoff: Mesh cutoff in Ry (default is 200 Ry).
    - kgrid: K-point mesh as a tuple (default is (10, 10, 10)).
    - fmax: Maximum force criterion for convergence in eV/Å (default is 0.005 eV/Å).
    - mode: Calculator mode to be used ('lcao' for SIESTA or 'pw' for GPAW, default is 'lcao').
    - filt: Boolean indicating whether to optimize unit cell parameters (True) or only atomic positions (False).
    Returns:
    - None. The function performs the relaxation and saves the relaxed structure to an xyz file.
    """
    cwd = os.getcwd()
    symbols = atoms.symbols

    # For SIESTA, calculations are performed with atomic orbitals (LCAO)
    if mode == 'lcao':
        parprint(f"Relaxing structure for {symbols} using SIESTA.")
        # Calculation parameters in a dictionary
        calc_params = {
            'label': f'{symbols}',
            'xc': xcf,
            'basis_set': basis,
            'mesh_cutoff': MeshCutoff * Ry,
            'energy_shift': EnergyShift * Ry,
            'kpts': kgrid,
            'directory': dir,
            'pseudo_path': os.path.join(cwd, 'pseudos')
        }
        # fdf arguments in a dictionary
        fdf_args = {
            'PAO.BasisSize': basis,
            'PAO.SplitNorm': SplitNorm
        }
        # Set up the Siesta calculator
        calc = Siesta(**calc_params, fdf_arguments=fdf_args)
    
    # In GPAW, calculations are performed with plane waves (PW)
    elif mode == 'pw':
        parprint(f"Relaxing structure for {symbols} using GPAW.")
        from gpaw import GPAW
        parprint('Note that shift and split do not apply to pw calculations and will be ignored.')
        calc_params = {
            'xc': xcf,
            'basis': basis.lower(),
            'mode': {'name': 'pw', 'ecut': MeshCutoff * Ry},
            'kpts': {'size': kgrid, 'gamma': True},
            'occupations': {'name': 'fermi-dirac','width': 0.05},
            'convergence': {'density': 1e-6, 'forces': 1e-5},
            'txt': os.path.join(dir, f"{symbols}.txt")
        }
        # Set up the GPAW calculator
        calc = GPAW(**calc_params)
    
    # Attach the calculator to the atoms object
    atoms.calc = calc
    
    # Set filter to allow for unit cell and atoms to be simultaneously optimized
    if filt == True:
        # Set unit cell filter to the FrechetCellFilter
        opt_conf = FrechetCellFilter(atoms)
    elif filt == False:
        # Only optimize positions of the atoms
        opt_conf = atoms

    # Use BFGS optimizer
    opt = BFGS(opt_conf,
               logfile=os.path.join(dir, f"{symbols}.log"),
               trajectory=os.path.join(dir, f"{symbols}.traj"))
    # Run the optimization until forces are smaller than fmax
    opt.run(fmax=fmax)
    # Write atoms object to file
    atoms.write(os.path.join(dir, f"{symbols}.xyz"))
    # Remove unnecessary files generated during the relaxation
    cleanFiles(directory=dir, confirm=False)


def relax_siesta(atoms, xcf='PBEsol', basis='DZP', EnergyShift=0.01, SplitNorm=0.15,
                 MeshCutoff=200, kgrid=(10, 10, 10), fmax=0.005, smax=0.01, dir='results/bulk/relaxsiesta'):
    """Function to relax a bulk structure with a single Siesta calculation.
    Parameters:
    - atoms: ASE Atoms object representing the structure to be relaxed.
    - xcf: Exchange-correlation functional to be used (default is 'PBEsol').
    - basis: Basis set to be used (default is 'DZP').
    - EnergyShift: Energy shift in Ry (default is 0.01 Ry).
    - SplitNorm: Split norm for basis functions (default is 0.15).
    - MeshCutoff: Mesh cutoff in Ry (default is 200 Ry).
    - kgrid: K-point mesh as a tuple (default is (10, 10, 10)).
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
        'mesh_cutoff': MeshCutoff * Ry,
        'energy_shift': EnergyShift * Ry,
        'kpts': kgrid,
        'directory': dir,
        'pseudo_path': os.path.join(cwd, 'pseudos')
    }
    
    fdf_args = {
        'PAO.BasisSize': basis,
        'PAO.SplitNorm': SplitNorm,
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
    sile = si.get_sile(os.path.join(dir, f"{symbols}.XV"))
    geom = sile.read_geometry()
    # Convert into ase atoms object and save to xyz file
    atoms = geom.to.ase()
    atoms.write(os.path.join(dir, f"{symbols}.xyz"))
    # Remove unnecessary files generated during the relaxation
    cleanFiles(directory=dir, confirm=False)