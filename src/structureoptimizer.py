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
from ase.parallel import world
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

def relax_ase(perovskite, xcf='PBEsol', basis='DZP', EnergyShift=0.01, SplitNorm=0.15,
              MeshCutoff=200, kgrid=(10, 10, 10), fmax=0.005, pseudo='PBEsol', mode='lcao',
              filt=True, dir='results/bulk/relax'):
    """Function to relax a bulk structure using ASE BFGS optimizer with SIESTA or GPAW calculator.
    Parameters:
    - perovskite: Custom object representing the structure to be relaxed.
    - bulk: Boolean indicating whether the structure is bulk (True) or slab (False) (default is True).
    - xcf: Exchange-correlation functional to be used (default is 'PBEsol').
    - basis: Basis set to be used (default is 'DZP').
    - EnergyShift: Energy shift in Ry (default is 0.01 Ry).
    - SplitNorm: Split norm for basis functions (default is 0.15).
    - MeshCutoff: Mesh cutoff in Ry (default is 200 Ry).
    - kgrid: K-point mesh as a tuple (default is (10, 10, 10)).
    - fmax: Maximum force criterion for convergence in eV/Å (default is 0.005 eV/Å).
    - pseudo: Pseudopotential to be used (default is 'PBEsol').
    - mode: Calculator mode to be used ('lcao' for SIESTA or 'pw' for GPAW, default is 'lcao').
    - filt: Boolean indicating whether to optimize unit cell parameters (True) or only atomic positions (False).
    Returns:
    - None. The function performs the relaxation and saves the relaxed structure to an xyz file.
    """
    cwd = os.getcwd()
    formula = perovskite.formula
    atoms = perovskite.atoms
    bulk = perovskite.bulk

    if not bulk:
        # Center the slab in the cell and add vacuum in the z-direction
        atoms.center(axis=2, vacuum=10.0)
        # For slab calculations, set k-point sampling to 1 in the z-direction
        kgrid[2] = 1

    # For SIESTA, calculations are performed with atomic orbitals (LCAO)
    if mode == 'lcao':
        parprint(f"Relaxing structure for {formula} using SIESTA.")
        # Calculation parameters in a dictionary
        calc_params = {
            'label': f'{formula}',
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
        }
        # Set up the Siesta calculator
        calc = Siesta(**calc_params, fdf_arguments=fdf_args)
    
    # In GPAW, calculations are performed with plane waves (PW)
    elif mode == 'pw':
        from gpaw import GPAW
        parprint(f"Relaxing structure for {formula} using GPAW.")
        parprint('Note that shift and split do not apply to pw calculations and will be ignored.')
        calc_params = {
            'xc': xcf,
            'basis': basis.lower(),
            'mode': {'name': 'pw', 'ecut': MeshCutoff * Ry},
            'kpts': {'size': kgrid, 'gamma': True},
            'occupations': {'name': 'fermi-dirac','width': 0.05},
            'convergence': {'density': 1e-6, 'forces': 1e-5},
            'txt': os.path.join(dir, f"{formula}.txt")
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
               logfile=os.path.join(dir, f"{formula}.log"),
               trajectory=os.path.join(dir, f"{formula}.traj"))
    # Run the optimization until forces are smaller than fmax
    opt.run(fmax=fmax)
    if world.rank == 0:
        # Write atoms object to file (only on master process to avoid conflicts)
        atoms.write(os.path.join(dir, f"{formula}.xyz"))
    if mode == 'lcao':
        # Remove unnecessary files generated from SIESTA
        cleanFiles(directory=dir, confirm=False)


def relax_siesta(perovskite, xcf='PBEsol', basis='DZP', EnergyShift=0.01, SplitNorm=0.15,
                 MeshCutoff=200, kgrid=(10, 10, 10), pseudo='PBEsol', fmax=0.005, smax=0.01,
                 dir='results/bulk/relaxsiesta'):
    """Function to relax a bulk structure with a single Siesta calculation.
    Parameters:
    - perovskite: Custom object representing the structure to be relaxed.
    - bulk: Boolean indicating whether the structure is bulk (True) or slab (False) (default is True).
    - xcf: Exchange-correlation functional to be used (default is 'PBEsol').
    - basis: Basis set to be used (default is 'DZP').
    - EnergyShift: Energy shift in Ry (default is 0.01 Ry).
    - SplitNorm: Split norm for basis functions (default is 0.15).
    - MeshCutoff: Mesh cutoff in Ry (default is 200 Ry).
    - kgrid: K-point mesh as a tuple (default is (10, 10, 10)).
    - pseudo: Pseudopotential to be used (default is 'PBEsol').
    - fmax: Maximum force criterion for convergence in eV/Å (default is 0.005 eV/Å).
    - smax: Maximum stress criterion for convergence in GPa (default is 0.01 GPa).
    Returns:
    - None. The function performs the relaxation and saves the relaxed structure to an xyz file.
    """
    cwd = os.getcwd()
    formula = perovskite.formula
    atoms = perovskite.atoms
    bulk = perovskite.bulk
    symbols = atoms.symbols

    # Species information for setting up ghost atoms
    spc = [
        Species(symbol=f'{symbols[0]}'),
        Species(symbol=f'{symbols[0]}', ghost=True),
        Species(symbol=f'{symbols[1]}'),
        Species(symbol=f'{symbols[1]}', ghost=True),
        Species(symbol=f'{symbols[2]}'),
        Species(symbol=f'{symbols[2]}', ghost=True)
    ]

    if not bulk:
        # Center the slab in the cell and add vacuum in the z-direction
        atoms.center(axis=2, vacuum=10.0)
        # For slab calculations, set k-point sampling to 1 in the z-direction
        kgrid[2] = 1

    # Calculation parameters in a dictionary
    calc_params = {
        'label': f'{formula}',
        'xc': xcf,
        'basis_set': basis,
        'mesh_cutoff': MeshCutoff * Ry,
        'energy_shift': EnergyShift * Ry,
        'kpts': kgrid,
        'directory': dir,
        'pseudo_path': os.path.join(cwd, 'pseudos', f'{pseudo}'),
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
    sile = si.get_sile(os.path.join(dir, f"{formula}.XV"))
    geom = sile.read_geometry()
    # Convert into ase atoms object and save to xyz file
    atoms = geom.to.ase()
    atoms.write(os.path.join(dir, f"{formula}.xyz"))
    # Remove unnecessary files generated during the relaxation
    cleanFiles(directory=dir, confirm=False)