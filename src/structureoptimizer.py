# Importing packages and modules
import os
import time
import numpy as np
import sisl as si
# ASE
from ase import Atoms
from ase.calculators.siesta import Siesta
from ase.calculators.siesta.parameters import Species, PAOBasisBlock
from ase.units import Ry
from ase.optimize import BFGS
from ase.filters import FrechetCellFilter
from ase.parallel import parprint
from ase.io import write
# Custom modules
from src.cleanfiles import cleanFiles

# Try to import world from gpaw.mpi for parallel processing
# If not available, fall back to ase.parallel.world
try:
    from gpaw.mpi import world
except ImportError:
    from ase.parallel import world

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

def filter(atoms, bulk=True):
    """Function to set up a filter for optimizing unit cell parameters and atomic positions.
    Parameters:
    - atoms: ASE Atoms object representing the structure to be optimized.
    - bulk: Boolean indicating whether the structure is bulk (True) or slab (False)
    Returns:
    - ASE Atoms object with the appropriate filter applied for optimization.
    """
    # Mask for cell optimization: 1 means optimize that parameter, 0 means keep it fixed
    # Format: [εxx, εyy, εzz, εyz, εxz, εxy]
    mask = [1, 1, 1, 1, 1, 1]
    if not bulk:
        # For slab calculations, only optimize in-plane cell parameters and atomic positions,
        # while keeping out-of-plane cell parameter fixed
        mask = [1, 1, 0, 0, 0, 1]
    # Set unit cell filter to the FrechetCellFilter
    atoms_filt = FrechetCellFilter(atoms, mask=mask)
    return atoms_filt

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
    # Define current working directory and extract information from the perovskite object
    cwd = os.getcwd()
    formula = perovskite.formula
    symbols = perovskite.symbols
    atoms = perovskite.atoms
    bulk = perovskite.bulk
    # Convert kgrid to a list to allow for modification
    kgrid = list(kgrid)

    if not bulk:
        # For slab calculations, set k-point sampling to 1 in the z-direction
        kgrid[2] = 1

    # Defining custom basis sets
    if basis in ['DZPp']:
        #basis = 'DZP'

        ba_basis = PAOBasisBlock("""5   # number of l-shells
        n=5   0   1                     # n, l, Nzeta 
            3.968   
            1.000   
        n=6   0   2                     # n, l, Nzeta 
            9.316      7.217   
            1.000      1.000   
        n=5   1   1                     # n, l, Nzeta 
            4.677   
            1.000   
        n=6   1   1                     # n, l, Nzeta 
            9.316   
            1.000
        n=5   2   1                     # n, l, Nzeta 
            8.000   
            1.000   
        """)

        sr_basis = PAOBasisBlock("""5   # number of l-shells
        n=4   0   1                     # n, l, Nzeta
            3.511
            1.000
        n=5   0   2                     # n, l, Nzeta
            8.773      6.728
            1.000      1.000
        n=4   1   1                     # n, l, Nzeta
            4.114
            1.000
        n=5   1   1                     # n, l, Nzeta
            8.773
            1.000
        n=4   2   1                     # n, l, Nzeta
            8.000
            1.000
        """)

        ti_basis = PAOBasisBlock("""6   # number of l-shells
        n=3   0   1                     # n, l, Nzeta
            2.844
            1.000
        n=4   0   2                     # n, l, Nzeta
            7.565      5.669
            1.000      1.000
        n=3   1   1                     # n, l, Nzeta
            3.189
            1.000
        n=4   1   1                     # n, l, Nzeta
            7.565
            1.000
        n=3   2   2                     # n, l, Nzeta
            5.233      3.466
            1.000      1.000
        n=3   3   1                     # n, l, Nzeta
            6.000
            1.000
        """)

        o_basis = PAOBasisBlock("""2    # number of l-shells
        n=2   0   2                     # n, l, Nzeta
            3.540      2.304
            1.000      1.000
        n=2   1   2 P   1               # n, l, Nzeta, Polarization, NzetaPol
            4.291      2.777
            1.000      1.000
        """)

        # Create dictionary of species with custom basis sets for SIESTA calculations
        basis_sets = {
            'Ba': ba_basis,
            'Sr': sr_basis,
            'Ti': ti_basis,
            'O': o_basis
        }

        species=[
            Species(symbol=symbols[0], basis_set=basis_sets[symbols[0]]),
            Species(symbol=symbols[1], basis_set=basis_sets[symbols[1]]),
            Species(symbol=symbols[2], basis_set=basis_sets[symbols[2]]),
        ]

    else:
        species=[
            Species(symbol=symbols[0], basis_set=basis),
            Species(symbol=symbols[1], basis_set=basis),
            Species(symbol=symbols[2], basis_set=basis),
        ]

    # For SIESTA, calculations are performed with atomic orbitals (LCAO)
    if mode == 'lcao':
        parprint(f"Relaxing structure for {formula} using SIESTA.")
        # Calculation parameters in a dictionary
        calc_params = {
            'label': f'{formula}',
            'xc': xcf,
            #'basis_set': basis,
            'mesh_cutoff': MeshCutoff * Ry,
            'energy_shift': EnergyShift * Ry,
            'kpts': kgrid,
            'directory': dir,
            'pseudo_path': os.path.join(cwd, 'pseudos', f'{pseudo}')
        }
        # fdf arguments in a dictionary
        fdf_args = {
            #'PAO.BasisSize': basis,
            'PAO.SplitNorm': SplitNorm,
            'SCF.DM.Tolerance': 1e-6,
        }
        if not bulk:
            # Add dipole correction for slab calculations to avoid spurious interactions between periodic images
            fdf_args['Slab.DipoleCorrection'] = 'T'
        # Set up the Siesta calculator
        calc = Siesta(species=species, **calc_params, fdf_arguments=fdf_args)
    
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
        if not bulk:
            # Add dipole correction for slab calculations to avoid spurious interactions between periodic images
            calc_params["poissonsolver"] = {"dipolelayer": "xy"}
        # Set up the GPAW calculator
        calc = GPAW(**calc_params)
    
    # Attach the calculator to the atoms object
    atoms.calc = calc

    # Apply filter to optimize unit cell parameters and atomic positions if filt is True
    # Otherwise only optimize atomic positions
    if filt:
        atoms_filt = filter(atoms, bulk=True)
    else:
        atoms_filt = atoms
    
    if world.rank == 0:
        t0 = time.time() # Start timer
    
    # Set up the BFGS optimizer on all processes
    opt = BFGS(atoms_filt,
               logfile=os.path.join(dir, f"{formula}.log"),
               trajectory=os.path.join(dir, f"{formula}.traj"))
    # Run the optimization until forces are smaller than fmax
    opt.run(fmax=fmax)

    if world.rank == 0:
        t1 = time.time() # Stop timer
        # Write the time taken for optimization to a file
        np.save(os.path.join(dir, f"time.npy"), t1-t0)
    
    # Write atoms object to a file
    write(os.path.join(dir, f"{formula}.xyz"), atoms)

    if mode == 'lcao':
        # Remove unnecessary files generated from SIESTA
        cleanFiles(directory=dir, confirm=False)
    # Wait for all parallel processes to finish
    world.barrier()
    


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

    # Convert kgrid to a list to allow for modification
    kgrid = list(kgrid)

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
        'MD.Steps': '100',
        'MD.VariableCell': 'T',
        'MD.MaxForceTol': f'{fmax} eV/Ang',
        'MD.MaxStressTol': f'{smax} GPa',
        'MD.TargetPressure': '0.0 GPa'
    }
    
    # Set up the Siesta calculator and attach it to the atoms object
    calc = Siesta(**calc_params, fdf_arguments=fdf_args)
    atoms.calc = calc
    # Run the Siesta calculation for relaxation and cell optimization
    t0 = time.time() # Start timer
    atoms.get_potential_energy()
    t1 = time.time() # Stop timer
    # Read relaxed structure geometry from HSX file
    sile = si.get_sile(os.path.join(dir, f"{formula}.XV"))
    geom = sile.read_geometry()
    # Convert into ase atoms object and save to xyz file
    atoms = geom.to.ase()
    atoms.write(os.path.join(dir, f"{formula}.xyz"))
    # Write the time taken for optimization to a file
    np.save(os.path.join(dir, f"time.npy"), t1-t0)
    # Remove unnecessary files generated during the relaxation
    cleanFiles(directory=dir, confirm=False)