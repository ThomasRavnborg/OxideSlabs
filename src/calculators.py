import os
import numpy as np
from ase.units import Ry
from ase.optimize import BFGS
from ase.filters import FrechetCellFilter
from ase.calculators.siesta import Siesta
from ase.calculators.singlepoint import SinglePointCalculator
from src.structure import is_atom_bulk
from src.cleanfiles import cleanFiles

def run_siesta(atoms, xcf='PBEsol', basis='DZPp',
               EnergyShift=0.01, SplitNorm=0.15,
               MeshCutoff=1000, kgrid=(10, 10, 10),
               dir='results/bulk/basis'):
    """Function to run a single Siesta self-consistent calculation
    Arguments:
    - atoms: ASE Atoms object representing the structure to be relaxed.
    - xcf: Exchange-correlation functional to be used (default is 'PBEsol').
    - basis: Basis set to use for the calculation (default: 'DZP').
             If basis ends with (lower-case) p, a polarization orbital will be added to the A-site (Ba)
    - EnergyShift: Energy shift in Ry (default is 0.01 Ry).
    - SplitNorm: Split norm for basis functions (default is 0.15).
    - MeshCutoff: Mesh cutoff in Ry (default is 1000 Ry).
    - kgrid: K-point mesh as a tuple (default is (10, 10, 10)).
    - dir: Directory to save the results (default is 'results/bulk/basis').
    Returns:
    - None. The function runs the Siesta calculation and saves the results in the specified directory.
    """
    # Define current working directory and extract information from the perovskite object
    cwd = os.getcwd()
    formula = atoms.get_chemical_formula()

    # Custom basis sets ending with 'p' are generated with the same parameters as the standard basis sets
    # However, an extra polarization (d) orbital is added to the A-site during LCAO basis generation
    if basis.endswith('p'):
        basis = basis[:-1]

    kgrid = list(kgrid)
    if not is_atom_bulk(atoms):
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
        'pseudo_path': os.path.join(cwd, f'pseudos/{xcf}')
    }
    dir_fdf = os.path.join(cwd, dir)
    # fdf arguments in a dictionary
    fdf_args = {
        '%include': os.path.join(dir_fdf, 'basis.fdf'),
        'PAO.SplitNorm': SplitNorm,
        'SCF.DM.Tolerance': 1e-6
    }
    if not is_atom_bulk(atoms):
        # Add dipole correction for slab calculations to avoid spurious interactions between periodic images
        fdf_args['Slab.DipoleCorrection'] = 'T'
    
    # Set up the Siesta calculator and attach it to the atoms object
    calc = Siesta(**calc_params, fdf_arguments=fdf_args)
    atoms.calc = calc
    # Run the calculation
    atoms.get_potential_energy()

    # Clean directory of SIESTA calculations
    cleanFiles(directory=dir, confirm=False)


def opt_filter(atoms, strained=False, mask=None):
    """Function to set up a filter for optimizing unit cell parameters and atomic positions.
    Parameters:
    - atoms: ASE Atoms object representing the structure to be optimized.
    - mask: List of integers (0 or 1) indicating which cell parameters to optimize.
            Format is [εxx, εyy, εzz, εyz, εxz, εxy]. By default, all parameters are optimized.
            If the structure is a slab, the out-of-plane cell parameters (εzz, εyz, εxz) are set to 0 to keep them fixed.
    - strained: Boolean indicating whether the structure is strained (True) or not (False)
    Returns:
    - ASE Atoms object with the appropriate filter applied for optimization.
    """
    # Check if atoms object is bulk or slab
    bulk = is_atom_bulk(atoms)
    if mask is None:
        # By default, optimize all cell parameters
        mask = [1, 1, 1, 1, 1, 1]
    if not bulk:
        # For slab calculations, keep the out-of-plane cell parameter fixed
        mask[2] = 0
        mask[3] = 0
        mask[4] = 0
    if strained:
        # For strained calculations, keep the in-plane cell parameters fixed
        mask[0] = 0
        mask[1] = 0
        mask[5] = 0
    # Note that if slab and strained, none of the cell parameters will be optimized

    # Set unit cell filter to the FrechetCellFilter
    atoms_filt = FrechetCellFilter(atoms, mask=mask)
    return atoms_filt


def relax_siesta(atoms, xcf='PBEsol', basis='DZPp',
                 EnergyShift=0.01, SplitNorm=0.15,
                 MeshCutoff=1000, kgrid=(10, 10, 10),
                 dir='results/bulk/relax', strained=False):
    """Function to relax a bulk structure using ASE BFGS optimizer with SIESTA calculator.
    Parameters:
    - atoms: ASE Atoms object representing the structure to be relaxed.
    - xcf: Exchange-correlation functional to be used (default is 'PBEsol').
    - basis: Basis set to use for the calculation (default: 'DZP').
             If basis ends with (lower-case) p, a polarization orbital will be added to the A-site (Ba)
    - EnergyShift: Energy shift in Ry (default is 0.01 Ry).
    - SplitNorm: Split norm for basis functions (default is 0.15).
    - MeshCutoff: Mesh cutoff in Ry (default is 1000 Ry).
    - kgrid: K-point mesh as a tuple (default is (10, 10, 10)).
    - dir: Directory to save the results (default is 'results/bulk/relax').
    - strained: Boolean indicating whether strain has been applied to the structure (default is False, meaning no strain).
    Returns:
    - None. The function performs the relaxation and saves the relaxed structure to an xyz file.
    """
    # Define current working directory and extract information from the perovskite object
    cwd = os.getcwd()
    formula = atoms.get_chemical_formula()

    # Relaxation parameters
    fmax = 1    # meV/Å
    filt = True

    # Custom basis sets ending with 'p' are generated with the same parameters as the standard basis sets
    # However, an extra polarization (d) orbital is added to the A-site during LCAO basis generation
    if basis.endswith('p'):
        basis = basis[:-1]

    kgrid = list(kgrid)
    if not is_atom_bulk(atoms):
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
        'pseudo_path': os.path.join(cwd, f'pseudos/{xcf}')
    }
    dir_fdf = os.path.join(cwd, dir)
    # fdf arguments in a dictionary
    fdf_args = {
        '%include': os.path.join(dir_fdf, 'basis.fdf'),
        'PAO.SplitNorm': SplitNorm,
        'SCF.DM.Tolerance': 1e-6
    }
    if not is_atom_bulk(atoms):
        # Add dipole correction for slab calculations to avoid spurious interactions between periodic images
        fdf_args['Slab.DipoleCorrection'] = 'T'
    
    # Set up the Siesta calculator and attach it to the atoms object
    calc = Siesta(**calc_params, fdf_arguments=fdf_args)
    atoms.calc = calc

    # Apply filter to optimize unit cell parameters and atomic positions if filt is True
    # Otherwise only optimize atomic positions
    if filt:
        atoms_filt = opt_filter(atoms, strained)
    else:
        atoms_filt = atoms
   
    # Set up the BFGS optimizer on all processes
    opt = BFGS(atoms_filt,
               logfile=os.path.join(dir, f"{formula}.log"),
               trajectory=os.path.join(dir, f"{formula}.traj"))
    # Run the optimization until forces are smaller than fmax
    opt.run(fmax=fmax*1.e-3)

    # Remove unnecessary files generated from SIESTA
    cleanFiles(directory=dir, confirm=False)


def copy_calc_results(ase_atoms, sort=False):
    """Function to copy the results of a calculation from an ASE Atoms object to a new one.
    Arguments:
        ase_atoms (ase.Atoms): The ASE Atoms object containing the results of a calculation.
        sort (bool): Whether to sort the atoms by alphabetical order of chemical symbols (default: False).
    Returns:
        atoms_copy (ase.Atoms): A new ASE Atoms object with the same structure and the results of the calculation copied from the original one.
    """

    symbols = ase_atoms.get_chemical_symbols()

    if sort:
        indices = np.argsort(symbols)
    else:
        indices = np.arange(len(symbols))

    # Extract the results of the calculation from the original ASE Atoms object
    energy = ase_atoms.get_potential_energy()
    forces = ase_atoms.get_forces()[indices]
    stress = ase_atoms.get_stress()
    # Create a new ASE Atoms object with the same structure as the original one
    atoms_copy = ase_atoms.copy()[indices]
    # Assign the results of the calculation to the new ASE Atoms object using a SinglePointCalculator
    calc = SinglePointCalculator(
        atoms_copy,
        energy=energy,
        forces=forces,
        stress=stress
    )
    atoms_copy.calc = calc
    return atoms_copy