import os
from ase.units import Ry
from ase.calculators.siesta import Siesta
from src.structure import get_reduced_formula, check_if_bulk
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
    - dir: Directory to save the results (default is 'results/bulk/phonons').
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
    if not check_if_bulk(atoms):
        # For slab calculations, set k-point sampling to 1 in the z-direction
        kgrid[2] = 1

    #kspacing = kspacing_from_kgrid(atoms, kgrid)
    #kgrid = kgrid_from_kspacing(atoms, kspacing)

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
    if not check_if_bulk(atoms):
        # Add dipole correction for slab calculations to avoid spurious interactions between periodic images
        fdf_args['Slab.DipoleCorrection'] = 'T'
    
    # Set up the Siesta calculator and attach it to the atoms object
    calc = Siesta(**calc_params, fdf_arguments=fdf_args)
    atoms.calc = calc
    # Run the calculation
    atoms.get_potential_energy()

    # Clean directory of SIESTA calculations
    cleanFiles(directory=dir, formats=['.DM'], confirm=False)
