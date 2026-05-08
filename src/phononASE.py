
from ase import Atoms
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.phonon.band_structure import get_band_qpoints_and_path_connections


def ase_to_phonopy(atoms_ase):
    """Function to convert ASE Atoms object to PhonopyAtoms object.
    Parameters:
    - atoms_ase: ASE Atoms object representing the structure.
    Returns:
    - PhonopyAtoms object with the same structure as the input ASE Atoms.
    """
    # Convert ASE Atoms to PhonopyAtoms
    return PhonopyAtoms(cell=atoms_ase.get_cell(),
                        positions=atoms_ase.get_positions(),
                        symbols=atoms_ase.get_chemical_symbols())


def phonopy_to_ase(atoms_phonopy, bulk=True):
    """Function to convert PhonopyAtoms object to ASE Atoms object.
    Parameters:
    - atoms_phonopy: PhonopyAtoms object representing the structure.
    - bulk: Boolean indicating whether to use the bulk structure.
    Returns:
    - ASE Atoms object with the same structure as the input PhonopyAtoms.
    """
    # Set periodic boundary conditions (pbc) based on whether the structure is bulk or slab
    if bulk:
        pbc = True
    else:
        pbc = (True, True, False)
    # Convert PhonopyAtoms to ASE Atoms
    return Atoms(cell=atoms_phonopy.cell,
                 positions=atoms_phonopy.positions,
                 symbols=atoms_phonopy.symbols,
                 pbc=pbc)


def phonon_to_atoms(phonon, cell='unit'):
    """Function to convert a Phonopy object to an ASE Atoms object.
    Parameters:
    - phonon: Phonopy object representing the structure.
    - cell: String indicating whether to use 'unit' or 'super' cell.
    Returns:
    - ASE Atoms object with the same structure as the input Phonopy object.
    """
    # Determine which cell to use based on the input parameter
    if cell == 'unit':
        cell = phonon.unitcell
    elif cell == 'super':
        cell = phonon.supercell
    else:
        raise ValueError("Cell must be 'unit' or 'super'")
    # Check if the supercell is a slab
    if phonon.supercell_matrix.diagonal()[-1] == 1:
        bulk = False
    else:
        bulk = True
    # Convert PhonopyAtoms to ASE Atoms
    return phonopy_to_ase(cell, bulk)


def is_phonon_bulk(phonon):
    """Function to determine if the phonon calculation is for a bulk or slab system based on the supercell matrix.
    Parameters:
        - phonon: Phonopy object containing phonon data.
    Returns:
        - bulk: Boolean indicating if the system is bulk (True) or slab (False).
    """
    if phonon.supercell_matrix.diagonal()[-1] == 1:
        return False
    else:
        return True


def write_dispersion_yaml(phonon, filename):

    if is_phonon_bulk(phonon):
        path = [[[0.0, 0.0, 0.0],[0.5, 0.0, 0.0],[0.5, 0.5, 0.5],
                 [0.5, 0.5, 0.0],[0.0, 0.0, 0.0]]]
        labels = ["$\\Gamma$", "X", "R", "M", "$\\Gamma$"]
    else:
        path = [[[0.0, 0.0, 0.0],[0.5, 0.0, 0.0],
                 [0.5, 0.5, 0.0],[0.0, 0.0, 0.0]]]
        labels = ["$\\Gamma$", "X", "M", "$\\Gamma$"]

    qpoints, connections = get_band_qpoints_and_path_connections(path, npoints=100)

    # --- Band structure ---
    phonon.run_band_structure(
        qpoints,
        path_connections=connections,
        labels=labels,
        with_eigenvectors=True
    )

    phonon.write_yaml_band_structure(filename=filename)