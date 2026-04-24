
from ase import Atoms
from phonopy.structure.atoms import PhonopyAtoms


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