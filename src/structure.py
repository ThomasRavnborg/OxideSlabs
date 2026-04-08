import math
import numpy as np
import re
from ase import Atoms
from ase.build import make_supercell

class Perovskite:
    """Class to create a perovskite structure.
    Parameters:
    - formula: Chemical formula of the perovskite ('ABX3').
    - a: Lattice constant in Angstroms.
    - N: Number of unit cells in each direction for bulk (int) or thickness of the slab in unit cells (float).
    - bulk: Boolean indicating whether to create a bulk structure (True) or a slab structure (False).
    - dvac: Vacuum spacing in the z-direction for slab structures (default is 20.0).
    Attributes:
    - atoms: ASE Atoms object representing the perovskite structure.
    """
    def __init__(self, formula='ABX3', N=1, bulk=True, dvac=20.0):
        self.formula = formula
        self.symbols = [formula[0:2], formula[2:4], formula[4]]
        self.ncells = N
        self.bulk = bulk

        lats = {'BaTiO3': 3.98,
                'SrTiO3': 3.90}
        a = lats.get(formula, 3.9)  # Default to 3.9 if formula not in dictionary

        sca_pos = [[0, 0, 0],
                   [1/2, 1/2, 1/2],
                   [1/2, 0, 1/2],
                   [0, 1/2, 1/2],
                   [1/2, 1/2, 0]]
        def unitCell(a):
            return [[a, 0, 0], [0, a, 0], [0, 0, a]]
        
        # Create an atoms object for the unit cell of the perovskite structure
        atoms_ucell = Atoms(self.formula, cell=unitCell(a), scaled_positions=sca_pos)

        if bulk:
            atoms_ucell.pbc = (True, True, True)
            self.atoms = atoms_ucell * (N, N, N)
        else:
            # Create an asymmetric (if N is an int) or symmetric (if N is a float) slab
            if isinstance(N, int):
                # Create asymetric slab supercell by copying the unit cell
                P = np.diag([1, 1, N])
                slab = make_supercell(atoms_ucell, P)
            elif isinstance(N, float):
                # Round up from the float value
                n = math.ceil(N)
                # Create asymetric slab supercell by copying the unit cell
                P = np.diag([1, 1, n])
                slab = make_supercell(atoms_ucell, P)
                # Remove A and X atom at the end to make the slab symmetric
                for i in sorted([0, 4], reverse=True):
                    del slab[i]
            slab.pbc = (True, True, False)
            # Center the slab in the cell and add vacuum in the z-direction
            slab.center(axis=2, vacuum=dvac)
            self.atoms = slab
    def __repr__(self):
        return f'Perovskite(formula={self.formula}, cell={self.atoms.cell}, positions={self.atoms.positions.tolist()}, pbc={self.atoms.pbc})'
    
    def set_atoms(self, atoms):
        """Set the atoms object for the perovskite structure."""
        self.atoms = atoms

    def apply_strain(self, strain):
        """Apply a specified strain to the perovskite structure.
        Parameters:
        - strain: Strain value to be applied (e.g., 0.01 for 1% tensile strain, -0.01 for 1% compressive strain).
        """
        # Get the current cell parameters
        cell = self.atoms.cell.copy()
        # Apply the specified strain to the in-plane lattice parameters (a and b)
        cell[0, 0] *= (1 + strain)  # Strain applied to a
        cell[1, 1] *= (1 + strain)  # Strain applied to b
        # Update the cell parameters of the atoms object
        self.atoms.set_cell(cell, scale_atoms=True)


def get_reduced_formula(ase_atoms):
    """Function to extract the reduced formula in the form of ABX3 from an ASE Atoms object.
    Args:
        ase_atoms (ase.Atoms): An ASE Atoms object representing the structure.
    Returns:
        str: The reduced formula in the form of ABX3 if found, otherwise the full formula."""
    # Get the full chemical formula from the ASE Atoms object
    full_formula = ase_atoms.get_chemical_formula(mode='reduce')
    # Use a regular expression to find the pattern of the form ABX3
    pattern = r'([A-Z][a-z]?)([A-Z][a-z]?)([A-Z][a-z]?)3'
    match = re.search(pattern, full_formula)
    # If a match is found, return the matched pattern; otherwise, return the full formula
    if match:
        formula = match[0]
    else:
        #print("No match found for the ABX3 pattern in the formula.")
        formula = full_formula
    
    return formula