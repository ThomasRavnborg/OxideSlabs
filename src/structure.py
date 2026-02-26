import math
import numpy as np
from ase import Atoms
from ase.build import make_supercell

class Perovskite:
    """Class to create a perovskite structure.
    Parameters:
    - formula: Chemical formula of the perovskite ('ABX3').
    - a: Lattice constant in Angstroms.
    - N: Number of unit cells in each direction for bulk (int) or thickness of the slab in unit cells (float).
    - bulk: Boolean indicating whether to create a bulk structure (True) or a slab structure (False).
    Attributes:
    - atoms: ASE Atoms object representing the perovskite structure.
    """
    def __init__(self, formula='ABX3', a=4.0, N=1, bulk=True):
        self.formula = formula
        self.bulk = bulk
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
            self.atoms = slab
    def __repr__(self):
        return f'Perovskite(symbols={self.atoms.symbols}, cell={self.atoms.cell}, positions={self.atoms.positions.tolist()}, pbc={self.atoms.pbc})'
    
    def set_atoms(self, atoms):
        """Set the atoms object for the perovskite structure."""
        self.atoms = atoms