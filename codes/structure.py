
from ase import Atoms


class Perovskite:
    """Class to create a perovskite structure.
    Parameters:
    - formula: Chemical formula of the perovskite ('ABX3').
    - a: Lattice constant in Angstroms.
    Attributes:
    - atoms: ASE Atoms object representing the perovskite structure.
    """
    def __init__(self, formula='ABX3', a=4.0):
        self.formula = formula
        self.a = a
        sca_pos = [[0, 0, 0],
                   [1/2, 1/2, 1/2],
                   [1/2, 0, 1/2],
                   [0, 1/2, 1/2],
                   [1/2, 1/2, 0]]
        def unitCell(a):
            return [[a, 0, 0], [0, a, 0], [0, 0, a]]
        self.atoms = Atoms(self.formula, cell=unitCell(self.a),
                           pbc=True, scaled_positions=sca_pos)