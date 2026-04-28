import math
import numpy as np
import re
from ase import Atoms
from ase.build import make_supercell

class Perovskite:
    """Class to create a perovskite structure.
    
    Parameters:
    - formula: Chemical formula of the perovskite ('ABX3').
    - bulk: Boolean indicating whether to create a bulk structure (True) or a slab structure (False).
    - dslab: Thickness of the slab in unit cells (default is 1). For bulk structures, this parameter is ignored.
    - dvac: Vacuum spacing in the z-direction for slab structures in Å (default is 20.0).
    
    Attributes:
    - atoms: ASE Atoms object representing the perovskite structure.
    """

    def __init__(self, formula='ABX3', bulk=True, dslab=1, dvac=20.0):
        self.formula = formula
        #self.symbols = [formula[0:2], formula[2:4], formula[4]]
        self.ncells = dslab if not bulk else 1
        #self.bulk = bulk

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
            self.atoms = atoms_ucell
        else:
            # Create an asymmetric (if dslab is an int) or symmetric (if dslab is a float) slab
            if isinstance(dslab, int):
                # Create asymetric slab supercell by copying the unit cell
                P = np.diag([1, 1, dslab])
                slab = make_supercell(atoms_ucell, P)
            elif isinstance(dslab, float):
                # Round up from the float value
                n = math.ceil(dslab)
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
    formula = ase_atoms.get_chemical_formula(mode='reduce')
    
    """
    # Use a regular expression to find the pattern of the form ABX3 or A4B4X12 in the full formula
    pattern1 = r'([A-Z][a-z]?)([A-Z][a-z]?)([A-Z][a-z]?)3'
    pattern2 = r'([A-Z][a-z]?)4([A-Z][a-z]?)4([A-Z][a-z]?)12'
    match = re.search(pattern1, full_formula)
    if not match:
        match = re.search(pattern2, full_formula)
    # If a match is found, return the matched pattern; otherwise, return the full formula
    if match:
        formula = match[0]
    else:
        #print("No match found for the ABX3 pattern in the formula.")
        formula = full_formula
    
    """

    return formula


def check_if_bulk(atoms):
    pbc = atoms.get_pbc()
    if not all(pbc):
        return False
    return True


def wrap_to_reference(unwrapped_traj, ref_atoms):
    wrapped_traj = []

    cell = ref_atoms.get_cell()

    # Reference fractional positions
    ref_scaled = ref_atoms.get_scaled_positions()

    for atoms in unwrapped_traj:
        new_atoms = atoms.copy()

        # Convert to fractional coordinates
        scaled = np.linalg.solve(cell.T, atoms.get_positions().T).T

        # Compute displacement relative to reference
        delta = scaled - ref_scaled

        # Wrap relative displacement into [-0.5, 0.5]
        delta -= np.floor(delta + 0.5)

        # Reconstruct wrapped positions
        new_scaled = ref_scaled + delta

        new_atoms.set_scaled_positions(new_scaled)
        wrapped_traj.append(new_atoms)

    return wrapped_traj



def kspacing_from_kgrid(atoms, kgrid):
    cell = atoms.get_cell()
    rec = 2 * np.pi * np.linalg.inv(cell).T
    b_lengths = np.linalg.norm(rec, axis=1)

    kgrid = np.array(kgrid)
    kspacing = b_lengths / kgrid

    return sum(kspacing) / len(kspacing)

def kgrid_from_kspacing(atoms, kspacing):
    cell = atoms.get_cell()
    rec = 2 * np.pi * np.linalg.inv(cell).T
    b_lengths = np.linalg.norm(rec, axis=1)
    kpts = np.maximum(1, np.round(b_lengths / kspacing).astype(int))
    return kpts