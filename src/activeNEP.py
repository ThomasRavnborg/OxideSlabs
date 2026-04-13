import os
import json
import subprocess
import numpy as np
import phonopy as ph
from ase.io import read, write
from calorine.nep import setup_training
from hiphive.structure_generation import generate_phonon_rattled_structures
from src.phononcalc import phonon_to_atoms, phonopy_to_ase
from src.frozenphonon import copy_calc_results
from src.calculators import run_siesta

class ActiveLearningNEP:

    def __init__(self, run_dir):
        self.dir = run_dir

        xyz_file = os.path.join(self.dir, 'dataset.xyz')

        if os.path.exists(xyz_file):
            self.data = read(xyz_file, ':')
            print(f"Loaded {len(self.data)} structures from dataset.xyz")
        else:
            print("No existing dataset.xyz file in specified directory.")
            print("Data set preparation must be run to generate initial data for training.")
            self.data = None

    def prepare_dataset(self, overwrite=False,
                        n_rattle=100, temperatures=[300, 500, 700],
                        n_strain=10, strains=[-1.0, -0.5, 0.5, 1.0]):
        
        if self.data is not None and not overwrite:
            print("Existing data found. Data preperation skipped.")
            return

        yaml_files = [f for f in os.listdir(self.dir) if f.endswith(".yaml")]

        if not yaml_files:
            raise RuntimeError("No phonopy (.yaml) files found in specified directory.")

        else:
            data = []
            for yaml_file in yaml_files:
                phonon = ph.load(os.path.join(self.dir, yaml_file))
        
                # Produce the force constants
                phonon.produce_force_constants()
                # Get the force constants matrix
                fc = phonon.force_constants

                atoms = phonon_to_atoms(phonon, cell='super')

                # Convert the phonon object to an ASE Atoms object
                structures = [atoms]

                # Extract all displaced structures and convert them to ASE Atoms objects
                for atoms_phonopy in phonon.supercells_with_displacements:
                    # Convert the phonopy Atoms object to an ASE Atoms object and append
                    displaced_structures = phonopy_to_ase(atoms_phonopy, bulk=True)
                    structures.append(displaced_structures)
                
                print(f"Extracted {len(structures)} structures from phonon calculation (including original and displaced)")
                
                def _get_rattled_structures(atoms, fc, T, n_structures):
                    # Produce phonon rattled structures using the force constants and the original structure
                    rattled_structures = generate_phonon_rattled_structures(atoms, fc2=fc, temperature=T,
                                                                            n_structures=n_structures)
                    return rattled_structures

                for T in temperatures:
                    structures.extend(_get_rattled_structures(atoms, fc, T, n_rattle))
                
                print(f"Generated {n_rattle*len(temperatures)} rattled structures")

                def _get_strained_structures(atoms, strain):
                    strain *= 0.01  # Convert percentage to a decimal
                    
                    strained_structures = []
                    # Pick random component of the cell matrix to strain
                    for _ in range(n_strain):
                        atoms_strain = atoms.copy()
                        cell = atoms.get_cell()
                        i = np.random.randint(0, 3)
                        j = np.random.randint(0, 3)
                        cell[i, j] *= (1 + strain)
                        atoms_strain.set_cell(cell, scale_atoms=True)
                        strained_structures.append(atoms_strain)
                    return strained_structures

                for strain in strains:
                    structures.extend(_get_strained_structures(atoms, strain))

                print(f"Generated {n_strain*len(strains)} strained structures")

                print(f"{len(structures)} total structures")
                data.extend(structures)
            
            self.data = data
            write(os.path.join(self.dir, "dataset.xyz"), data)
        
    def run_DFT(self):

        try:
            with open(os.path.join(self.dir, 'dft_params.json'), 'r') as f:
                dft_params = json.load(f)

        except FileNotFoundError:
            print("DFT parameters file not found.")
            return

        for i in range(len(self.data)):
            struct = self.data[i]
            if struct.calc is None:
                run_siesta(struct, **dft_params, dir=os.path.join(self.dir, 'siesta'))
                self.data[i] = copy_calc_results(struct)
            
            write(os.path.join(self.dir, "dataset.xyz"), self.data)

    def setup_nep(self, parameters_nep):
        if self.data is None:
            print("No data available to setup NEP training. Please prepare the dataset first.")
            return
        else:
            elements = set(self.data[0].symbols)
            params = dict(version=4, type=[len(elements), ' '.join(elements)])
            params.update(parameters_nep)

        setup_training(params, self.data,
                       rootdir=self.dir, overwrite=True,
                       mode='bagging', train_fraction=0.9, n_splits=1)

    def train_nep(self):

        subprocess.run(["nep"], cwd=os.path.join(self.dir, "nepmodel_split1"),
                       check=True, text=True)

