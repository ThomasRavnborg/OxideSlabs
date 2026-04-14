import os
import shutil
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
        self.run_dir = run_dir
        #self.itr_dir = os.path.join(run_dir, "iteration1")

        xyz_file = os.path.join(self.run_dir, 'dataset.xyz')

        if os.path.exists(xyz_file):
            self.data = read(xyz_file, ':')
            print(f"Sucessfully loaded {len(self.data)} structures from dataset.xyz")
            count = sum(1 for atoms in self.data if atoms.calc is None)
            if count > 0:
                if count == len(self.data):
                    print("Warning! No structures have calculator results.")
                    print("DFT calculations must be run, or the dataset will be empty.")
                elif count == 1:
                    print("Warning! 1 structure has no calculator results.")
                    print("DFT calculations must be run, or this structure will be omitted.")
                else:
                    print(f"Warning! {count} structures have no calculator results.")
                    print("DFT calculations must be run, or these will be omitted.")
        else:
            print("No existing dataset.xyz file in specified directory.")
            print("Data set preparation must be run to generate initial data for training.")
            self.data = None

    def prepare_dataset(self, overwrite=False,
                        n_rattle=100, temperatures=[300, 500, 700],
                        n_strain=10, strains=[-1.0, -0.5, 0.5, 1.0]):
        
        if self.data is not None and not overwrite:
            print("Existing data found. Data preperation skipped.")
            print("To overwrite existing data, set overwrite=True.")
            return

        yaml_files = [f for f in os.listdir(self.run_dir) if f.endswith(".yaml")]

        if not yaml_files:
            raise RuntimeError("No phonopy (.yaml) files found in specified directory.")

        else:
            data = []
            for yaml_file in yaml_files:
                phonon = ph.load(os.path.join(self.run_dir, yaml_file))
        
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
            write(os.path.join(self.run_dir, "dataset.xyz"), data)
        
    def run_DFT(self):

        try:
            with open(os.path.join(self.run_dir, 'dft_params.json'), 'r') as f:
                dft_params = json.load(f)

        except FileNotFoundError:
            print("DFT parameters file not found.")
            return

        for i in range(len(self.data)):
            struct = self.data[i]
            if struct.calc is None:
                run_siesta(struct, **dft_params, dir=os.path.join(self.run_dir, 'siesta'))
                self.data[i] = copy_calc_results(struct)
            
            write(os.path.join(self.run_dir, "dataset.xyz"), self.data)

    def setup_nep(self, parameters_nep):
        # Check if data is available and has calculator results before setting up NEP training
        if self.data is None:
            print("No data available to setup NEP training. Please prepare the dataset first.")
            return
        # Attempt to read energies from energies.json, which should have been generated by the DFT calculations
        try:
            with open(os.path.join(self.run_dir, 'energies.json'), 'r') as f:
                energies = json.load(f)
        except FileNotFoundError:
            energies = None
            print("Atomic energies file (energies.json) not found in directory.")
            print("It is highly recommended to have the atomic energies of the constituent elements for better NEP training.")

        # Take out only the structures with calculator results for NEP training
        data = [copy_calc_results(atoms) for atoms in self.data if atoms.calc is not None]

        # Check if any structures have calculator results before proceeding with NEP training setup
        if len(data) == 0:
            print("No structures have calculator results. NEP training cannot be setup.")
            return

        # Warn if some structures do not have calculator results and will be omitted from NEP training
        if len(data) < len(self.data):
            print(f"Warning: {len(self.data) - len(data)} structures have no calculator results and will be omitted from NEP training.")

        # Shift the energies of the structures by the sum of the energies of the constituent atoms, if energies are available
        unique_elements = set()
        for atoms in data:
            elements = atoms.get_chemical_symbols()
            unique_elements.update(elements)
            if energies is not None:
                atoms.calc.results['energy'] -= sum(energies[element] for element in elements)

        # Set up the input files for NEP training
        params = dict(version=4, type=[len(unique_elements), ' '.join(unique_elements)])
        params.update(parameters_nep)

        # Set up training and testing data for NEP training using bagging method
        # This randomly splits the data into training and testing sets according to the specified train_fraction
        setup_training(params, data,
                       rootdir=self.run_dir, overwrite=True,
                       mode='bagging', train_fraction=0.9, n_splits=1)
        print(f"NEP training setup complete. {len(data)} structures selected for training/testing.")

    def train_nep(self):
        train_dir = os.path.join(self.run_dir, "nepmodel_split1")
        # Still not done!
        subprocess.run(["nep"], cwd=train_dir,
                       check=True, text=True)

    def setup_MD(self, temperature=300, n_steps=5000, dump_interval=10):
        md_dir = os.path.join(self.run_dir, "md")
        os.makedirs(md_dir, exist_ok=True)

        # Copy the trained NEP model to the MD directory
        nep_src = os.path.join(self.run_dir, "nepmodel_split1", "nep.txt")
        nep_dst = os.path.join(md_dir, "nep.txt")

        if not os.path.exists(nep_src):
            raise RuntimeError("nep.txt not found. Train NEP first.")

        shutil.copy(nep_src, nep_dst)
        
        # Choose a structure from the dataset to run MD on
        if self.data is None or len(self.data) == 0:
            raise RuntimeError("No structures available for MD")
        # For now, just take the first structure with calculator results.
        atoms = self.data[0]  # simple choice for now

        model_path = os.path.join(md_dir, "model.xyz")

        write(model_path, atoms)

        # =========================
        # 3. Write GPUMD input
        # =========================
        run_in = f"""\
        potential nep.txt

        velocity {temperature}
        time_step 1.0

        ensemble nvt {temperature} {temperature} 100

        dump_position {dump_interval}
        dump_extrapolation {dump_interval}

        run {n_steps}
        """

        with open(os.path.join(md_dir, "run.in"), "w") as f:
            f.write(run_in)


    def run_MD(self):
        md_dir = os.path.join(self.run_dir, "md")
        print("Running GPUMD...")

        subprocess.run(
            ["gpumd"],
            cwd=md_dir,
            check=True
        )

        print("MD finished. Outputs:")
        print("- dump.xyz (structures)")
        print("- extrapolation.dat (gamma)")