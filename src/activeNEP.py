import os
import shutil
import json
import subprocess
import numpy as np
import phonopy as ph
from ase.io import read, write
from calorine.nep import setup_training, get_descriptors
#from calorine.nep import get_descriptors
from hiphive.structure_generation import generate_phonon_rattled_structures
from src.phononcalc import phonon_to_atoms, phonopy_to_ase
from src.frozenphonon import copy_calc_results
from src.fdfcreate import generate_basis
from src.calculators import run_siesta

class ActiveLearningNEP:

    def __init__(self, run_dir, iteration=1, overwrite=False):

        # Define run directory and iteration number for active learning loop
        self.run_dir = run_dir
        self.iteration = iteration

        # Create iteration folder
        self.iter_dir = os.path.join(run_dir, f"iteration_{iteration}")
        os.makedirs(self.iter_dir, exist_ok=True)

        # Attempt to load existing training and test datasets from the run directory, if they exist.
        try:
            self.train_data = read(os.path.join(self.run_dir, "train.xyz"), ":")
            self.test_data = read(os.path.join(self.run_dir, "test.xyz"), ":")
            print(f"Loaded {len(self.train_data)} training structures and {len(self.test_data)} test structures", flush=True)
            
            self.data = self.train_data + self.test_data
            self.count = len([s for s in self.data if s.calc is None])

            if self.count > 0:
                if self.count == len(self.data):
                    print("Warning! No structures have calculator results.")
                    print("DFT calculations must be runwith run_DFT(), or the dataset will be empty.", flush=True)
                elif self.count == len(self.data) - 1:
                    print("Warning! 1 structure has no calculator results.")
                    print("DFT calculations must be runwith run_DFT(), or this structure will be omitted.", flush=True)
                else:
                    print(f"Warning! {self.count}/{len(self.data)} structures have no calculator results.")
                    print("DFT calculations must be run with run_DFT(), or these will be omitted.", flush=True)
        
        except Exception:
            self.train_data = None
            self.test_data = None
            self.data = None
            print("No train.xyz and test.xyz files found.")
            print("Use prepare_dataset() to create the datasets based on .yaml files.", flush=True)

    def _get_rattled_structures(self, atoms, fc, T, n_structures):
        # Produce phonon rattled structures using the force constants and the original structure
        rattled_structures = generate_phonon_rattled_structures(atoms, fc2=fc, temperature=T,
                                                                n_structures=n_structures)
        return rattled_structures

    def _strain_structure(self, atoms, strain):
        strain *= 0.01  # Convert percentage to a decimal

        # Get the current cell parameters
        cell = atoms.cell.copy()
        # Randomly pick a lattice parameter (a, b, or c) to apply the strain to
        i = np.random.randint(0, 3)
        # Apply the specified strain to the in-plane lattice parameters (a and b)
        cell[i, i] *= (1 + strain)  # Strain applied to a
        # Update the cell parameters of the atoms object
        atoms.set_cell(cell, scale_atoms=True)

    # 0. Prepare dataset for NEP training

    def prepare_dataset(self, N_structures=500, temperatures = [300, 500, 700],
                        frac_strain=0.1, strains=[-1.0, -0.5, 0.5, 1.0],
                        train_fraction=0.9, overwrite=False):

        if self.data is not None and not overwrite:
            print("Existing data found. Data preperation skipped.")
            print("To overwrite existing data, set overwrite=True.", flush=True)
            return

        yaml_files = [f for f in os.listdir(self.run_dir) if f.endswith(".yaml")]
        n_rattle = N_structures//(len(yaml_files)*len(temperatures))

        if not yaml_files:
            raise RuntimeError("No phonopy (.yaml) files found in specified directory.")

        else:
            train_data = []
            test_data = []

            for yaml_file in yaml_files:

                structures = []
                augmented_structures = []

                phonon = ph.load(os.path.join(self.run_dir, yaml_file))
        
                # Produce the force constants
                phonon.produce_force_constants()
                # Get the force constants matrix
                fc = phonon.force_constants

                # Convert the phonon object to an ASE Atoms object
                atoms = phonon_to_atoms(phonon, cell='super')
                structures.append(atoms)

                # Extract all displaced structures and convert them to ASE Atoms objects
                for atoms_phonopy in phonon.supercells_with_displacements:
                    # Convert the phonopy Atoms object to an ASE Atoms object and append
                    displaced_structures = phonopy_to_ase(atoms_phonopy, bulk=True)
                    structures.append(displaced_structures)
            
                #print(f"Extracted {len(structures)} structures from phonon calculation (including original and displaced)", flush=True)

                for T in temperatures:
                    augmented_structures.extend(self._get_rattled_structures(atoms, fc, T, n_rattle))
                
                #print(f"Generated {n_rattle*len(temperatures)} rattled structures", flush=True)
                
                # Randomly pick out a fraction of the augmented structures to apply random strains to
                n_strain = int(len(augmented_structures) * frac_strain)
                idx = np.random.choice(len(augmented_structures), size=n_strain, replace=False)

                for i in idx:
                    strain = np.random.choice(strains)
                    self._strain_structure(augmented_structures[i], strain)

                #print(f"Strained {n_strain*len(strains)} of the rattled structures", flush=True)

                #print(f"{len(structures)} total structures", flush=True)
                train_data.extend(structures)

                n_train = int(len(augmented_structures) * train_fraction)
                np.random.shuffle(augmented_structures)

                train_data.extend(augmented_structures[:n_train])
                test_data.extend(augmented_structures[n_train:])

                #print(f"{len(data)} structures in the dataset", flush=True)
            
            print(f"Total dataset prepared: {len(train_data) + len(test_data)} structures")
            print(f"Training set: {len(train_data)} structures, Test set: {len(test_data)} structures", flush=True)

            write(os.path.join(self.run_dir, "train.xyz"), train_data)
            write(os.path.join(self.run_dir, "test.xyz"), test_data)

            self.train_data = train_data
            self.test_data = test_data
    

    # 1. Run DFT calculations

    def run_DFT(self, which='all'):

        try:
            with open(os.path.join(self.run_dir, 'dft_params.json'), 'r') as f:
                dft_params = json.load(f)

        except FileNotFoundError:
            print("DFT parameters file not found.", flush=True)
            return

        if self.data is None:
            print("No data available to run DFT. Please prepare the dataset first.", flush=True)
            return
        
        #self.count = len([s for s in self.data if s.calc is None])
        print(f"Running DFT on {self.count} structures without calculator assigned...", flush=True)
        
        def _label_DFT(data, label='train'):
            for i in range(len(data)):
                struct = data[i]
                if struct.calc is None:
                    #generate_basis(struct, dir=os.path.join(self.run_dir, 'DFT'))
                    run_siesta(struct, **dft_params, dir=os.path.join(self.run_dir, 'DFT'))
                data[i] = copy_calc_results(struct)
                write(os.path.join(self.run_dir, f"{label}.xyz"), data)

        if which in ('train', 'all'):
            _label_DFT(self.train_data, label='train')
        if which in ('test', 'all'):
            _label_DFT(self.test_data, label='test')

        #write(os.path.join(self.run_dir, "train.xyz"), self.train_data)
        #write(os.path.join(self.run_dir, "test.xyz"), self.test_data)

        self.count = 0

    # 2. Set up NEP training

    def setup_nep(self, parameters_nep):
        # Check if data is available and has calculator results before setting up NEP training
        if self.data is None:
            print("No data available to setup NEP training. Please prepare the dataset first.", flush=True)
            return
        # Attempt to read energies from energies.json, which should have been generated by the DFT calculations
        try:
            with open(os.path.join(self.run_dir, 'energies.json'), 'r') as f:
                energies = json.load(f)
        except FileNotFoundError:
            energies = None
            print("Atomic energies file (energies.json) not found in directory.", flush=True)
            print("It is highly recommended to have the atomic energies of the constituent elements for better NEP training.", flush=True)

        # Take out only the structures with calculator results for NEP training
        data = [copy_calc_results(atoms) for atoms in self.data if atoms.calc is not None]

        # Check if any structures have calculator results before proceeding with NEP training setup
        if len(data) == 0:
            print("No structures have calculator results. NEP training cannot be setup.", flush=True)
            return

        # Warn if some structures do not have calculator results and will be omitted from NEP training
        if len(data) < len(self.data):
            print(f"Warning: {len(self.data) - len(data)} structures have no calculator results and will be omitted from NEP training.", flush=True)

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
                       rootdir=self.iter_dir, overwrite=True,
                       mode='bagging', train_fraction=0.9, n_splits=1)
        print(f"NEP training setup complete. {len(data)} structures selected for training/testing.", flush=True)

        #self.train_data = read(os.path.join(self.iter_dir, "nepmodel_split1", "train.xyz"), ":")
        #self.test_data = read(os.path.join(self.iter_dir, "nepmodel_split1", "test.xyz"), ":")
    

    def setup_nep2(self, parameters_nep):
        # Check if data is available and has calculator results before setting up NEP training
        
        try:
            # Copy train.xyz and test.xyz to the iteration directory for NEP training
            write(os.path.join(self.iter_dir, "train.xyz"), self.train_data)
            write(os.path.join(self.iter_dir, "test.xyz"), self.test_data)
        
        except Exception:
            print(f"Missing train.xyz and/or test.xyz. Cannot setup NEP training.", flush=True)
            return

        if self.count > 0:
            print(f"Warning! {self.count} structures have no calculator results.", flush=True)
            #return
        
        self.unique_elements = set()
        
        def _shift_energies(data):
            # Attempt to read energies from energies.json, which should have been generated by the DFT calculations
            try:
                with open(os.path.join(self.run_dir, 'energies.json'), 'r') as f:
                    energies = json.load(f)

            except FileNotFoundError:
                energies = None
                print("Atomic energies file (energies.json) not found in directory.")
                print("It is highly recommended to have the atomic energies of the constituent elements for better NEP training.", flush=True)

            energies = None # temporary fix

            # Shift the energies of the structures by the sum of the energies of the constituent atoms, if energies are available
            for atoms in data:
                elements = atoms.get_chemical_symbols()
                self.unique_elements.update(elements)
                if energies is not None:
                    atoms.calc.results['energy'] -= sum(energies[element] for element in elements)

        _shift_energies(self.train_data)
        _shift_energies(self.test_data)

        # Set up the input files for NEP training
        params = dict(version=4, type=[len(self.unique_elements), ' '.join(self.unique_elements)])
        params.update(parameters_nep)

        # Dump params to a .in file
        with open(os.path.join(self.iter_dir, "nep.in"), "w") as f:
            for key, value in params.items():
                if isinstance(value, list):
                    value_str = ' '.join(map(str, value))
                else:
                    value_str = str(value)
                f.write(f"{key}  {value_str}\n")



    def train_nep(self):
        train_dir = os.path.join(self.iter_dir, "nepmodel_split1")
        subprocess.run(["nep"], cwd=train_dir,
                       check=True, text=True)
    

    def _set_prediction_mode(self, nep_in, dataset=None):
        with open(nep_in, "r") as f:
            lines = f.readlines()

        lines = [l for l in lines if not l.strip().startswith(("prediction", "dataset"))]

        lines.append("prediction 1\n")          # Set prediction mode
        lines.append("output_descriptor 2\n")   # Output per-atom descriptors
        if dataset is not None:
            lines.append(f"dataset {dataset}\n")

        with open(nep_in, "w") as f:
            f.writelines(lines)


    def run_prediction_mode(self):
        nep_dir = os.path.join(self.iter_dir, "nepmodel_split1")
        nep_in = os.path.join(nep_dir, "nep.in")

        self._set_prediction_mode(nep_in)

        print("Running NEP in prediction mode...", flush=True)
        subprocess.run(["nep"], cwd=nep_dir,
                       check=True, text=True)


    def _extract_descriptors(self, desc_file):

        A = np.loadtxt(desc_file)

        A = (A - A.mean(axis=0)) / (A.std(axis=0) + 1e-12)

        print(f"Loaded descriptor matrix: {A.shape}", flush=True)

        return A

    def _calculate_descriptors(self, structures):
        A = []
        for structure in structures:
            A.append(get_descriptors(structure, os.path.join(self.iter_dir, "nep.txt")))
        A = np.vstack(A)

        A = (A - A.mean(axis=0)) / (A.std(axis=0) + 1e-12)

        print(f"Computed descriptor matrix: {A.shape}", flush=True)

        return A


    def compute_descriptors(self, get_out=True, write_out=True):
        
        desc_file = os.path.join(self.iter_dir, "descriptor.out")
        nep_file = os.path.join(self.iter_dir, "nep.txt")

        if os.path.exists(desc_file) and get_out:
            A = self._extract_descriptors(desc_file)

        elif os.path.exists(nep_file):
            print("Descriptor file not found, but nep.txt exists. Computing descriptors with Calorine.", flush=True)
            A = self._calculate_descriptors(self.train_data)
            # Save to descriptor.out for future use
            if write_out:
                np.savetxt(os.path.join(self.iter_dir, "descriptor.out"), A, fmt="%.6e")

        else:
            raise RuntimeError("Neither descriptor.out nor nep.txt found. Cannot compute descriptors.")

        self.descriptors = A


    def _extract_active_set(self, asi_file):

        from src.asiIO import load_asi
        def find_inverse(m):
            return np.linalg.pinv(m, rcond=1e-8)

        A_inv = load_asi(asi_file)
        A_active = find_inverse(A_inv)

        return A_active

    def _calculate_active_set(self, structures, batch_size=None):

        from src.MaxVol import calculate_maxvol

        struct_index = []

        for i, atoms in enumerate(structures):
            n = len(atoms)
            struct_index.extend([i] * n)
        struct_index = np.array(struct_index)

        print("Performing MaxVol...")

        descriptors = self._calculate_descriptors(structures)

        A_active, active_index = calculate_maxvol(
            descriptors, struct_index, batch_size=batch_size
        )
        
        #A_inv = find_inverse(A_active)
        
        return A_active, active_index

    
    def build_active_set(self, batch_size=None, get_asi=True, write_asi=True):
        """
        Wrapper for MaxVol active set selection.

        Args:
            B (np.ndarray): descriptor matrix (N, M)
            batch_size (int or None)

        Returns:
            active_set_struct (np.ndarray): selected descriptor matrix (M, M-ish)
            active_struct_index (np.ndarray): indices of selected environments
            active_rows (np.ndarray): row indices in B
        """
        """
        from src.MaxVol import calculate_maxvol
        from src.asiIO import load_asi, save_asi

        def find_inverse(m):
            return np.linalg.pinv(m, rcond=1e-8)
        """

        from src.asiIO import save_asi

        def find_inverse(m):
            return np.linalg.pinv(m, rcond=1e-8)

        asi_file = os.path.join(self.iter_dir, "active_set.asi")
        if os.path.exists(asi_file) and get_asi:
            print("Existing active set inverse found. Loading...")
            A_active = self._extract_active_set(asi_file)
            self.active = A_active
        else:
            print("Building active set...")
            A_active, active_index = self._calculate_active_set(self.train_data, batch_size=batch_size)
            self.active = A_active
            self.active_index = active_index
            if write_asi:
                save_asi(find_inverse(A_active), asi_file)
                print(f"Active set inverse saved to {asi_file}", flush=True)



    def setup_MD(self, temperature=300, n_steps=10000, dump_interval=100):
        md_dir = os.path.join(self.iter_dir, "md")
        os.makedirs(md_dir, exist_ok=True)

        # Copy the trained NEP model to the MD directory
        nep_src = os.path.join(self.iter_dir, "nep.txt")
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

        run_in = f"""
potential nep.txt

velocity {temperature}
time_step 1.0
dump_exyz {dump_interval} 0 1

compute_extrapolation asi_file ../active_set.asi gamma_low 5 gamma_high 10 check_interval {dump_interval} dump_interval {dump_interval}

ensemble npt_mttk iso 0 0 temp {temperature} {temperature+200}
run {n_steps}

ensemble npt_mttk iso 0 0 temp {temperature+200} {temperature}
run {n_steps}
        """

        with open(os.path.join(md_dir, "run.in"), "w") as f:
            f.write(run_in)


    def run_MD(self):
        md_dir = os.path.join(self.iter_dir, "md")
        print("Running GPUMD...")

        subprocess.run(["gpumd"], cwd=md_dir,
                       check=True, text=True)
    

    def extract_md_descriptors(self):

        md_dir = os.path.join(self.iter_dir, "md")
        nep_dir = os.path.join(self.iter_dir, "nep.txt")

        dump_file = os.path.join(md_dir, "dump.xyz")
        if not os.path.exists(dump_file):
            raise RuntimeError("dump.xyz not found")

        structures = read(dump_file, ":")

        tmp_file = os.path.join(nep_dir, "md_structures.xyz")
        write(tmp_file, structures)

        nep_in = os.path.join(nep_dir, "nep.in")
        self._set_prediction_mode(nep_in, dataset=tmp_file)

        print("Extracting descriptors for MD structures...")
        subprocess.run(["nep"], cwd=nep_dir, check=True)

        desc_file = os.path.join(nep_dir, "descriptor.out")
        A = np.loadtxt(desc_file)

        A = (A - A.mean(axis=0)) / (A.std(axis=0) + 1e-12)

        self.md_descriptors = A

        print(f"MD descriptor matrix: {A.shape}", flush=True)


    def find_candidates(self, new_structures, gamma_th=5.0, n_MaxVol=1):

        gamma_structures = self.calculate_gamma(new_structures)

        # Filter structures with gamma above a certain threshold
        highgamma_structures = [s for s, g in zip(new_structures, gamma_structures) if g > gamma_th]
        
        if len(highgamma_structures) == 0:
            print(f"No structures found with gamma > {gamma_th}", flush=True)
            return []
        
        print(f"Found {len(highgamma_structures)} high-gamma structures with gamma > {gamma_th}")
        
        """
        print(f"Performing {n_MaxVol}-round MaxVol selection on high-gamma structures...", flush=True)
        candidate_structures = []
        remaining_structures = highgamma_structures.copy()

        for _ in range(n_MaxVol):

            # Perform MaxVol on the candidate structures to find the most informative ones
            A_active, active_struc_index = self._calculate_active_set(remaining_structures)
            active_struc_index = np.unique(active_struc_index)
            unique_structures = [remaining_structures[i] for i in active_struc_index]

            # Append the unique structures from this round of MaxVol to the candidate list
            candidate_structures.extend(unique_structures)
            # Remove the selected structures from the high-gamma pool for the next round
            remaining_structures = [s for i, s in enumerate(remaining_structures) if i not in active_struc_index]

        """
        # Perform MaxVol on the candidate structures to find the most informative ones
        A_active, active_struc_index = self._calculate_active_set(highgamma_structures)
        active_struc_index = np.unique(active_struc_index)
        candidate_structures = [highgamma_structures[i] for i in active_struc_index]

        print(f"Found {len(candidate_structures)} candidate structures from MD", flush=True)

        return candidate_structures



    def select_structures(self, gamma_threshold=5.0):

        md_dir = os.path.join(self.iter_dir, "md")
        dump_file = os.path.join(md_dir, "dump.xyz")

        structures = read(dump_file, ":")

        gamma = self.parse_gamma()

        n = min(len(structures), len(gamma))

        selected = [
            structures[i]
            for i in range(n)
            if gamma[i] > gamma_threshold
        ]

        self.candidate_structures = selected

        print(f"Selected {len(selected)} high-gamma structures", flush=True)


    def calculate_gamma(self, structures):
        """
        Compute extrapolation grade (gamma) for all environments.

        Args:
            structures (list): List of structure objects

        Returns:
            gamma (np.ndarray): shape (N,)
        """
        
        B = self._calculate_descriptors(structures)

        def find_inverse(m):
            return np.linalg.pinv(m, rcond=1e-8)
        
        A_inv = find_inverse(self.active)

        # projection matrix
        C = B @ A_inv  # (N, M)

        gamma = np.max(np.abs(C), axis=1)

        struct_index = self._map_env_to_struct(structures)

        gamma_struct = []
        for s in np.unique(struct_index):
            gamma_struct.append(np.max(gamma[struct_index == s]))

        return np.array(gamma_struct)



    def parse_gamma(self):

        md_dir = os.path.join(self.iter_dir, "md")
        gamma_file = os.path.join(md_dir, "extrapolation.dat")

        if not os.path.exists(gamma_file):
            raise RuntimeError("extrapolation.dat not found")

        gamma = np.loadtxt(gamma_file)

        # Case 1: already 1D (per frame)
        if gamma.ndim == 1:
            return gamma

        # Case 2: per-atom → take max per frame
        elif gamma.ndim == 2:
            gamma_frame = gamma.max(axis=1)
            return gamma_frame

        else:
            raise RuntimeError("Unknown gamma format")
    

    def _map_env_to_struct(self, structures):

        counts = [len(s) for s in structures]

        mapping = []
        for i, n in enumerate(counts):
            mapping.extend([i] * n)

        return np.array(mapping)


    def filter_structures(self, n_select=50):

        if not hasattr(self, "candidate_structures"):
            raise RuntimeError("Run select_structures first")

        if len(self.candidate_structures) == 0:
            print("No structures to filter")
            return

        self.extract_md_descriptors()

        A = self.md_descriptors
        idx_env = self._maxvol(A, n_select)

        mapping = self._map_env_to_struct(self.candidate_structures)
        selected_ids = sorted(set(mapping[idx_env]))

        selected = [self.candidate_structures[i] for i in selected_ids]

        self.selected_structures = selected

        out_file = os.path.join(self.iter_dir, "newdata.xyz")
        write(out_file, selected)

        print(f"Selected {len(selected)} diverse structures", flush=True)
    


    def update_dataset(self):

        new_file = os.path.join(self.iter_dir, "newdata.xyz")

        if not os.path.exists(new_file):
            raise RuntimeError("newdata.xyz not found")

        new_structs = read(new_file, ":")

        if self.data is None:
            self.data = new_structs
        else:
            self.data.extend(new_structs)

        dataset_file = os.path.join(self.run_dir, "dataset.xyz")
        write(dataset_file, self.data)

        print(f"Dataset updated: {len(self.data)} total structures", flush=True)
