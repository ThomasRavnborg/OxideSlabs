import os
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

    def __init__(self, run_dir):

        # Define run directory
        self.run_dir = run_dir

        # Load iteration number if it exists, otherwise start at 1
        iter_file = os.path.join(self.run_dir, "iteration.txt")
        if os.path.exists(iter_file):
            with open(iter_file) as f:
                self.iteration = int(f.read().strip())
        else:
            self.iteration = 1
            with open(iter_file, "w") as f:
                f.write(str(self.iteration))
        
        print(f"Current iteration: {self.iteration}", flush=True)

        # Create iteration folder
        self.iter_dir = os.path.join(self.run_dir, f"iteration_{self.iteration}")
        os.makedirs(self.iter_dir, exist_ok=True)

        # Attempt to load existing training and test datasets from the run directory, if they exist.
        try:
            self.train_data = read(os.path.join(self.run_dir, "train.xyz"), ":")
            self.test_data = read(os.path.join(self.run_dir, "test.xyz"), ":")
            print(f"Loaded {len(self.train_data)} training structures and {len(self.test_data)} test structures", flush=True)
            
            self.data = self.train_data + self.test_data
            self.unique_elements = set()
            for atoms in self.data:
                self.unique_elements.update(atoms.get_chemical_symbols())
            self.unique_elements = sorted(self.unique_elements)
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
            self.unique_elements = set()
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
            self.data = self.train_data + self.test_data
            for atoms in self.data:
                self.unique_elements.update(atoms.get_chemical_symbols())
            self.unique_elements = sorted(self.unique_elements)
    

    # 1. Run DFT calculations

    def run_DFT(self):

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

        # Run DFT calculations on structures without calculator results and update the train and test datasets with the results
        _label_DFT(self.train_data, label='train')
        _label_DFT(self.test_data, label='test')

        self.count = 0

    # 2. Set up NEP training

    def setup_nep(self, parameters_nep):
        # Check if data is available and has calculator results before setting up NEP training
        
        try:
            # Load train.xyz and test.xyz from run directory to modify them for NEP training
            nep_train_data = read(os.path.join(self.run_dir, "train.xyz"), ":")
            nep_test_data = read(os.path.join(self.run_dir, "test.xyz"), ":")
        
        except Exception:
            print(f"Missing train.xyz and/or test.xyz. Cannot setup NEP training.", flush=True)
            return

        if self.count > 0:
            print(f"Warning! {self.count} structures have no calculator results.", flush=True)
            return
        
        def _shift_energies(data):
            # Attempt to read energies from energies.json, which should have been generated by the DFT calculations
            try:
                with open(os.path.join(self.run_dir, 'energies.json'), 'r') as f:
                    energies = json.load(f)

            except FileNotFoundError:
                energies = None
                print("Atomic energies file (energies.json) not found in directory.")
                print("It is highly recommended to have the atomic energies of the constituent elements for better NEP training.", flush=True)

            energies = None # temporary

            # Shift the energies of the structures by the sum of the energies of the constituent atoms, if energies are available
            for atoms in data:
                elements = atoms.get_chemical_symbols()
                if energies is not None:
                    atoms.calc.results['energy'] -= sum(energies[element] for element in elements)

        # Shift energies of nep train and test data
        _shift_energies(nep_train_data)
        _shift_energies(nep_test_data)
        print(f"Shifted energies for {len(nep_train_data)} training structures and {len(nep_test_data)} testing structures.")

        # Save the modified train.xyz and test.xyz to the iteration directory for NEP training
        write(os.path.join(self.iter_dir, "train.xyz"), nep_train_data)
        write(os.path.join(self.iter_dir, "test.xyz"), nep_test_data)

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

        lines = [l for l in lines if not l.strip().startswith(("prediction", 'output_descriptor', "dataset"))]

        lines.append("prediction 1\n")          # Set prediction mode
        lines.append("output_descriptor 2\n")   # Output per-atom descriptors
        if dataset is not None:
            lines.append(f"dataset {dataset}\n")

        with open(nep_in, "w") as f:
            f.writelines(lines)
    
    def _unset_prediction_mode(self, nep_in):
        with open(nep_in, "r") as f:
            lines = f.readlines()

        lines = [l for l in lines if not l.strip().startswith(("prediction", 'output_descriptor', "dataset"))]

        with open(nep_in, "w") as f:
            f.writelines(lines)


    def run_prediction_mode(self):
        #nep_dir = os.path.join(self.iter_dir, "nepmodel_split1")
        nep_in = os.path.join(self.iter_dir, "nep.in")

        # Check if there is an existing descriptor.out file and remove it
        desc_file = os.path.join(self.iter_dir, "descriptor.out")
        if os.path.exists(desc_file):
            print("Existing descriptor.out file found. Was removed.", flush=True)
            os.remove(desc_file)

        self._set_prediction_mode(nep_in)

        print("Running NEP in prediction mode...", flush=True)
        subprocess.run(["nep"], cwd=self.iter_dir,
                       check=True, text=True)
        
        self._unset_prediction_mode(nep_in)


    def _extract_descriptors(self, desc_file):

        B = np.loadtxt(desc_file)
        print(f"Loaded descriptor matrix: {B.shape}", flush=True)

        return B


    def _calculate_descriptors(self, structures):
        B = []
        for structure in structures:
            B.append(get_descriptors(structure, os.path.join(self.iter_dir, "nep.txt")))
        B = np.vstack(B)

        print(f"Computed descriptor matrix: {B.shape}", flush=True)

        return B


    def compute_descriptors(self, get_out=True, write_out=True):
        
        desc_file = os.path.join(self.iter_dir, "descriptor.out")
        nep_file = os.path.join(self.iter_dir, "nep.txt")

        if os.path.exists(desc_file) and get_out:
            B = self._extract_descriptors(desc_file)

        elif os.path.exists(nep_file):
            print("Descriptor file not found, but nep.txt exists. Computing descriptors with Calorine.", flush=True)
            B = self._calculate_descriptors(self.train_data)
            # Save to descriptor.out for future use
            if write_out:
                np.savetxt(os.path.join(self.iter_dir, "descriptor.out"), B)

        else:
            raise RuntimeError("Neither descriptor.out nor nep.txt found. Cannot compute descriptors.")

        self.descriptors = B


    def _map_descriptors_to_type(self, structures, descriptors):

        type_map = {elem: i for i, elem in enumerate(self.unique_elements)}

        type_index = []
        struct_index = []
        atom_index = []

        for i, atoms in enumerate(structures):
            for j, atom in enumerate(atoms):
                type_index.append(type_map[atom.symbol])
                struct_index.append(i)
                atom_index.append(j)

        type_index = np.array(type_index)
        struct_index = np.array(struct_index)
        atom_index = np.array(atom_index)

        descriptors_by_type = {}
        struct_index_by_type = {}
        atom_index_by_type = {}

        for t in np.unique(type_index):
            mask = (type_index == t)
            descriptors_by_type[t] = descriptors[mask]
            struct_index_by_type[t] = struct_index[mask]
            atom_index_by_type[t] = atom_index[mask]

        return descriptors_by_type, struct_index_by_type, atom_index_by_type



    def _extract_active_set(self, asi_file):

        from src.asiIO import load_asi
        def find_inverse(m):
            return np.linalg.pinv(m, rcond=1e-8)

        active_set_inv = load_asi(asi_file)
        #A_active = find_inverse(A_inv)

        return active_set_inv

    def _calculate_active_set(self, structures, descriptors, batch_size=None):

        from src.MaxVol import calculate_maxvol

        B_type, struct_index_type, atom_index_type = self._map_descriptors_to_type(structures, descriptors)

        active_set = {}
        active_set_struct = []  # the index of structure

        for t in B_type:
            print(f"Performing MaxVol for type {t}...")

            A_t = B_type[t]
            struct_t = struct_index_type[t]

            A_active_t, selected_index = calculate_maxvol(
                A_t,
                struct_t,
                batch_size=batch_size
            )

            active_set[t] = A_active_t
            active_set_struct.extend(selected_index)
        
        active_set_struct = list(set(active_set_struct))
        active_set_struct.sort()
        
        active_set_inv = {
            t: np.linalg.inv(A).astype(np.float32)
            for t, A in active_set.items()
        }
        
        return active_set_inv, active_set_struct

    


    
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

        #def find_inverse(m):
        #    return np.linalg.pinv(m, rcond=1e-8)

        asi_file = os.path.join(self.iter_dir, "active_set.asi")
        if os.path.exists(asi_file) and get_asi:
            print("Existing active set inverse found. Loading...")
            self.active_inv = self._extract_active_set(asi_file)
        else:
            print("Building active set...")
            self.active_inv, _ = self._calculate_active_set(self.train_data, self.descriptors, batch_size=batch_size)
            if write_asi:
                save_asi(self.active_inv, asi_file)
                print(f"Active set inverse saved to {asi_file}", flush=True)



    def setup_MD(self, temperature=300, n_steps=10000, dump_interval=100):
        md_dir = os.path.join(self.iter_dir, "md")
        os.makedirs(md_dir, exist_ok=True)

        # Copy the trained NEP model to the MD directory
        nep_src = os.path.join(self.iter_dir, "nep.txt")
        #nep_dst = os.path.join(md_dir, "nep.txt")

        if not os.path.exists(nep_src):
            raise RuntimeError("nep.txt not found. Train NEP first.")

        #shutil.copy(nep_src, nep_dst)
        
        # Choose a structure from the dataset to run MD on
        if self.data is None or len(self.data) == 0:
            raise RuntimeError("No structures available for MD")
        # For now, just take the first structure with calculator results.
        atoms = self.data[0]  # simple choice for now

        model_path = os.path.join(md_dir, "model.xyz")

        write(model_path, atoms)

        run_in = f"""
        potential ../nep.txt

        velocity {temperature}
        time_step 1.0
        dump_exyz {dump_interval} 0 1

        compute_extrapolation asi_file ../active_set.asi gamma_low 1 gamma_high 10 check_interval {dump_interval} dump_interval {dump_interval}

        ensemble npt_mttk iso 0 0 temp {temperature} {temperature+200}
        run {n_steps}

        ensemble npt_mttk iso 0 0 temp {temperature+200} {temperature}
        run {n_steps}
        """

        with open(os.path.join(md_dir, "run.in"), "w") as f:
            text = "\n".join(line.strip() for line in run_in.splitlines())
            f.write(text)


    def run_MD(self):
        md_dir = os.path.join(self.iter_dir, "md")
        print("Running GPUMD...")

        subprocess.run(["gpumd"], cwd=md_dir, check=True, text=True)
    
    """
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

        #A = (A - A.mean(axis=0)) / (A.std(axis=0) + 1e-12)

        self.md_descriptors = A

        print(f"MD descriptor matrix: {A.shape}", flush=True)
    """

    def assign_gamma(self, structures, descriptors):
        
        # initialize gamma arrays
        for atoms in structures:
            atoms.arrays["gamma"] = np.zeros(len(atoms))

        #B = self._calculate_descriptors(structures)   # (N, D) descriptor matrix for all environments
        B_type, struct_index_by_type, atom_index_by_type = self._map_descriptors_to_type(structures, descriptors)
        #A_inv, active_set_struct = self._calculate_active_set(structures, B, batch_size=None)
        
        # loop over atom types
        for t, Ainv in self.active_inv.items():

            B_t = B_type[t]
            struct_t = struct_index_by_type[t]
            atom_t = atom_index_by_type[t]

            # compute gamma
            g = B_t @ Ainv
            g = np.max(np.abs(g), axis=1)

            # map back to structures
            for val, s_idx, a_idx in zip(g, struct_t, atom_t):
                structures[s_idx].arrays["gamma"][a_idx] = val

        #return structures

    def filter_structures(self, structures, gamma_th=5.0):

        print(f"Filtering out structures with gamma < {gamma_th}...")
        print("Calculating descriptors for structures...", flush=True)

        # Calculate descriptors for the structures and assign gamma values to each atom (environment)
        B = self._calculate_descriptors(structures)
        self.assign_gamma(structures, B)

        # Filter structures with gamma above a certain threshold
        selected_structures = [atoms for atoms in structures if atoms.arrays["gamma"].max() > gamma_th]
        
        # If no structures are selected, print a warning and return an empty list
        if len(selected_structures) == 0:
            print(f"No structures found with gamma > {gamma_th}.")
            print("Consider lowering the gamma threshold.", flush=True)
            return []
        
        print(f"Found {len(selected_structures)} structures with gamma > {gamma_th}.")
        
        write(os.path.join(self.iter_dir, "large_gamma.xyz"), selected_structures)
        
        #return selected_structures


    def select_structures(self, structures):

        print("Performing diversity selection with MaxVol")

        # Combine training data with new structures
        data = self.train_data + structures

        # Calculate descriptors for the combined set of structures
        B = self._calculate_descriptors(data)

        # Compute an active set for the combined set of structures
        A_inv, active_set_struct = self._calculate_active_set(data, B, batch_size=None)

        # Return new structures that are in the active set but not in the original training data
        filtered_structures = [data[i] for i in active_set_struct if i >= len(self.train_data)]
        
        print(f"Found {len(filtered_structures)} filtered structures.", flush=True)

        write(os.path.join(self.iter_dir, "newdata.xyz"), filtered_structures)
        #return filtered_structures


    def _map_env_to_struct(self, structures):

        counts = [len(s) for s in structures]

        mapping = []
        for i, n in enumerate(counts):
            mapping.extend([i] * n)

        return np.array(mapping)



    def update_dataset(self):

        new_file = os.path.join(self.iter_dir, "newdata.xyz")

        if not os.path.exists(new_file):
            raise RuntimeError("newdata.xyz not found")

        new_structs = read(new_file, ":")

        self.train_data.extend(new_structs)

        write(os.path.join(self.run_dir, "train.xyz"), self.train_data)

        print(f"Sucessfully added {len(new_structs)} structures to train.xyz")
        print(f"Total structures in train.xyz: {len(self.train_data)}", flush=True)

        # Update iteration number and overwrite iteration.txt
        self.iteration += 1
        iter_file = os.path.join(self.run_dir, "iteration.txt")
        with open(iter_file, "w") as f:
            f.write(str(self.iteration))
