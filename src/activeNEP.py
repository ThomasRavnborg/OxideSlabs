import os
import json
import random
import shutil
import subprocess
import numpy as np
import phonopy as ph
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from ase.io import read, write
from ase.optimize import BFGS
from ase.build import sort
from calorine.nep import get_descriptors, get_parity_data, read_loss, read_structures
from ase.filters import FrechetCellFilter
from calorine.calculators import CPUNEP
from calorine.tools import get_force_constants
from hiphive.structure_generation import generate_phonon_rattled_structures
from src.latexfig import LatexFigure
from src.phononASE import phonon_to_atoms, phonopy_to_ase
from src.calculators import run_siesta, copy_calc_results
from src.structureoptimizer import opt_filter
from src.MaxVol import calculate_maxvol
from src.structure import check_if_bulk
from src.asiIO import save_asi, load_asi
from src.gpumdIO import save_run_in

class ActiveLearningNEP:

    def __init__(self, run_dir):

        # Define run directory
        self.run_dir = run_dir

        # Load iteration number if it exists, otherwise start at 1
        iter_file = os.path.join(self.run_dir, "iteration.txt")
        if os.path.exists(iter_file):
            with open(iter_file) as f:
                self.iteration = int(f.read().strip())
            self.iter_dir = os.path.join(self.run_dir, f"iteration_{self.iteration}")
        else:
            self.iteration = 1
            with open(iter_file, "w") as f:
                f.write(str(self.iteration))
            # Create iteration folder
            self.iter_dir = os.path.join(self.run_dir, f"iteration_{self.iteration}")
            os.makedirs(self.iter_dir, exist_ok=True)
        
        print(f"Current iteration: {self.iteration}", flush=True)
        # Create iteration folder
        #self.iter_dir = os.path.join(self.run_dir, f"iteration_{self.iteration}")
        #os.makedirs(self.iter_dir, exist_ok=True)

        # Attempt to load existing training and test datasets from the run directory, if they exist.
        try:
            # Load train.xyz and test.xyz from run directory
            self.train_data = read(os.path.join(self.run_dir, "train.xyz"), ":")
            self.test_data = read(os.path.join(self.run_dir, "test.xyz"), ":")
            print(f"Loaded {len(self.train_data)} training structures and {len(self.test_data)} test structures", flush=True)
            
            self.data = self.train_data + self.test_data
            self.species = set()
            for atoms in self.data:
                self.species.update(atoms.get_chemical_symbols())

            self.species = sorted(self.species)
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
            else:
                print("All structures have calculator results.", flush=True)

                asi_file = os.path.join(self.iter_dir, "active_set.asi")
                xyz_file = os.path.join(self.iter_dir, "active_set.xyz")

                if os.path.exists(asi_file) and os.path.exists(xyz_file):
                    print("Existing active set inverse (.asi) and structures (.xyz) found. Loading...")
                    active_set_inv = load_asi(asi_file)
                    self.active_set_inv = dict(zip(self.species, active_set_inv.values()))
                    self.active_set_struct = read(xyz_file, ":")

        except Exception:
            self.train_data = None
            self.test_data = None
            self.data = None
            self.species = set()
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
                bulk = check_if_bulk(atoms)

                # Extract all displaced structures and convert them to ASE Atoms objects
                for atoms_phonopy in phonon.supercells_with_displacements:
                    bulk = check_if_bulk(atoms_phonopy)
                    # Convert the phonopy Atoms object to an ASE Atoms object and append
                    displaced_structures = phonopy_to_ase(atoms_phonopy, bulk=bulk)
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

            self.train_data = [sort(atoms) for atoms in train_data]
            self.test_data = [sort(atoms) for atoms in test_data]

            write(os.path.join(self.run_dir, "train.xyz"), self.train_data)
            write(os.path.join(self.run_dir, "test.xyz"), self.test_data)

            self.data = self.train_data + self.test_data
            for atoms in self.data:
                self.species.update(atoms.get_chemical_symbols())
            self.species = sorted(self.species)
    

    # 1. Run DFT calculations

    def run_DFT(self):

        try:
            with open(os.path.join(self.run_dir, 'dft_params.json'), 'r') as f:
                dft_params = json.load(f)

        except FileNotFoundError:
            print("DFT parameters file (dft_params.json) not found.", flush=True)
            return

        if self.data is None:
            print("No data available to run DFT. Please prepare the dataset first.", flush=True)
            return
        
        basis_file = os.path.join(self.run_dir, 'basis.fdf')
        dft_dir = os.path.join(self.run_dir, 'DFT')
        target_file = os.path.join(dft_dir, 'basis.fdf')

        os.makedirs(dft_dir, exist_ok=True)

        # Check if basis file exists in the run directory
        if os.path.exists(basis_file):
            # If it exists in the run directory, check if it already exists in the DFT directory
            if os.path.exists(target_file):
                # If it already exists in the DFT directory, do nothing
                return
            # If it does not exist in the DFT directory, copy it from the run directory
            shutil.copy(basis_file, target_file)
        else:
            # If it does not exist in the run directory, create an empty basis.fdf file in the DFT directory
            if not os.path.exists(target_file):
                open(target_file, 'w').close()

        if self.count == 0:
            print("All structures have calculator results. No DFT calculations needed.")
            return

        print(f"Running DFT on {self.count} structures without calculator assigned...", flush=True)
        def _label_DFT(data, label='train'):
            for i in range(len(data)):
                struct = data[i]
                if struct.calc is None:
                    #generate_basis(struct, dir=os.path.join(self.run_dir, 'DFT'))
                    run_siesta(struct, **dft_params, dir=dft_dir)
                    data[i] = copy_calc_results(struct)
                    write(os.path.join(self.run_dir, f"{label}.xyz"), data)

        # Run DFT calculations on structures without calculator results and update the train and test datasets with the results
        _label_DFT(self.train_data, label='train')
        _label_DFT(self.test_data, label='test')

        self.count = 0

    # 2. Set up NEP training

    def setup_nep(self, cutoff=[8, 4], neuron=30, generation=100000, batch=1000000):
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

            # Shift the energies of the structures by the sum of the energies of the constituent atoms, if energies are available
            for atoms in data:
                elements = atoms.get_chemical_symbols()
                if energies is not None:
                    atoms.calc.results['energy'] -= sum(energies[element] for element in elements)

        # Sort atoms by alphabetical order of chemical symbols
        #nep_train_data = [copy_calc_results(atoms, sort=True) for atoms in nep_train_data]
        #nep_test_data = [copy_calc_results(atoms, sort=True) for atoms in nep_test_data]

        # Shift energies of nep train and test data
        _shift_energies(nep_train_data)
        _shift_energies(nep_test_data)
        print(f"Shifted energies for {len(nep_train_data)} training structures and {len(nep_test_data)} testing structures.")

        nep_dir = os.path.join(self.iter_dir, "nep")
        os.makedirs(nep_dir, exist_ok=True)

        # Save the modified train.xyz and test.xyz to the iteration directory for NEP training
        write(os.path.join(nep_dir, "train.xyz"), nep_train_data)
        write(os.path.join(nep_dir, "test.xyz"), nep_test_data)

        # Set up the input files for NEP training
        params = dict(version=4, type=[len(self.species), ' '.join(self.species)])
        params.update(cutoff=cutoff, neuron=neuron, generation=generation, batch=batch)

        # Dump params to a .in file
        with open(os.path.join(nep_dir, "nep.in"), "w") as f:
            for key, value in params.items():
                if isinstance(value, list):
                    value_str = ' '.join(map(str, value))
                else:
                    value_str = str(value)
                f.write(f"{key}  {value_str}\n")


    def train_nep(self):
        nep_dir = os.path.join(self.iter_dir, "nep")
        subprocess.run(["nep"], cwd=nep_dir, check=True, text=True)
        # Copy nep.txt to iteration directory for later use in descriptor calculation
        nep_txt_src = os.path.join(nep_dir, "nep.txt")
        nep_txt_target = os.path.join(self.iter_dir, "nep.txt")
        shutil.copy(nep_txt_src, nep_txt_target)
    

    """
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
    """

    """
    def _extract_descriptors(self, desc_file):

        B = np.loadtxt(desc_file)
        print(f"Loaded descriptor matrix: {B.shape}", flush=True)

        return B


    def _calculate_descriptors(self, structures):
        B = []
        for structure in structures:
            B.append(self._calculate_descriptor(structure))
        B = np.vstack(B)

        print(f"Computed descriptor matrix: {B.shape}", flush=True)

        return B
    """

    def _calculate_descriptors(self, structure):
        B = get_descriptors(structure, os.path.join(self.iter_dir, "nep.txt"))
        structure.arrays['descriptor'] = B

    def assign_descriptors(self, structures):
        from tqdm import tqdm
        # Filter out structures that already have descriptors calculated
        structures = [s for s in structures if 'descriptor' not in s.arrays]
        if len(structures) == 0:
            print("All structures already have descriptors calculated.")
            return
        print(f"Calculating descriptors for {len(structures)} structures...")
        for structure in tqdm(structures):
            if 'descriptor' not in structure.arrays:
                self._calculate_descriptors(structure)

    """

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

    """

    def _collect_descriptors(self, structures, specie):

        B_specie = []           # per-species descriptor matrix (n_atoms_of_specie, d)
        struct_indicies = []    # list of structure indices corresponding to each row in B_specie
        atom_indicies = []      # list of atom indices corresponding to each row in B_specie

        for struct_index, structure in enumerate(structures):

            # mask for desired chemical symbol
            mask = (structure.symbols == specie)

            if not np.any(mask):
                continue

            desc = structure.arrays['descriptor'][mask]  # shape (n_selected_atoms, d)

            B_specie.append(desc)
            struct_indicies.extend([struct_index] * len(desc))
            atom_indicies.extend(np.where(mask)[0])
        if len(B_specie) == 0:
            raise ValueError(f"No atoms with symbol {specie} found.")

        B_specie = np.vstack(B_specie)
        struct_indicies = np.array(struct_indicies)
        atom_indicies = np.array(atom_indicies)

        return B_specie, struct_indicies, atom_indicies


    def _calculate_active_set(self, structures, gamma_tol, maxvol_iter, batch_size, n_refinement):

        self.assign_descriptors(structures)
        
        # Create active set matrix A for each specie
        active_set = {}
        active_set_index = []  # the index of structure

        # Loop over species
        for specie in self.species:
            print(f'Building active set for {specie}...')
            B_specie, struct_indicies, _ = self._collect_descriptors(structures, specie)
            A_specie, select_indicies = calculate_maxvol(B_specie, struct_indicies,
                                                         gamma_tol, maxvol_iter,
                                                         batch_size, n_refinement)
            active_set[specie] = A_specie
            active_set_index.extend(select_indicies)
        
        # Remove duplicates and sort the structure indices
        active_set_index = list(set(active_set_index))
        active_set_index.sort()

        # Calculate the inverse of each active set matrix
        active_set_inv = {
            t: np.linalg.inv(A).astype(np.float32)
            for t, A in active_set.items()
        }

        return active_set_inv, active_set_index


    def build_active_set(self, gamma_tol=1.001, maxvol_iter=1000, batch_size=None, n_refinement=10):
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

        asi_file = os.path.join(self.iter_dir, "active_set.asi")
        xyz_file = os.path.join(self.iter_dir, "active_set.xyz")

        if os.path.exists(asi_file) and os.path.exists(xyz_file):
            print("Existing active set inverse (.asi) and structures (.xyz) found. Loading...")
            active_set_inv = load_asi(asi_file)
            self.active_set_inv = dict(zip(self.species, active_set_inv.values()))
            self.active_set_struct = read(xyz_file, ":")

        else:
            print("Building active set...")
            active_set_inv, active_set_index = self._calculate_active_set(self.train_data,
                                                                          gamma_tol, maxvol_iter,
                                                                          batch_size, n_refinement)
           
            self.active_set_inv = active_set_inv
            save_asi(self.active_set_inv, asi_file)
            print(f"Active set inverse saved to {asi_file}")
            
            self.active_set_struct = [self.train_data[i] for i in active_set_index]

            active_set_struct_copy = [copy_calc_results(atoms) for atoms in self.active_set_struct]
            for atoms in active_set_struct_copy:
                del atoms.arrays['descriptor']

            write(xyz_file, active_set_struct_copy)
            print(f"Active set structure saved to {xyz_file}", flush=True)


    """
    def setup_MD(self, n_traj=1, n_steps=10000, T0=20, T1=700):

        # Check if there is any data loaded
        if self.data is None or len(self.data) == 0:
            raise RuntimeError("No structures available for MD")

        # Look for a NEP model in the iteration directory
        nep_src = os.path.join(self.iter_dir, "nep.txt")
        if not os.path.exists(nep_src):
            raise RuntimeError("nep.txt not found. Train NEP first.")
        
        # Create directory for MD results
        md_dir = os.path.join(self.iter_dir, "md")
        os.makedirs(md_dir, exist_ok=True)
        
        # Split up structures by chemical formula
        labels = set()
        train_data_dict = {}
        for atoms in self.train_data:
            label = atoms.get_chemical_formula()
            labels.add(label)
            if label not in train_data_dict:
                train_data_dict[label] = []
            train_data_dict[label].append(atoms)
        labels = list(labels)

        for i, label in enumerate(labels):

            for j in range(n_traj):
                run = i*n_traj + j + 1

                if j == 0:
                    atoms = train_data_dict[label][0].copy()
                else:
                    atoms = random.choice(train_data_dict[label]).copy()

                md_run_dir = os.path.join(md_dir, f"run_{run:03d}")
                os.makedirs(md_run_dir, exist_ok=True)
                # Write atoms object to md_run_dir without calculator results

                write(os.path.join(md_run_dir, "model.xyz"), atoms)
                # Create run.in file for GPUMD to run MD simulations with the trained NEP model
                save_run_in(n_steps, T0, T1, md_run_dir)
    
    def run_MD(self):
        md_dir = os.path.join(self.iter_dir, "md")
        # List all folders in md_dir
        folders = [d for d in os.listdir(md_dir) if os.path.isdir(os.path.join(md_dir, d))]
        print(f"Running {len(folders)} MD simulations with GPUMD ...", flush=True)

        for folder in folders:
            md_run_dir = os.path.join(md_dir, folder)
            subprocess.run(["gpumd"], cwd=md_run_dir, check=True, text=True)
    
            def collect_MD_structures(self):
        md_dir = os.path.join(self.iter_dir, "md")
        # List all folders in md_dir
        folders = [d for d in os.listdir(md_dir) if os.path.isdir(os.path.join(md_dir, d))]
        trajs = []
        for folder in folders:
            md_run_dir = os.path.join(md_dir, folder)
            model = read(os.path.join(md_run_dir, "model.xyz"))
            traj = read(os.path.join(md_run_dir, "dump.xyz"), index=":")
            # Wrap trajectory to reference structure
            #traj = wrap_to_reference(traj, model)
            # Append the wrapped trajectory to the trajs list
            trajs.extend(traj)

        # Save all trajectories to a single file
        write(os.path.join(md_dir, "md_structures.xyz"), trajs)
    """

    def setup_MD(self, dt=1, n_steps=1*1e6, n_dump=1000, temperatures=[300, 500, 700]):
        """Function to set up MD simulations with GPUMD using the trained NEP model.
        It creates a directory for each trajectory and saves the initial structure and run.in file for GPUMD.
        - Args:
            dt (float): time step in fs
            n_steps (int): number of MD steps
            n_dump (int): total number of dumps to save during MD
            temperatures (list): list of temperatures for the MD simulations
        - Returns:
             None: The function saves the initial structures and run.in files for GPUMD in the iteration directory under "md/label/temperature/".
             The MD simulations can then be run with the run_MD() method, and the resulting trajectories can be collected with the collect_MD_structures() method.
        """

        # Check if there is any data loaded
        if self.data is None or len(self.data) == 0:
            raise RuntimeError("No structures available for MD")

        # Look for a NEP model in the iteration directory
        nep_src = os.path.join(self.iter_dir, "nep.txt")
        if not os.path.exists(nep_src):
            raise RuntimeError("nep.txt not found. Train NEP first.")
        
        # Create directory for MD results
        md_dir = os.path.join(self.iter_dir, "md")
        os.makedirs(md_dir, exist_ok=True)
        
        # Split up structures by chemical formula
        labels = set()
        train_data_dict = {}
        for atoms in self.train_data:
            label = atoms.get_chemical_formula()
            labels.add(label)
            if label not in train_data_dict:
                train_data_dict[label] = []
            train_data_dict[label].append(atoms)
        labels = list(labels)

        for label in labels:
            
            label_dir = os.path.join(md_dir, label)
            # Create directory for this chemical formula
            os.makedirs(label_dir, exist_ok=True)

            # Take the first structure for this chemical formula as the initial structure for the MD simulations.
            atoms = train_data_dict[label][0].copy()
            bulk = check_if_bulk(atoms)
            # Relax the structure with the NEP model
            self.relax_atoms(atoms)

            atoms_copy = copy_calc_results(atoms)

            for T in temperatures:

                temp_dir = os.path.join(label_dir, f"{T}K")
                # Create directory for this temperature
                os.makedirs(temp_dir, exist_ok=True)
                
                # Write atoms object to temp_dir without calculator results
                write(os.path.join(temp_dir, "model.xyz"), atoms_copy)
                # Create run.in file for GPUMD to run MD simulations with the trained NEP model
                save_run_in(dt, n_steps, n_dump, T, bulk, temp_dir)

    def run_MD(self):
        """Function to run MD simulations with GPUMD using the initial structures and run.in files set up by the setup_MD() method.
        It loops over the directories created by setup_MD() and runs GPUMD in each run directory, which should contain the model.xyz and run.in files for each trajectory.
        - Args:
            None: The function assumes that the setup_MD() method has already been run to create the necessary directories and input files for GPUMD.
        - Returns:
            None: The function runs GPUMD in each trajectory directory, which will generate the MD trajectories and save them in the same directories.
            The resulting trajectories can then be collected with the collect_MD_structures() method.
        """
        # List all folders in md_dir
        md_dir = os.path.join(self.iter_dir, "md")
        md_folders = [d for d in os.listdir(md_dir) if os.path.isdir(os.path.join(md_dir, d))]
        # Loop over chemical formula folders, then temperature folders to run GPUMD simulations
        for label in md_folders:
            label_dir = os.path.join(md_dir, label)
            temp_folders = [d for d in os.listdir(label_dir) if os.path.isdir(os.path.join(label_dir, d))]
            for temp in temp_folders:
                temp_dir = os.path.join(label_dir, temp)
                # Run GPUMD in the temp_dir, which should contain the model.xyz and run.in files for this trajectory
                subprocess.run(["gpumd"], cwd=temp_dir, check=True, text=True)


    def collect_MD_structures(self):
        """Function to collect the MD trajectories generated by the run_MD() method.
        It loops over the directories created by setup_MD() and collects the trajectories from the dump.xyz files generated by GPUMD simulations.
        - Args:
            None: The function assumes that the run_MD() method has already been run to generate the MD trajectories with GPUMD.
        - Returns:
            None: The function collects the trajectories from the dump.xyz files in each trajectory directory and saves all trajectories to a single file called md_structures.xyz in the md directory of the current iteration.
        """
        trajs = []
        # List all folders in md_dir
        md_dir = os.path.join(self.iter_dir, "md")
        md_folders = [d for d in os.listdir(md_dir) if os.path.isdir(os.path.join(md_dir, d))]
        # Loop over chemical formula folders, then temperature folders and
        # collect the trajectories from the dump.xyz files generated by GPUMD simulations
        trajs = []
        for label in md_folders:
            label_dir = os.path.join(md_dir, label)
            temp_folders = [d for d in os.listdir(label_dir) if os.path.isdir(os.path.join(label_dir, d))]
            for temp in temp_folders:
                temp_dir = os.path.join(label_dir, temp)
                # Read the trajectory from the dump.xyz file generated by GPUMD in this run directory
                traj = read(os.path.join(temp_dir, "dump.xyz"), index=":")
                trajs.extend(traj)
        
        for atoms in trajs:
            # Remove all info
            atoms.info.clear()
        # Sort atoms in each structure by chemical symbol to ensure consistent ordering
        trajs = [sort(atoms) for atoms in trajs]

        # Save all trajectories to a single file
        write(os.path.join(md_dir, "md_structures.xyz"), trajs)

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

    def assign_gamma(self, structures):

        if self.active_set_inv is None:
            raise RuntimeError("Active set inverse not found. Build active set first.")

        for structure in structures:
            structure.arrays["gamma"] = np.zeros(len(structure))
        
        self.assign_descriptors(structures)

        # loop over atom types
        for specie, A_inv in self.active_set_inv.items():

            B_specie, struct_indicies, atom_indicies = self._collect_descriptors(structures, specie)

            # compute gamma
            gamma = B_specie @ A_inv
            gamma = np.max(np.abs(gamma), axis=1)

            # map back to structures
            for val, s_idx, a_idx in zip(gamma, struct_indicies, atom_indicies):
                structures[s_idx].arrays["gamma"][a_idx] = val
    

    """
    def assign_gamma(self, structures, descriptors):
        
        # initialize gamma arrays
        for structure in structures:
            structure.arrays["gamma"] = np.zeros(len(structure))

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
    """

    def filter_structures(self, structures, gamma_th=5.0):

        print(f"Filtering out structures with gamma < {gamma_th}...")
        print("Calculating descriptors for structures...", flush=True)

        # Assign gamma values to each atom (environment)
        self.assign_gamma(structures)

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


    def select_structures(self, structures, gamma_tol=1.001, maxvol_iter=1000, batch_size=None, n_refinement=10):

        print("Performing diversity selection with MaxVol")

        # Combine training data with new structures
        data = self.train_data + structures

        # Compute an active set for the combined set of structures
        active_set_inv, active_set_index = self._calculate_active_set(data,
                                                                      gamma_tol, maxvol_iter,
                                                                      batch_size, n_refinement)

        # Return new structures that are in the active set but not in the original training data
        filtered_structures = []
        for i in active_set_index:
            if i >= len(self.train_data):
                atoms = data[i].copy()
                del atoms.arrays['descriptor']
                del atoms.arrays['gamma']
                filtered_structures.append(atoms)
        #filtered_structures = [data[i] for i in active_set_index if i >= len(self.train_data)]
        
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

        self.count = len(new_structs)

        # Update iteration number and overwrite iteration.txt
        self.iteration += 1
        self.iter_dir = os.path.join(self.run_dir, f"iteration_{self.iteration}")
        os.makedirs(self.iter_dir, exist_ok=True)
        iter_file = os.path.join(self.run_dir, "iteration.txt")
        with open(iter_file, "w") as f:
            f.write(str(self.iteration))


    def relax_atoms(self, atoms, filt=True, fmax=0.1, strained=False):

        calc = CPUNEP(os.path.join(self.iter_dir, 'nep.txt'))

        # Attach the calculator to the atoms object
        atoms.calc = calc

        # Apply filter to optimize unit cell parameters and atomic positions if filt is True
        # Otherwise only optimize atomic positions
        if filt:
            atoms_filt = opt_filter(atoms, strained)
        else:
            atoms_filt = atoms
        
        # Set up the BFGS optimizer on all processes
        opt = BFGS(atoms_filt)
        # Run the optimization until forces are smaller than fmax
        opt.run(fmax=fmax*1.e-3)
        #return atoms



    def calculate_phonon(self, atoms, Nc=2):
        calc = CPUNEP(os.path.join(self.iter_dir, 'nep.txt'))
        atoms = atoms.copy()
        atoms.calc = calc
        if check_if_bulk(atoms):
            supercell = [Nc, Nc, Nc]
        else:
            supercell = [Nc, Nc, 1]
        phonon = get_force_constants(atoms, calc, supercell)
        return phonon

    

    def plot_loss(self, width=1, AR=0.25):
        loss = read_loss(os.path.join(self.iter_dir, 'loss.out'))

        lf = LatexFigure()
        fig, axes = lf.create(AR=AR, width=width, subplots=(2, 1), sharex=True)

        ax = axes[0]
        ax.set_ylabel('Loss')
        ax.plot(loss.total_loss, label='total')
        ax.plot(loss.L2, label='$l_2$')
        ax.plot(loss.L1, label='$l_1$')
        ax.set_yscale('log')
        ax.legend()

        ax = axes[1]
        ax.plot(loss.RMSE_F_train, label='forces (eV/Å)')
        ax.plot(loss.RMSE_V_train, label='virial (eV/atom)')
        ax.plot(loss.RMSE_E_train, label='energy (eV/atom)')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Generation')
        ax.set_ylabel('RMSE')
        ax.legend()

        #fig.subplots_adjust(hspace=0)
        fig.set_constrained_layout_pads(hspace=0.0, h_pad=0.0)
        plt.show()



    def plot_parity(self, width=0.8, AR=0.9):
        training_structures, _ = read_structures(self.iter_dir)

        units = dict(
            energy='eV/atom',
            force='eV/Å',
            virial='eV/atom',
            stress='GPa',
        )
        # Make a 2x2 grid of parity plots for energy, force, virial, and stress
        lf = LatexFigure()
        fig, axes = lf.create(width=width, AR=AR, subplots=(2, 2))
        kwargs = dict(alpha=0.2, s=0.3)
        axes = axes.flatten()

        # Loop over the properties and units, get the parity data, calculate R2 and RMSE, and plot the parity plots
        for icol, (prop, unit) in enumerate(units.items()):
            df = get_parity_data(training_structures, prop, flatten=True)
            R2 = r2_score(df.target, df.predicted)
            rmse = np.sqrt(mean_squared_error(df.target, df.predicted))

            ax = axes[icol]
            ax.set_xlabel(f'Target {prop} ({unit})')
            ax.set_ylabel(f'Predicted ({unit})')
            # Plot line x = y for reference
            ax.plot([df.target.min(), df.target.max()], [df.target.min(), df.target.max()], 'k-', lw=0.8, alpha=0.2)
            ax.scatter(df.target, df.predicted, **kwargs)
            ax.set_aspect('equal')
            mod_unit = unit.replace('eV', 'meV').replace('GPa', 'MPa')
            ax.text(0.1, 0.75, f'{1e3*rmse:.1f} {mod_unit}\n' + '$R^2= $' + f' {R2:.5f}', transform=ax.transAxes)
        fig.align_labels()
        plt.show()