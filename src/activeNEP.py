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

    def __init__(self, run_dir, iteration=1, overwrite=False):

        self.run_dir = run_dir
        self.iteration = iteration

        # Global dataset
        self.dataset_path = os.path.join(run_dir, "dataset.xyz")

        # Iteration folder
        self.iter_dir = os.path.join(run_dir, f"iteration_{iteration}")
        os.makedirs(self.iter_dir, exist_ok=True)

        # Snapshot dataset for THIS iteration
        self.iter_dataset_path = os.path.join(self.iter_dir, "dataset.xyz")

        if os.path.exists(self.dataset_path):
            self.data = read(self.dataset_path, ":")
            print(f"Loaded global dataset: {len(self.data)} structures", flush=True)
            self.count = len([s for s in self.data if s.calc is None])
            if self.count > 0:
                if self.count == len(self.data):
                    print("Warning! No structures have calculator results.")
                    print("DFT calculations must be runwith run_DFT(), or the dataset will be empty.", flush=True)
                elif self.count == 1:
                    print("Warning! 1 structure has no calculator results.")
                    print("DFT calculations must be runwith run_DFT(), or this structure will be omitted.", flush=True)
                else:
                    print(f"Warning! {self.count} structures have no calculator results.")
                    print("DFT calculations must be run with run_DFT(), or these will be omitted.", flush=True)
        
        else:
            print("No global dataset found.", flush=True)
            self.data = None

    # 0. Prepare dataset for NEP training

    def prepare_dataset(self, overwrite=False,
                        n_rattle=100, temperatures=[300, 500, 700],
                        n_strain=10, strains=[-1.0, -0.5, 0.5, 1.0]):
        
        if self.data is not None and not overwrite:
            print("Existing data found. Data preperation skipped.")
            print("To overwrite existing data, set overwrite=True.", flush=True)
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
                
                print(f"Extracted {len(structures)} structures from phonon calculation (including original and displaced)", flush=True)
                
                def _get_rattled_structures(atoms, fc, T, n_structures):
                    # Produce phonon rattled structures using the force constants and the original structure
                    rattled_structures = generate_phonon_rattled_structures(atoms, fc2=fc, temperature=T,
                                                                            n_structures=n_structures)
                    return rattled_structures

                for T in temperatures:
                    structures.extend(_get_rattled_structures(atoms, fc, T, n_rattle))
                
                print(f"Generated {n_rattle*len(temperatures)} rattled structures", flush=True)

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

                print(f"Generated {n_strain*len(strains)} strained structures", flush=True)

                print(f"{len(structures)} total structures", flush=True)
                data.extend(structures)

            self.data = data
            write(os.path.join(self.run_dir, "dataset.xyz"), data)
    
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
        
        self.count = len([s for s in self.data if s.calc is None])
        print(f"Running DFT on {self.count} structures without calculator assigned...", flush=True)
        for i in range(len(self.data)):
            struct = self.data[i]
            if struct.calc is None:
                run_siesta(struct, **dft_params, dir=os.path.join(self.run_dir, 'siesta'))
                self.data[i] = copy_calc_results(struct)
            
            write(os.path.join(self.run_dir, "dataset.xyz"), self.data)

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
    

    def train_nep(self):
        train_dir = os.path.join(self.iter_dir, "nepmodel_split1")
        subprocess.run(["nep"], cwd=train_dir,
                       check=True, text=True)
    

    def _set_prediction_mode(self, nep_in, dataset=None):
        with open(nep_in, "r") as f:
            lines = f.readlines()

        lines = [l for l in lines if not l.strip().startswith(("prediction", "dataset"))]

        lines.append("prediction 1\n")
        if dataset is not None:
            lines.append(f"dataset {dataset}\n")

        with open(nep_in, "w") as f:
            f.writelines(lines)


    def run_prediction_mode(self):
        nep_dir = os.path.join(self.iter_dir, "nepmodel_split1")
        nep_in = os.path.join(nep_dir, "nep.in")

        self._set_prediction_mode(nep_in)

        print("Running NEP in prediction mode...", flush=True)
        subprocess.run(["nep"], cwd=nep_dir, check=True)

    def _maxvol(self, A, n_select, n_iter=20):

        N = A.shape[0]
        n_select = min(n_select, N)

        idx = np.random.choice(N, n_select, replace=False)

        for _ in range(n_iter):
            sub = A[idx]

            try:
                inv = np.linalg.pinv(sub)
            except np.linalg.LinAlgError:
                break

            scores = np.linalg.norm(A @ inv, axis=1)

            worst = np.argmax(scores)
            replace = np.argmin(scores[idx])

            idx[replace] = worst

        return np.unique(idx)

    def extract_descriptors(self):
        nep_dir = os.path.join(self.iter_dir, "nepmodel_split1")
        desc_file = os.path.join(nep_dir, "descriptor.out")

        if not os.path.exists(desc_file):
            raise RuntimeError("descriptor.out not found")

        A = np.loadtxt(desc_file)

        A = (A - A.mean(axis=0)) / (A.std(axis=0) + 1e-12)

        self.descriptors = A

        print(f"Loaded descriptor matrix: {A.shape}", flush=True)

    def build_active_set(self, n_active=200):
        # Extract descriptors for the dataset structures
        self.extract_descriptors()
        A = self.descriptors

        print("Running MaxVol for active set...", flush=True)
        idx = self._maxvol(A, n_active)

        asi_path = os.path.join(self.iter_dir, "active_set.asi")

        with open(asi_path, "w") as f:
            f.write("# Active set indices\n")
            for i in idx:
                f.write(f"{i}\n")

        self.active_set = idx

        print(f"Active set written: {len(idx)} environments", flush=True)



    def setup_MD(self, temperature=300, n_steps=5000, dump_interval=10):
        md_dir = os.path.join(self.iter_dir, "md")
        os.makedirs(md_dir, exist_ok=True)

        # Copy the trained NEP model to the MD directory
        nep_src = os.path.join(self.iter_dir, "nepmodel_split1", "nep.txt")
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
        replicate 5 5 5
        potential nep.txt

        velocity {temperature}
        time_step 1.0

        compute_extrapolation asi_file ../active_set.asi check_interval 10 gamma_low 5 gamma_high 10
        ensemble npt_mttk iso 0 0 temp {temperature} {temperature+200}
        run {n_steps}

        compute_extrapolation asi_file ../active_set.asi check_interval 10 gamma_low 5 gamma_high 10
        ensemble npt_mttk iso 0 0 temp {temperature+200} {temperature}
        run {n_steps}
        """

        with open(os.path.join(md_dir, "run.in"), "w") as f:
            f.write(run_in)


    def run_MD(self):
        md_dir = os.path.join(self.iter_dir, "md")
        print("Running GPUMD...")

        subprocess.run(
            ["gpumd"],
            cwd=md_dir,
            check=True
        )
    

    def extract_md_descriptors(self):

        md_dir = os.path.join(self.iter_dir, "md")
        nep_dir = os.path.join(self.iter_dir, "nepmodel_split1")

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
