import os
import json
import pandas as pd
from glob import glob
import shutil

class SiestaProject:
    """A class to manage Siesta calculations for a given material.
    Handles folder structure, parameter normalization, ID assignment, and summary CSV.
    Each calculation gets a unique ID and a folder with subfolders for relax, bandstructure and phonons.
    The summary CSV tracks all calculations and their completion status.
    Methods:
    - normalize_params(params): Normalize calculation parameters for consistency.
    - _get_next_id(): Get the next available unique ID for a new calculation.
    - _find_existing_id(params): Check if a calculation with the same parameters already exists and return its ID.
    - _create_structure(calc_id, params): Create the folder structure for a calculation and save the parameters.
    - remove_calculation(calc_id): Remove a calculation by ID, including its folder and summary CSV entry.
    - _relax_completed(calc_id), _band_completed(calc_id), _phonon_completed(calc_id): Check if the respective calculation step is completed by looking for specific output files.
    - _update_summary(calc_id, params): Update the summary CSV with the status of a calculation.
    - display_summary(): Display the summary CSV in a readable format.
    - prepare_calculation(raw_params): Main entry point to prepare a calculation. Normalizes parameters, checks for existing calculations, creates structure if needed, and updates summary.
    - what_to_run(calc_id): Determine which calculation step should be run next based on completion status.
    """


    def __init__(self, material="BaTiO3", bulk=True):
        root = 'results'
        if bulk:
            sym = "bulk"
        else:
            sym = "slab"

        self.material = material
        self.material_path = os.path.join(root, sym, material)
        self.summary_file = os.path.join(self.material_path, "summary.csv")

        os.makedirs(self.material_path, exist_ok=True)


    # -----------------------------
    # Parameter handling
    # -----------------------------

    def normalize_params(self, params):
        normalized = {}

        for key, value in params.items():

            #if isinstance(value, tuple):
            #    normalized[key] = "x".join(str(v) for v in value)

            #elif isinstance(value, float):
            #    normalized[key] = f"{value:.6f}"

            #else:
            normalized[key] = str(value)

        return normalized


    # -----------------------------
    # ID handling
    # -----------------------------

    def _get_next_id(self):
        """
        Find the first available ID in the sequence.
        IDs are 4-digit zero-padded strings (0001, 0002, ...)
        """
        if not os.path.exists(self.summary_file):
            return "0001"

        df = pd.read_csv(self.summary_file, dtype=str)

        used_ids = sorted([int(x) for x in df["ID"].values])

        # Start at 1 and find the first gap
        next_id = 1
        for uid in used_ids:
            if uid == next_id:
                next_id += 1
            else:
                break

        return f"{next_id:04d}"

    def _find_existing_id(self, params):

        if not os.path.exists(self.summary_file):
            return None

        df = pd.read_csv(self.summary_file, dtype=str)

        mask = True
        for key, value in params.items():
            if key not in df.columns:
                return None
            mask &= (df[key] == value)

        matches = df[mask]

        if len(matches) > 0:
            return matches.iloc[0]["ID"]

        return None


    # -----------------------------
    # Folder structure
    # -----------------------------

    def _create_structure(self, calc_id, params):

        calc_path = os.path.join(self.material_path, calc_id)

        for sub in ["relax", "bands", "phonons"]:
            os.makedirs(os.path.join(calc_path, sub), exist_ok=True)

        # Write parameter file
        with open(os.path.join(calc_path, "parameters.json"), "w") as f:
            json.dump(params, f, indent=4)

        return calc_path

    def remove_calculation(self, calc_id):
        """
        Remove a calculation by ID.
        - Deletes the folder entirely
        - Deletes the row in the summary CSV
        """
        
        # Remove folder
        folder = os.path.join(self.material_path, calc_id)
        if os.path.exists(folder):
            shutil.rmtree(folder)
            print(f"[{calc_id}] Folder removed.")
        else:
            print(f"[{calc_id}] Folder does not exist.")

        # Remove row from CSV
        if os.path.exists(self.summary_file):
            df = pd.read_csv(self.summary_file, dtype=str)
            if calc_id in df["ID"].values:
                df = df[df["ID"] != calc_id]
                df.to_csv(self.summary_file, index=False)
                print(f"[{calc_id}] Row removed from summary CSV.")
            else:
                print(f"[{calc_id}] Not found in summary CSV.")
        else:
            print("Summary CSV does not exist.")

    # -----------------------------
    # Completion checks
    # -----------------------------

    def _relax_completed(self, calc_id):
        filepath = os.path.join(
            self.material_path, calc_id, "relax", f"{self.material}.xyz"
        )
        return os.path.exists(filepath)
    
    def _band_completed(self, calc_id):
        filepath = os.path.join(
            self.material_path, calc_id, "bands", f"{self.material}.bands"
        )
        return os.path.exists(filepath)

    def _phonon_completed(self, calc_id):
        filepath = os.path.join(
            self.material_path, calc_id, "phonons", f"{self.material}.yaml"
        )
        return os.path.exists(filepath)


    # -----------------------------
    # CSV handling
    # -----------------------------

    def _update_summary(self, calc_id, params):

        params = self.normalize_params(params)

        relax = str(self._relax_completed(calc_id))
        band = str(self._band_completed(calc_id))
        phonon = str(self._phonon_completed(calc_id))

        row = {
            "ID": calc_id,
            **params,
            "relax": relax,
            "bands": band,
            "phonons": phonon,
        }

        new_df = pd.DataFrame([row], dtype=str)

        if os.path.exists(self.summary_file):

            df = pd.read_csv(self.summary_file, dtype=str)

            if calc_id in df["ID"].values:
                df.set_index("ID", inplace=True)
                new_df.set_index("ID", inplace=True)
                df.update(new_df)
                df.reset_index(inplace=True)
            else:
                df = pd.concat([df, new_df], ignore_index=True)

        else:
            df = new_df

        df.to_csv(self.summary_file, index=False)

    def get_summary(self):
        if not os.path.exists(self.summary_file):
            print("No summary yet")
            return
        
        df = pd.read_csv(self.summary_file, dtype=str)
        df_display = df.copy()
        
        # Add unit (Ry) to energy-related columns
        for col in ['EnergyShift', 'MeshCutoff']:
            df_display[col] = df_display[col].astype(str) + " Ry"

        for col in ["relax", "bands", "phonons"]:
            df_display[col] = df_display[col].map({"True": "✓", "False": "✗"})
        # Sort by ID
        return df_display.sort_values(by="ID")


    # -----------------------------
    # Main entry point
    # -----------------------------

    def prepare_calculation(self, raw_params):

        params = self.normalize_params(raw_params)

        existing_id = self._find_existing_id(params)

        if existing_id:
            calc_id = existing_id
        else:
            calc_id = self._get_next_id()
            self._create_structure(calc_id, params)

        self._update_summary(calc_id, params)

        return calc_id


    # -----------------------------
    # What should run next?
    # -----------------------------

    def what_to_run(self, calc_id):

        relax_done = self._relax_completed(calc_id)
        band_done = self._band_completed(calc_id)
        phonon_done = self._phonon_completed(calc_id)

        if not relax_done:
            return "relax"

        if not band_done:
            return "bands"

        if not phonon_done:
            return "phonons"

        return "complete"
