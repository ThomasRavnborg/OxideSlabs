import os
import pandas as pd

def pack_results(dir):
    q_folders = os.listdir(dir)
    q_folders = [d for d in q_folders if os.path.isdir(os.path.join(dir, d))]
    #N = len(q_folders)

    results = {}
    for qpoint in q_folders:
        results[qpoint] = {}
        # Find all mode folders from the given q-point folder
        dir_q = os.path.join(dir, qpoint)
        mode_folders = os.listdir(dir_q)
        for g_id in mode_folders:
            results[qpoint][g_id] = {}
            dir_mode = os.path.join(dir_q, g_id)
            try:
                # Read the frequency from the freq.txt file in the mode folder
                with open(os.path.join(dir_mode, 'freq.txt'), 'r') as f:
                    freq = f.read()
            except FileNotFoundError:
                # If no frequency is found, use the mode name as the label
                freq = g_id
            # Find all degenerate folders in the mode folder
            deg_folders = os.listdir(dir_mode)
            # Remove any files, so we only have folders
            deg_folders = [d for d in deg_folders if os.path.isdir(os.path.join(dir_mode, d))]
            #print(deg_folders)
            for mode_id in deg_folders:
                df = pd.read_csv(os.path.join(dir_mode, mode_id, 'energies.csv'))
                results[qpoint][g_id][mode_id] = {"df": df, 'frequency': freq}
    return results