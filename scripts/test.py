# Write a .txt file in /results
import os
dir = os.getcwd()
# Go up one level from the current directory
dir = os.path.dirname(dir)
print(dir)
# Save numpy array in results directory
import numpy as np
results_dir = os.path.join(dir, 'results')
print(results_dir)
np.save(os.path.join(results_dir, 'test.npy'), np.array([1, 2, 3, 4, 5]))