# Write a .txt file in /results
import os
dir = os.getcwd()
with open(os.path.join(dir, 'results', 'test.txt'), 'w') as f:
    f.write('This is a test file.')
