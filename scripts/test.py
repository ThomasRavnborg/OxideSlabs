# Write a .txt file in /results
import os
dir = os.getcwd()
# Go up one level from the current directory
dir = os.path.dirname(dir)
print(dir)
print(os.path.join(dir, 'results'))
# Create .txt file in /results
with open(os.path.join(dir, 'results', 'test.txt'), 'w') as f:
    f.write('This is a test file.')