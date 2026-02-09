#!/bin/bash
#SBATCH --partition=xeon24el8
#SBATCH -N 1
#SBATCH -n 24
#SBATCH --time=10:00:00

# Purge all loaded modules and load the Siesta module
module purge
module load Siesta/5.4.0-foss-2024a
# Set the ASE_SIESTA_COMMAND environment variable to run Siesta with mpirun
export ASE_SIESTA_COMMAND="mpirun siesta < PREFIX.fdf > PREFIX.out"

# Go into scripts folder
cd scripts
# Run the Python script passed in the environment
uv run python "$SCRIPT_TO_RUN"