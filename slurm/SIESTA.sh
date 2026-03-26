#!/bin/bash
#SBATCH --partition=xeon24el8
#SBATCH -N 1
#SBATCH -n 24
#SBATCH --time=24:00:00
#SBATCH --job-name=siesta
#SBATCH --output=logs/slurm-%j.out

set -e

# Change to the directory where the job was submitted from
cd "$SLURM_SUBMIT_DIR"
# Set up environment variables for Python and GPAW
export PYTHONPATH="$PWD:$PYTHONPATH"
export ASE_SIESTA_COMMAND="mpirun siesta < PREFIX.fdf > PREFIX.out"

# Purge existing modules and load the SIESTA module
module purge
module load Siesta/5.4.0-foss-2024a

# Get the script path from the command line argument
SCRIPT_PATH="scripts/$1"
# Run the SIESTA calculation using uv run
echo "Running uv run $SCRIPT_PATH"
uv run python "$SCRIPT_PATH"