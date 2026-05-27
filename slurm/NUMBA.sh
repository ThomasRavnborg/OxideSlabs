#!/bin/bash
#SBATCH --partition=xeon24el8
#SBATCH -N 1
#SBATCH -n 24
#SBATCH --time=50:00:00
#SBATCH --job-name=numba
#SBATCH --output=results/logs/slurm-%j.out

set -e

# Change to the directory where the job was submitted from
cd "$SLURM_SUBMIT_DIR"
# Set up environment variables for Python
export PYTHONPATH="$PWD:$PYTHONPATH"
export NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Purge existing modules
module purge

# Get the script path from the command line argument
SCRIPT_PATH="scripts/$1"
# Run the SIESTA calculation using uv run
echo "Running uv run $SCRIPT_PATH"
uv run python "$SCRIPT_PATH"