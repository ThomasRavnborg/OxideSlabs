#!/bin/bash
#SBATCH --partition=h200
#SBATCH -N 1-1
#SBATCH -n 24
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --job-name=gpujob
#SBATCH --output=results/logs/slurm-%j.out

set -e

# Change to the directory where the job was submitted from
cd "$SLURM_SUBMIT_DIR"

# Set up environment variable for Python
export PYTHONPATH="$PWD:$PYTHONPATH"

# Purge existing modules and load the GPUMD module
module purge
module load GPUMD/v5.0-H200
# Check GPU status
nvidia-smi

# Get the script path from the command line argument
SCRIPT_PATH="scripts/$1"
# Run the SIESTA calculation using uv run
echo "Running uv run $SCRIPT_PATH"
uv run python "$SCRIPT_PATH"