#!/bin/bash
#SBATCH --partition=xeon24el8
#SBATCH -N 1
#SBATCH -n 24
#SBATCH --time=24:00:00
#SBATCH --job-name=gpaw
#SBATCH --output=logs/slurm-%j.out

set -e

# Change to the directory where the job was submitted from
cd "$SLURM_SUBMIT_DIR"
# Set up environment variables for Python and GPAW
export PYTHONPATH="$PWD:$PYTHONPATH"
export GPAW_SETUP_PATH="$(dirname "$(readlink -f "$0")")/gpaw_data:$GPAW_SETUP_PATH"

# Purge existing modules and load the GPAW module
module purge
module load GPAW/25.7.0-foss-2025b

# Get the script path from the command line argument
SCRIPT_PATH="scripts/$1"

# Run the GPAW calculation using 24 processors
echo "Running gpaw -P 24 python $SCRIPT_PATH"
gpaw -P 24 python "$SCRIPT_PATH"