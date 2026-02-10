#!/bin/bash
#SBATCH --partition=xeon24el8
#SBATCH -N 1
#SBATCH -n 24
#SBATCH --time=10:00:00

# Purge all loaded modules and load the Siesta module
module purge
module load Siesta/5.4.0-foss-2024a

# Check if SCRIPT_TO_RUN is passed as an argument
if [ -z "$1" ]; then
    echo "Usage: sbatch run.sh SCRIPT.py"
    exit 1
fi
SCRIPT_TO_RUN="$1"

# Normalize script path to ensure it starts with "scripts/"
if [[ "$SCRIPT_TO_RUN" != scripts/* ]]; then
    SCRIPT_TO_RUN="scripts/$SCRIPT_TO_RUN"
fi

# Set ASE_SIESTA_COMMAND to use mpirun with Siesta
export ASE_SIESTA_COMMAND="mpirun siesta < PREFIX.fdf > PREFIX.out"
# Run the specified script using uv
echo "Running $SCRIPT_TO_RUN"
uv run python "$SCRIPT_TO_RUN"