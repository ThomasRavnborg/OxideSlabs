#!/bin/bash
#SBATCH --partition=xeon24el8
#SBATCH -N 1
#SBATCH -n 24
#SBATCH --time=10:00:00

# Purge all loaded modules and load the Siesta module
module purge
module load Siesta/5.4.0-foss-2024a

# Set ASE_SIESTA_COMMAND
export ASE_SIESTA_COMMAND="mpirun siesta < PREFIX.fdf > PREFIX.out"

# Ensure SCRIPT_TO_RUN is set
if [ -z "$SCRIPT_TO_RUN" ]; then
    echo "Error: SCRIPT_TO_RUN is not set"
    exit 1
fi

# Run script from project root, specifying path
SCRIPT_PATH="scripts/$SCRIPT_TO_RUN"

echo "Running $SCRIPT_PATH"

uv run python "$SCRIPT_PATH"
