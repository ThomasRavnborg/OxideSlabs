#!/bin/bash
#SBATCH --partition=xeon24el8_test
#SBATCH -N 1
#SBATCH -n 24
#SBATCH --time=00:10:00
#SBATCH --job-name=siesta

set -e

cd "$SLURM_SUBMIT_DIR"

export PYTHONPATH="$PWD:$PYTHONPATH"
export GPAW_SETUP_PATH="$(dirname "$(readlink -f "$0")")/gpaw_data:$GPAW_SETUP_PATH"

module purge
#module load Siesta/5.4.0-foss-2024a
module load GPAW/25.7.0-foss-2025b

# === Input ===
if [ -z "$1" ]; then
    echo "Usage: sbatch slurm/run.sh test.py"
    exit 1
fi

SCRIPT_NAME="$1"
SCRIPT_PATH="scripts/$SCRIPT_NAME"

echo "argv: $@"
echo "SCRIPT_PATH='$SCRIPT_PATH'"
ls -l scripts

# === Hard checks ===
if [ -d "$SCRIPT_PATH" ]; then
    echo "Error: $SCRIPT_PATH is a directory"
    exit 1
fi

if [ ! -f "$SCRIPT_PATH" ]; then
    echo "Error: file not found: $SCRIPT_PATH"
    exit 1
fi

export ASE_SIESTA_COMMAND="mpirun siesta < PREFIX.fdf > PREFIX.out"

echo "Running uv run python $SCRIPT_PATH"
uv run python "$SCRIPT_PATH"
