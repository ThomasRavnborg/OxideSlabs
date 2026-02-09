#!/bin/bash
#SBATCH --partition=xeon24el8
#SBATCH -N 1
#SBATCH -n 24
#SBATCH --time=10:00:00

cd "$SLURM_SUBMIT_DIR"

module purge
module load Siesta/5.4.0-foss-2024a

mkdir -p results

SCRIPT_NAME=$(basename "$SCRIPT_TO_RUN" .py)
SCRIPT_FILE="$SLURM_SUBMIT_DIR/scripts/$SCRIPT_TO_RUN"
LOG_FILE="results/${SCRIPT_NAME}.log"

echo "Running $SCRIPT_FILE → $LOG_FILE"
uv run python "$SCRIPT_FILE" > "$LOG_FILE" 2>&1
echo "Finished $SCRIPT_FILE"