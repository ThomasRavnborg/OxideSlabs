#!/usr/bin/env bash
set -e

# === Config ===
CLUSTER="s214434@sylg.fysik.dtu.dk"
REMOTE_DIR="~/OxideSlabs"
JOB_SCRIPT="run.sh"       # your SLURM batch script

# Check if script argument was given
if [ -z "$1" ]; then
  echo "Usage: ./submit_remote.sh SCRIPT_NAME"
  echo "Example: ./submit_remote.sh hello.py"
  exit 1
fi
SCRIPT_TO_RUN="$1"

# === 1. Show git status locally ===
echo "== Git status =="
git status --short

read -p "Commit changes? (y/n) " ans
[[ "$ans" == "y" ]] || exit 1

# === 2. Commit local changes (if any) ===
# Stage all changes, including new files
git add -A

# Commit (if there are changes)
if ! git diff --cached --quiet; then
    git commit -m "$(date +'%Y-%m-%d %H:%M:%S')"
else
    echo "No changes to commit."
fi

# === 3. Push code to GitHub and cluster ===
echo
echo "== Pushing code =="
git push origin
git push cluster

# === 4. Submit job on cluster ===

echo
echo "== Submitting job on cluster =="
ssh "$CLUSTER" SCRIPT_TO_RUN="$SCRIPT_TO_RUN" << 'EOF'
  cd ~/OxideSlabs
  git pull

  SCRIPT_BASE=$(basename "$SCRIPT_TO_RUN" .py)

  sbatch \
  --job-name="$SCRIPT_BASE" \
  --output="$SCRIPT_BASE.log" \
  ~/slurm/run.sh

EOF

echo
echo "== Job submitted for script: $SCRIPT_TO_RUN =="
