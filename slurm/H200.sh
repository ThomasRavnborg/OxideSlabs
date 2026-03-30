#!/bin/bash
#SBATCH --partition=h200
#SBATCH -N 1-1
#SBATCH -n 24
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --job-name=gpujob
#SBATCH --output=out.log

# Purge existing modules and load the GPUMD module
module purge
module load GPUMD/v5.0-H200

nvidia-smi

nep # Train using NEP executable