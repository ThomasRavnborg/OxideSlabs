#!/bin/bash
#SBATCH --partition=h200
#SBATCH -N 1-1
#SBATCH -n 24
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --job-name=gpujob
#SBATCH --output=out.log

module load GPUMD/v3.9.5-H200
nvidia-smi

nep # Train using NEP executable