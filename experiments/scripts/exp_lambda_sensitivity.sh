#!/bin/bash
#SBATCH --job-name=reduction_sensitivity
#SBATCH -o experiments/logs/%A_%a_sensitivity.log
#SBATCH --partition=short
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --array=1-10
#SBATCH --constraint=amd

module load Python/3.10.4-GCCcore-11.3.0
source .venv/bin/activate
python3 experiments/lambda_sensitivity.py $SLURM_ARRAY_TASK_ID
