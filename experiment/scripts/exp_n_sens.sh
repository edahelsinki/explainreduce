#!/bin/bash
#SBATCH --job-name=reduction_n_sensitivity
#SBATCH -o experiments/logs/%A_%a_n-sensitivity.log
#SBATCH -M ukko
#SBATCH --partition=short
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=8:00:00
#SBATCH --array=1-10
#SBATCH --constraint=amd

module load Python/3.10.4-GCCcore-11.3.0
source venv/bin/activate
python3 experiments/subsample_sensitivity.py $SLURM_ARRAY_TASK_ID
