#!/bin/bash
#SBATCH --job-name=reduction_methods
#SBATCH -o experiments/logs/%A_reduction_%a.log
#SBATCH -M ukko
#SBATCH --partition=short
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --array=1-10
#SBATCH --constraint=amd

module load Python/3.10.4-GCCcore-11.3.0
source venv/bin/activate
python3 experiments/compare_optimisation_schemes.py $SLURM_ARRAY_TASK_ID
