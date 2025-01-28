#!/bin/bash
#SBATCH --job-name=hyperparams
#SBATCH -o experiments/logs/%A_hyperparams_%a.log
#SBATCH -M ukko
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -G 1
#SBATCH --time=16:00:00
#SBATCH --array=1-10

module load Python/3.10.4-GCCcore-11.3.0
source venv/bin/activate
python3 experiments/hyperparameters.py $SLURM_ARRAY_TASK_ID
