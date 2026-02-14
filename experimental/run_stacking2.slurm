#!/bin/bash
#SBATCH --job-name=stacking2
#SBATCH --time=08:00:00
#SBATCH --time-min=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/stacking2_%A_%a.out
#SBATCH --error=logs/stacking2_%A_%a.err
#SBATCH --partition=e7
#SBATCH --reservation=e7
#SBATCH --array=0-29

# Folds are handled by SLURM array (0-29), storage is always disk.
# All other args are passed through to run_stacking2.py via "$@".
#
# Usage:
#   sbatch run_stacking2.slurm                                        # all models, all folds
#   sbatch run_stacking2.slurm -m loky-stacker-v6                     # specific model, all folds
#   sbatch run_stacking2.slurm -m loky-stacker-v6 -d Worms            # specific model + dataset

uv run run_stacking2.py --storage disk -j 4 -f $SLURM_ARRAY_TASK_ID "$@"
