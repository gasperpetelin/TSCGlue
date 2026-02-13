#!/bin/bash
#SBATCH --job-name=stacking2
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --output=stacking2_%j.out
#SBATCH --error=stacking2_%j.err
#SBATCH --reservation=e7
#SBATCH --array=0-29

# Folds are handled by SLURM array (0-29), storage is always disk.
# All other args are passed through to run_stacking2.py via "$@".
#
# Usage:
#   sbatch run_stacking2.slurm                                        # all models, all folds
#   sbatch run_stacking2.slurm -m loky-stacker-v6                     # specific model, all folds
#   sbatch run_stacking2.slurm -m loky-stacker-v6 -d Worms            # specific model + dataset

uv run run_stacking2.py --storage disk -f $SLURM_ARRAY_TASK_ID "$@"
