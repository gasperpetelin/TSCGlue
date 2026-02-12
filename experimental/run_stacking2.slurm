#!/bin/bash
#SBATCH --job-name=stacking2
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --output=stacking2_%j.out
#SBATCH --error=stacking2_%j.err

# Usage:
#   sbatch run_stacking2.slurm                          # run all models
#   sbatch run_stacking2.slurm loky-stacker-v6          # run a specific model
#   sbatch run_stacking2.slurm loky-stacker-v6,loky-stacker-v7  # run multiple models

uv run run_stacking2.py "$@"
