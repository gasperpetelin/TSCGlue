#!/bin/bash
#SBATCH --job-name=timing
#SBATCH --time=12:00:00
#SBATCH --time-min=00:30:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=logs/timing_%A_%a.out
#SBATCH --error=logs/timing_%A_%a.err
#SBATCH --array=5

# Default setup:
# - single fold via SLURM array value 5
# - disk storage
# - n_jobs defaults to 16 via the environment override below
#
# Usage examples:
#   sbatch run_timing.slurm -m TSCGlue-17-4-26
#   sbatch run_timing.slurm -m TSCGlue-17-4-26 -d ArrowHead
#   sbatch --array=0-29 run_timing.slurm -m TSCGlue-17-4-26
#   N_JOBS=1 sbatch run_timing.slurm -m rocket-j1

set -x
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export PYTHONFAULTHANDLER=1

FOLD="${SLURM_ARRAY_TASK_ID:-5}"
JOBS="${N_JOBS:-16}"

uv run --no-sync run_timing.py --storage disk -j "${JOBS}" -f "${FOLD}" "$@"
