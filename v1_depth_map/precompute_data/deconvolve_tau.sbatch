#!/bin/bash --login
#SBATCH --job-name=deconvolve-tau
#SBATCH --partition=ncpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Deconvolve annotated suite2p traces with a new tau (see deconvolve_tau.py), in the
# 2p-preprocess env. Run from the run directory:
#   sbatch deconvolve_tau.sbatch SESSION --tau 1.1 [--out-dir DIR | --backup-dir DIR] [--compare]
set -euo pipefail
conda activate 2p-preprocess
export PYTHONUNBUFFERED=1
srun python "$SLURM_SUBMIT_DIR/src/v1_depth_map/precompute_data/deconvolve_tau.py" "$@"
