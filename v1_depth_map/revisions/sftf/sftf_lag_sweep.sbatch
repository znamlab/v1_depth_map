#!/bin/bash --login
#SBATCH --job-name=sftf-lag-sweep
#SBATCH --partition=ncpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

# Lag sweep of the frame-level SF/TF fits (see sftf_lag_sweep.py), written to results/. Run from the run
# directory (src/ with cottage_analysis and v1_depth_map, logs/) with a manifest of argument lines
# ("prepare SESSION", "fit SESSION ROI" or "merge"):
#   sbatch --array=0-N%M sftf_lag_sweep.sbatch MANIFEST
# (prepare needs more memory: add --mem=32G)
set -euo pipefail
conda activate v1_depth_map
export PYTHONPATH=$SLURM_SUBMIT_DIR/src${PYTHONPATH:+:$PYTHONPATH}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
mapfile -t tasks < "$SLURM_SUBMIT_DIR/${1:?manifest}"
OUT_DIR="${2:-$SLURM_SUBMIT_DIR/results}"
# shellcheck disable=SC2086  # the line holds several arguments
srun python "$SLURM_SUBMIT_DIR/src/v1_depth_map/revisions/sftf/sftf_lag_sweep.py" \
    ${tasks[$SLURM_ARRAY_TASK_ID]} --out-dir "$OUT_DIR" --n-jobs "$SLURM_CPUS_PER_TASK"
