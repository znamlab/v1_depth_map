#!/bin/bash --login
#SBATCH --job-name=sftf-running-models
#SBATCH --partition=ncpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

# Running-speed dependence of the SF/TF preferences (see sftf_running_models.py), written to
# results/. Run from the run directory (src/ with cottage_analysis and v1_depth_map, logs/) with a
# manifest of argument lines ("prepare SESSION", "fit SESSION ROI" or "merge SESSION"):
#   sbatch --array=0-N%M sftf_running_models.sbatch MANIFEST
# (prepare needs more memory: add --mem=32G)
set -euo pipefail
conda activate v1_depth_map
export PYTHONPATH=$SLURM_SUBMIT_DIR/src${PYTHONPATH:+:$PYTHONPATH}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
mapfile -t tasks < "$SLURM_SUBMIT_DIR/${1:?manifest}"
# shellcheck disable=SC2086  # the line holds several arguments
srun python "$SLURM_SUBMIT_DIR/src/v1_depth_map/precompute_data/sftf_running_models.py" \
    ${tasks[$SLURM_ARRAY_TASK_ID]} --out-dir "$SLURM_SUBMIT_DIR/results" --n-jobs "$SLURM_CPUS_PER_TASK"
