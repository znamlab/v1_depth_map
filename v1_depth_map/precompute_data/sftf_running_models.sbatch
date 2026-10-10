#!/bin/bash --login
#SBATCH --job-name=sftf-running-models
#SBATCH --partition=ncpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

# Running-speed dependence of the SF/TF preferences, frame-level model comparison (see
# sftf_running_models.py), written to results/. Run from the run directory (src/ with
# cottage_analysis and v1_depth_map, logs/) with a manifest of one session per line:
#   sbatch --array=0-N sftf_running_models.sbatch MANIFEST
set -euo pipefail
conda activate v1_depth_map
export PYTHONPATH=$SLURM_SUBMIT_DIR/src${PYTHONPATH:+:$PYTHONPATH}
# one process per CPU (joblib), each single-threaded
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
mapfile -t sessions < "$SLURM_SUBMIT_DIR/${1:?manifest}"
srun python "$SLURM_SUBMIT_DIR/src/v1_depth_map/precompute_data/sftf_running_models.py" \
    "${sessions[$SLURM_ARRAY_TASK_ID]}" --out-dir "$SLURM_SUBMIT_DIR/results" --n-jobs "$SLURM_CPUS_PER_TASK"
