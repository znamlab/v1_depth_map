#!/bin/bash --login
#SBATCH --job-name=sftf-trace-variants
#SBATCH --partition=ncpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

# Grating and depth tuning from alternative traces, written to results/ (see
# sftf_trace_variants.py). Run from the run directory (src/ with cottage_analysis and
# v1_depth_map, logs/) with a manifest of "SESSION VARIANT" lines:
#   sbatch --array=0-N sftf_trace_variants.sbatch MANIFEST
set -euo pipefail
conda activate v1_depth_map
export PYTHONPATH=$SLURM_SUBMIT_DIR/src${PYTHONPATH:+:$PYTHONPATH}
# one process per CPU (joblib), each single-threaded
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
mapfile -t tasks < "$SLURM_SUBMIT_DIR/${1:?manifest}"
read -r session variant <<< "${tasks[$SLURM_ARRAY_TASK_ID]}"
srun python "$SLURM_SUBMIT_DIR/src/v1_depth_map/precompute_data/sftf_trace_variants.py" \
    "$session" "$variant" --out-dir "$SLURM_SUBMIT_DIR/results" --n-jobs "$SLURM_CPUS_PER_TASK"
