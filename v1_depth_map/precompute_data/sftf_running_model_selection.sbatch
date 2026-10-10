#!/bin/bash --login
#SBATCH --job-name=sftf-model-selection
#SBATCH --partition=ncpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

# Running-speed model selection on 10 high-SNR cells (see sftf_running_model_selection.py).
# Run from run directory with src/ (cottage_analysis and v1_depth_map) and logs/:
#   sbatch --array=0-9 sftf_running_model_selection.sbatch MANIFEST
set -euo pipefail
conda activate v1_depth_map
export PYTHONPATH=$SLURM_SUBMIT_DIR/src${PYTHONPATH:+:$PYTHONPATH}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
mapfile -t tasks < "$SLURM_SUBMIT_DIR/${1:?manifest}"
srun python "$SLURM_SUBMIT_DIR/src/v1_depth_map/precompute_data/sftf_running_model_selection.py" \
    ${tasks[$SLURM_ARRAY_TASK_ID]} --out-dir "$SLURM_SUBMIT_DIR/results" --n-jobs "$SLURM_CPUS_PER_TASK"
