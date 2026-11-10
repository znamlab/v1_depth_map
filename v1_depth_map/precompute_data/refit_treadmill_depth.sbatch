#!/bin/bash --login
#SBATCH --job-name=refit-treadmill-depth
#SBATCH --partition=ncpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

# Treadmill 1D depth refit with the fixed seeding (see refit_treadmill_depth.py). Never
# writes neurons_df: results go to results/SESSION/treadmill_depth_fit{suffix}.pickle.
# Run from the run directory (src/ holding the cottage_analysis and v1_depth_map packages
# and CODE_VERSION.txt, logs/) with a
# manifest of "SESSION VARIANT" lines:
#   sbatch --array=0-N refit_treadmill_depth.sbatch MANIFEST
set -euo pipefail
conda activate v1_depth_map
export PYTHONPATH=$SLURM_SUBMIT_DIR/src${PYTHONPATH:+:$PYTHONPATH}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
mapfile -t tasks < "$SLURM_SUBMIT_DIR/${1:?manifest}"
read -r session variant <<< "${tasks[$SLURM_ARRAY_TASK_ID]}"
srun python "$SLURM_SUBMIT_DIR/src/v1_depth_map/precompute_data/refit_treadmill_depth.py" \
    "$session" "$variant" --out-dir "$SLURM_SUBMIT_DIR/results"
