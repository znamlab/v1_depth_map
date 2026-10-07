#!/bin/bash --login
#SBATCH --job-name=rerun-depth-sftf
#SBATCH --partition=ncpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

# Grating fit rerun overwriting neurons_df_sftf.pickle after a backup (see rerun_depth_sftf.py).
# Run from the run directory (src/ with cottage_analysis and v1_depth_map, backups/, logs/)
# with a manifest of "SESSION [ARGUMENTS]" lines:
#   sbatch --array=0-N rerun_depth_sftf.sbatch MANIFEST
set -euo pipefail
conda activate v1_depth_map
export PYTHONPATH=$SLURM_SUBMIT_DIR/src${PYTHONPATH:+:$PYTHONPATH}
# one process per CPU (joblib), each single-threaded
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
mapfile -t tasks < "$SLURM_SUBMIT_DIR/${1:?manifest}"
read -r session extra <<< "${tasks[$SLURM_ARRAY_TASK_ID]}"
# shellcheck disable=SC2086  # extra holds several arguments
srun python "$SLURM_SUBMIT_DIR/src/v1_depth_map/precompute_data/rerun_depth_sftf.py" "$session" \
    --backup-dir "$SLURM_SUBMIT_DIR/backups" --n-jobs "$SLURM_CPUS_PER_TASK" $extra
