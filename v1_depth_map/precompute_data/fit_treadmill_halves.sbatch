#!/bin/bash --login
#SBATCH --job-name=treadmill-halves
#SBATCH --partition=ncpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

# Fig 3 K-M control: treadmill trial-average RS/OF fits on the first / second half of each
# trial (see fit_treadmill_halves.py). Never writes the session folders or neurons_df.
# Run from the run directory (src/ holding cottage_analysis and v1_depth_map, logs/) with a
# manifest of "SESSION VARIANT" lines:
#   sbatch --array=0-N fit_treadmill_halves.sbatch MANIFEST
set -euo pipefail
conda activate v1_depth_map
export PYTHONPATH=$SLURM_SUBMIT_DIR/src${PYTHONPATH:+:$PYTHONPATH}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
mapfile -t tasks < "$SLURM_SUBMIT_DIR/${1:?manifest}"
read -r session variant <<< "${tasks[$SLURM_ARRAY_TASK_ID]}"
srun python "$SLURM_SUBMIT_DIR/src/v1_depth_map/precompute_data/fit_treadmill_halves.py" \
    "$session" "$variant" --out-dir "$SLURM_SUBMIT_DIR/results"
