#!/bin/bash --login
# Lab Slurm submission wrapper (Crick NEMO cluster), kept for provenance: adapt partition, modules, environment and paths to another cluster.
#SBATCH --job-name=rsof-batched-prepare
#SBATCH --partition=ncpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

# Loads one session and submits its RS/OF fits as an array over ROI batches, plus a merge
# job (see refit_rsof_batched.py). Run from the run directory (src/ holding the
# cottage_analysis and v1_depth_map packages, CODE_VERSION.txt, logs/) with a manifest of
# "SESSION TARGET [ARGUMENTS]" lines:
#   sbatch --array=0-N refit_rsof_batched.sbatch MANIFEST
# Merged fits go to results/SESSION/, work directories to work/.
set -euo pipefail
conda activate v1_depth_map
export PYTHONPATH=$SLURM_SUBMIT_DIR/src${PYTHONPATH:+:$PYTHONPATH}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
mapfile -t tasks < "$SLURM_SUBMIT_DIR/${1:?manifest}"
read -r session target extra <<< "${tasks[$SLURM_ARRAY_TASK_ID]}"
# shellcheck disable=SC2086  # extra holds several arguments
srun python "$SLURM_SUBMIT_DIR/src/v1_depth_map/precompute_data/refit_rsof_batched.py" \
    "$session" "$target" --out-dir "$SLURM_SUBMIT_DIR/results" \
    --work-root "$SLURM_SUBMIT_DIR/work" $extra
