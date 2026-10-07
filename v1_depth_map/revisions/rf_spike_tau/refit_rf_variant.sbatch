#!/bin/bash --login
#SBATCH --job-name=rf-refit-variant
#SBATCH --partition=ncpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --array=0-0
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

# Refit multidepth RFs with a variant of the pipeline fit (see refit_rf_variant.py),
# without touching neurons_df. Run from the run directory with a manifest of
# "SESSION [EXTRA ARGUMENTS]" lines, e.g.
#   PZAG16.3c_S20250317 --use-col spks
#   sbatch --array=0-N refit_rf_variant.sbatch MANIFEST OUT_DIR
set -euo pipefail

MANIFEST=${1:?manifest file}
OUT_DIR=${2:?output directory}
RUN_DIR=$SLURM_SUBMIT_DIR

conda activate v1_depth_map
export PYTHONPATH=$RUN_DIR/src${PYTHONPATH:+:$PYTHONPATH}
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK PYTHONUNBUFFERED=1

mapfile -t tasks < "$RUN_DIR/$MANIFEST"
read -r session extra <<< "${tasks[$SLURM_ARRAY_TASK_ID]}"
# shellcheck disable=SC2086  # extra holds several arguments
srun python "$RUN_DIR/src/v1_depth_map/revisions/rf_spike_tau/refit_rf_variant.py" "$session" \
  $extra --out "$RUN_DIR/$OUT_DIR"
