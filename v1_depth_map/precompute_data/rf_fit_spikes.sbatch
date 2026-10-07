#!/bin/bash --login
#SBATCH --job-name=rf-fit-spikes
#SBATCH --partition=ncpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

# RF fits on spikes saved as neurons_df_spks.pickle (see rf_fit_spikes.py). Run from the run
# directory with a manifest of "SESSION [ARGUMENTS]" lines:
#   sbatch --array=0-N rf_fit_spikes.sbatch MANIFEST
set -euo pipefail
conda activate v1_depth_map
export PYTHONPATH=$SLURM_SUBMIT_DIR/src${PYTHONPATH:+:$PYTHONPATH}
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK PYTHONUNBUFFERED=1
mapfile -t tasks < "$SLURM_SUBMIT_DIR/${1:?manifest}"
read -r session extra <<< "${tasks[$SLURM_ARRAY_TASK_ID]}"
# shellcheck disable=SC2086  # extra holds several arguments
srun python "$SLURM_SUBMIT_DIR/src/v1_depth_map/precompute_data/rf_fit_spikes.py" "$session" $extra
