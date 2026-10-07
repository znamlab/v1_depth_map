#!/bin/bash --login
#SBATCH --job-name=tau-isolated-transients
#SBATCH --partition=ncpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

# Decay tau of isolated transients (see tau_isolated_transients.py). Run from the run
# directory with a manifest of "SESSION [ARGUMENTS]" lines, e.g.
#   PZAH8.2h_S20230116 --project hey2_3d-vision_foodres_20220101
#   sbatch --array=0-N tau_isolated_transients.sbatch MANIFEST OUT_DIR
set -euo pipefail
conda activate 2p-preprocess
export PYTHONPATH=$SLURM_SUBMIT_DIR/src${PYTHONPATH:+:$PYTHONPATH}
export PYTHONUNBUFFERED=1
mapfile -t tasks < "$SLURM_SUBMIT_DIR/${1:?manifest}"
read -r session extra <<< "${tasks[$SLURM_ARRAY_TASK_ID]}"
# shellcheck disable=SC2086  # extra holds optional arguments
srun python "$SLURM_SUBMIT_DIR/src/v1_depth_map/revisions/rf_spike_tau/tau_isolated_transients.py" \
  "$session" --out "$SLURM_SUBMIT_DIR/${2:?out dir}" $extra
