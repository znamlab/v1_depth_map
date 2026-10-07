#!/bin/bash --login
#SBATCH --job-name=deconvolve-tau
#SBATCH --partition=ncpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

# Array version of deconvolve_tau.sbatch: one "SESSION [ARGUMENTS]" line of MANIFEST per task, e.g.
#   PZAH8.2h_S20230116 --project hey2_3d-vision_foodres_20220101 --tau 0.25 --check-only
#   sbatch --array=0-N deconvolve_tau_array.sbatch MANIFEST
set -euo pipefail
conda activate 2p-preprocess
export PYTHONUNBUFFERED=1
mapfile -t tasks < "$SLURM_SUBMIT_DIR/${1:?manifest}"
read -r session extra <<< "${tasks[$SLURM_ARRAY_TASK_ID]}"
# shellcheck disable=SC2086  # extra holds several arguments
srun python "$SLURM_SUBMIT_DIR/src/v1_depth_map/precompute_data/deconvolve_tau.py" "$session" $extra
