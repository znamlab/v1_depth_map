#!/bin/bash --login
#SBATCH --job-name=sftf-trial-avg-subsets
#SBATCH --partition=ncpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

# Fit SFTF base model by trial average across static, run_slow (5-50 cm/s), and run_fast (>=50 cm/s).
# Run from run directory (with src/, logs/, results/, manifest.txt):
#   sbatch --array=0-5 sftf_trial_average_subsets.sbatch manifest.txt [OUT_DIR]

set -euo pipefail
conda activate v1_depth_map
export PYTHONPATH=$SLURM_SUBMIT_DIR/src${PYTHONPATH:+:$PYTHONPATH}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1

mapfile -t sessions < "$SLURM_SUBMIT_DIR/${1:?manifest}"
SESSION="${sessions[$SLURM_ARRAY_TASK_ID]}"
OUT_DIR="${2:-$SLURM_SUBMIT_DIR/results}"

echo "Starting trial-average subset fits for $SESSION on $(hostname) with $SLURM_CPUS_PER_TASK CPUs..."
srun python "$SLURM_SUBMIT_DIR/src/v1_depth_map/precompute_data/sftf_trial_average_subsets.py" run "$SESSION" \
    --out-dir "$OUT_DIR" --n-jobs "$SLURM_CPUS_PER_TASK"
echo "Finished $SESSION successfully."
