#!/bin/bash --login
#SBATCH --job-name=refit-closedloop-depth
#SBATCH --partition=ncpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Closed-loop 1D depth refit of one session (see refit_closedloop_depth.py). Never writes
# neurons_df: results go to results/SESSION/closedloop_depth_fit.pickle. Run from the run
# directory (src/ holding the cottage_analysis and v1_depth_map packages, logs/):
#   sbatch refit_closedloop_depth.sbatch SESSION
set -euo pipefail
conda activate v1_depth_map
export PYTHONPATH=$SLURM_SUBMIT_DIR/src${PYTHONPATH:+:$PYTHONPATH}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
srun python "$SLURM_SUBMIT_DIR/src/v1_depth_map/precompute_data/refit_closedloop_depth.py" \
    "${1:?session}" --out-dir "$SLURM_SUBMIT_DIR/results"
