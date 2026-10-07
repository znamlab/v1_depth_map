#!/bin/bash --login
#SBATCH --job-name=rf-latency-sweep
#SBATCH --partition=ncpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Stimulus-lag sweep of the multidepth RF fit (see latency_sweep.py). Run from the run
# directory: sbatch latency_sweep.sbatch SESSION FIT OUT [EXTRA ARGUMENTS, e.g. --lags 0,1,2]
set -euo pipefail
conda activate v1_depth_map
export PYTHONPATH=$SLURM_SUBMIT_DIR/src${PYTHONPATH:+:$PYTHONPATH}
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK PYTHONUNBUFFERED=1
session=$1 fit=$2 out=$3
shift 3
srun python "$SLURM_SUBMIT_DIR/src/v1_depth_map/revisions/rf_spike_tau/latency_sweep.py" "$session" \
  --fit "$fit" --out "$out" "$@"
