#!/bin/bash --login
#SBATCH --job-name=rf-sd-lag-sweep
#SBATCH --partition=ncpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Single-depth snapshot lag sweep (see single_depth_lag_sweep.py). Run from the run directory:
#   sbatch single_depth_lag_sweep.sbatch SESSION OUT LAGS [--use-col spks]
set -euo pipefail
conda activate v1_depth_map
export PYTHONPATH=$SLURM_SUBMIT_DIR/src${PYTHONPATH:+:$PYTHONPATH}
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK PYTHONUNBUFFERED=1
srun python "$SLURM_SUBMIT_DIR/src/v1_depth_map/revisions/rf_spike_tau/single_depth_lag_sweep.py" "$1" --out "$2" --lags "$3" "${@:4}"
