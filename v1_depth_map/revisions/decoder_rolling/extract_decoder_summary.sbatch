#!/bin/bash --login
#SBATCH --job-name=decoder-summary
#SBATCH --partition=ncpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Summarize the NEMO decoder_results.pickle files (Nov 16 2026 rerun, no rolling average).
# Run from the run directory holding extract_decoder_summary.py, sessions.txt and logs/:
#   sbatch extract_decoder_summary.sbatch
set -euo pipefail
conda activate v1_depth_map
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
srun python -W ignore "$SLURM_SUBMIT_DIR/extract_decoder_summary.py" \
    --processed /nemo/lab/znamenskiyp/home/shared/projects/hey2_3d-vision_foodres_20220101 \
    --sessions "$SLURM_SUBMIT_DIR/sessions.txt" \
    --out "$SLURM_SUBMIT_DIR/new_norolling_summary.pickle"
