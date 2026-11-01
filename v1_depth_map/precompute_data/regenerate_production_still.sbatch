#!/bin/bash --login
#SBATCH --job-name=regen-still
#SBATCH --partition=ncpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

# Regenerate the production decoder_results.pickle stationary bin on NEMO (see
# regenerate_production_still.py), one session per array task. Run from the run
# directory, with a src/ checkout of v1_depth_map and cottage_analysis and a manifest of
# session names:
#   sbatch --array=0-N%M regenerate_production_still.sbatch MANIFEST
# Originals are kept as decoder_results.pickle.pre_still_fix_20261031_backup next to the
# new file; per-session summaries go to csv/SESSION.csv.
set -euo pipefail
conda activate v1_depth_map
export PYTHONPATH=$SLURM_SUBMIT_DIR/src/v1_depth_map:$SLURM_SUBMIT_DIR/src/cottage_analysis${PYTHONPATH:+:$PYTHONPATH}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
mapfile -t tasks < "$SLURM_SUBMIT_DIR/${1:?manifest}"
session=${tasks[$SLURM_ARRAY_TASK_ID]}
projects=/camp/lab/znamenskiyp/home/shared/projects
mkdir -p "$SLURM_SUBMIT_DIR/csv"
echo "$session" > "$TMPDIR/manifest.txt"
python -c "import cottage_analysis, v1_depth_map; print(cottage_analysis.__file__, v1_depth_map.__file__)"
srun python "$SLURM_SUBMIT_DIR/src/v1_depth_map/v1_depth_map/precompute_data/regenerate_production_still.py" \
    "$TMPDIR/manifest.txt" --fold-root "$projects/hey2_3d-vision_foodres_20220101" \
    --backup-suffix .pre_still_fix_20261031_backup --csv "$SLURM_SUBMIT_DIR/csv/$session.csv" \
    | awk '{ print strftime("%F %T"), $0; fflush() }'
