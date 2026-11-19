#!/bin/bash
# Lab Slurm submission wrapper (Crick NEMO cluster), kept for provenance: adapt partition, modules, environment and paths to another cluster.
#
#SBATCH --job-name=figure
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --partition=ncpu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=logs/psth_%j.log

. ~/.bash_profile
ml purge

ml Anaconda3/2020.07
source activate base

conda activate v1_depth_map

cd "$SLURM_SUBMIT_DIR"  # submit from this folder (logs/ must exist)
python make_depth_tuning_raster.py
