#!/bin/bash
# Lab Slurm submission wrapper (Crick NEMO cluster), kept for provenance: adapt partition, modules, environment and paths to another cluster.
#
#SBATCH --job-name=rs_stats
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --mem=16G
#SBATCH --partition=ncpu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=logs/rs_%j.log

. ~/.bash_profile
ml purge

ml Anaconda3/2020.07
source activate base

conda activate v1_depth_map

cd "$SLURM_SUBMIT_DIR"  # submit from this folder (logs/ must exist)
python calculate_rs_stats.py
