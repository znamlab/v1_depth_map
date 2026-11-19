#!/bin/bash
# Lab Slurm submission wrapper (Crick NEMO cluster), kept for provenance: adapt partition, modules, environment and paths to another cluster.
#
#SBATCH --job-name=batch_preprocess
#SBATCH --ntasks=1
#SBATCH --time=2:00:00
#SBATCH --mem=4G
#SBATCH --partition=ncpu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=logs/batch_preprocess_%j.log

ml purge
ml Anaconda3/2022.05

source activate base

conda activate 2p-preprocess
cd "$SLURM_SUBMIT_DIR"  # submit from this folder (logs/ must exist)
echo Processing ${SESSION} in project ${PROJECT}
python twop_preprocess_all_sessions.py
