#!/bin/bash
#
#SBATCH --job-name=batch_preprocess
#SBATCH --ntasks=1
#SBATCH --time=2:00:00
#SBATCH --mem=4G
#SBATCH --partition=ncpu
#SBATCH --mail-type=END,FAIL
#SBATCH --output="/camp/lab/znamenskiyp/home/users/hey2/codes/3d-vision-analysis-2p/v1_depth_map/batch_analysis/batch_preprocess/logs/batch_preprocess_%j.log"

ml purge
ml Anaconda3/2022.05

source activate base

conda activate 2p-preprocess
cd "/camp/lab/znamenskiyp/home/users/hey2/codes/3d-vision-analysis-2p/v1_depth_map/batch_analysis/batch_preprocess/"
echo Processing ${SESSION} in project ${PROJECT}
python twop_preprocess_all_sessions.py