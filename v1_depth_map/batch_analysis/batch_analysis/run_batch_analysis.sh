#!/bin/bash
#
#SBATCH --job-name=batch_analysis
#SBATCH --ntasks=1
#SBATCH --time=2:00:00
#SBATCH --mem=8G
#SBATCH --partition=ncpu
#SBATCH --mail-type=END,FAIL
#SBATCH --output="/camp/lab/znamenskiyp/home/users/hey2/codes/3d-vision-analysis-2p/v1_depth_map/batch_analysis/batch_analysis/logs/batch_analysis_%j.log"

. ~/.bash_profile
ml purge

ml Anaconda3/2020.07
source activate base

conda activate v1_depth_map
cd "/camp/lab/znamenskiyp/home/users/hey2/codes/3d-vision-analysis-2p/v1_depth_map/batch_analysis/batch_analysis/"
python analyze_all_sessions.py