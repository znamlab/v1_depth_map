#!/bin/bash
#
#SBATCH --job-name=batch_analysis
#SBATCH --ntasks=1
#SBATCH --time=2:00:00
#SBATCH --mem=8G
#SBATCH --partition=ncpu
#SBATCH --mail-type=END,FAIL
#SBATCH --output="/camp/lab/znamenskiyp/home/users/hey2/codes/3d-vision-analysis-2p/v1_depth_analysis/v1_manuscript_2023/batch_analysis_separate_recordings/logs/batch_analysis_%j.log"

. ~/.bash_profile
ml purge

ml Anaconda3/2020.07
source activate base

conda activate 2p_analysis_cottage2
cd "/camp/lab/znamenskiyp/home/users/hey2/codes/3d-vision-analysis-2p/v1_depth_analysis/v1_manuscript_2023/batch_analysis_separate_recordings/"
python analyze_all_sessions_separate_recordings.py