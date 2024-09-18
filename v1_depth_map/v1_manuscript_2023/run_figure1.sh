#!/bin/bash
#
#SBATCH --job-name=plot
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --partition=cpu
#SBATCH --mail-type=END,FAIL
#SBATCH --output="/camp/lab/znamenskiyp/home/users/hey2/codes/3d-vision-analysis-2p/v1_depth_analysis/v1_manuscript_2023/logs/plot_%j.log"

. ~/.bash_profile
ml purge

ml Anaconda3/2020.07
source activate base

conda activate 2p_analysis_cottage
cd "/camp/lab/znamenskiyp/home/users/hey2/codes/3d-vision-analysis-2p/v1_depth_analysis/v1_manuscript_2023/"
python figure1.py