#!/bin/bash
#
#SBATCH --job-name=batch_size
#SBATCH --ntasks=1
#SBATCH --time=2:00:00
#SBATCH --mem=4G
#SBATCH --partition=ncpu
#SBATCH --mail-type=END,FAIL
#SBATCH --output="/camp/lab/znamenskiyp/home/users/hey2/codes/3d-vision-analysis-2p/v1_depth_map/batch_analysis/batch_size_control/logs/batch_analysis_%j.log"

. ~/.bash_profile
ml purge

ml Anaconda3/2020.07
source activate base

conda activate v1_depth_map
cd "/camp/lab/znamenskiyp/home/users/hey2/codes/3d-vision-analysis-2p/v1_depth_map/batch_analysis/batch_size_control/"
python size_control_all_sessions.py