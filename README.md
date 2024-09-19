# v1_depth_map
Analysis code for v1 depth map project.

## Installation
1. Create an empty environment `conda create --name <env_name>`
2. Activate the environment by `conda activate <env_name>`.
3. Install pip `conda install pip`.
3. Install this package by `pip install .`.

## Batch analysis
1. Run the bash script in each folder under `./v1_depth_map/batch_analysis` to conduct corresponding analysis for all sessions.
2. Remember to change the path in the bash script for `#SBATCH --output=` and `cd` to your local path to this repo.

## Figure plotting
Run the notebooks under `./v1_depth_map/figures` to plot the corresponding figures.