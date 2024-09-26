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
3. Remember to change the conda environment name to your own environment.

## Figure plotting
1. Run the notebooks under `./v1_depth_map/figures` to plot the corresponding figures.
2. When first running the figure notebooks, change `reload` to `True` to reload data. 

## Precompute data
1. To precompute data for plotting figures, run the corresponding bash script.