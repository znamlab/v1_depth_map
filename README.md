# v1_depth_map
Analysis code for v1 depth map project.

## Installation
1. Create an empty environment `conda create --name <env_name>`
2. Activate the environment by `conda activate <env_name>`.
3. Install pip `conda install pip`.
3. Install this package by `pip install .`.

## Setting up for offline uses of database
1. Follow installation instructures for https://github.com/znamlab/flexiznam.git. 
2. Set up config for the `flexiznam` package by `flexiznam config`. The config file should be found at `~/.flexiznam/config.yml`.
3. Make sure that the path under project_paths points to the path you have downloaded the data. e.g.:
```
project_paths:
    hey2_3d-vision_foodres_20220101:
        processed: your_data_path
        raw: your_data_path
```
4. Turn on the offline mode by adding the following to your config:
```
offline_mode: true
offline_yaml: offline_database.json
```

## Figure plotting
1. Run the notebooks under `./v1_depth_map/figures` to plot the corresponding figures.
2. When first running the figure notebooks, change `reload` to `True` to reload data. 

## Batch analysis
1. Run the bash script in each folder under `./v1_depth_map/batch_analysis` to conduct corresponding analysis for all sessions.
2. Remember to change the path in the bash script for `#SBATCH --output=` and `cd` to your local path to this repo.
3. Remember to change the conda environment name to your own environment.

## Precompute data
1. To precompute data for plotting figures, run the corresponding bash script.