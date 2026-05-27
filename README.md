# v1_depth_map
Analysis code for v1 depth map project.

## Installation
1. Create an empty environment `conda create --name v1_depth_map "python==3.12"`
2. Activate the environment by `conda activate v1_depth_map`.
3. Install pip `conda install pip`.
3. Install this package by `pip install .` (add `[figures]` for plotting). 

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

## Re-downloading and Syncing Data / Database (for Lab Members)

If the database has been updated or you need to download/sync the figures and revisions dataset to a new local directory (e.g., an external drive like `/Volumes/BlackPasspo/v1_depth_map`), follow these steps:

### 1. Re-download and Merging the Database Snapshots
To update the offline registry (`offline_database.json`), you must be connected to the Crick VPN/network. Run:
```bash
python v1_depth_map/precompute_data/download_database.py
```
This script connects to live `flexilims` server, downloads up-to-date snapshots for both the figures project (`hey2_3d-vision_foodres_20220101`) and the revisions project (`colasa_3d-vision_revisions`), merges them, and saves the file to `<v1_depth_map_processed_root>/offline_database.json`.

### 2. Syncing Figures and Revisions Data
To sync or update the target raw and processed dataset directories (excluding massive raw video sequences and raw TIFF files to save time and space), use the automated sync utility:
```bash
python v1_depth_map/revisions/copy_v1_figures_data.py /path/to/destination --skip-existing
```
* **Arguments**:
  * `dest` (positional): The target directory where `raw` and `processed` data folders will be created (e.g., `/Volumes/BlackPasspo/v1_depth_map`).
  * `--skip-existing`: Skip copying files that already exist in the target directory to resume or accelerate a sync run.
  * `--dry-run`: Display all source-to-destination paths that would be copied without making any modifications.

