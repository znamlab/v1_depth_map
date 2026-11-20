# v1_depth_map

Analysis and figure code for "A depth map of visual space in the primary visual cortex"
(Znamenskiy lab, The Francis Crick Institute).

This repository draws every figure and computes every number of the manuscript from the
accompanying data package. The core analysis library (fits,
synchronisation, plotting helpers) is
[`cottage_analysis`](https://github.com/znamlab/cottage_analysis); this repository holds the
figure notebooks, the project-specific helpers they use, and the scripts that produced the
processed data.

## Three levels of reproduction

There are three ways to use this code, from quick to very slow. Each level starts from the
output of the one below it, and all of those outputs are in the data package, so you can
stop at whichever level you need.

| Level | What it does | Reads | Writes | Cost |
|---|---|---|---|---|
| 1. Draw the figures | Runs the figure notebooks on the shipped caches | `analysis/` (and a few example sessions in `raw/`, `processed/`) | figures in `analysis/`, statistics in this repository | 1 to 2 hours for all figures on a laptop, up to 20 GB of RAM |
| 2. Regenerate the caches | Rebuilds the notebook caches from the per-session data | `processed/`, `raw/` | `analysis/` | Laptop; several hours, more with the deep option (see below) |
| 3. Re-analyse | Reruns preprocessing and the model fits | `raw/`, `processed/` | `processed/` | Slow; written for a Slurm cluster |

## Installation

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/znamlab/v1_depth_map.git
cd v1_depth_map
uv sync --extra figure
```

Some notebooks convert SVG to PDF with `rsvg-convert` (part of librsvg: `brew install
librsvg` or `apt install librsvg2-bin`).

The figures were made with Arial, which cannot be redistributed. If matplotlib cannot find
it, a fallback font is used and a warning is printed, so text sizes can differ slightly. To use your own
copy, set `COTTAGE_FONT_DIR` to a folder containing the Arial `.ttf` files.

## Data package

The data package has the following layout:

```
v1_depth_map_data/
  README.md                 contents, session list, file formats
  offline_database.json     snapshot of the lab database: mice, sessions, recordings, datasets
  config_template.yml       flexiznam configuration to copy and edit
  raw/<project>/<mouse>/<session>/<recording>/         stimulus and behaviour logs
  processed/<project>/<mouse>/<session>/               per-session results (inputs of level 2)
  analysis/                 notebook caches and figure outputs (output of levels 1 and 2)
```

There are two projects: `hey2_3d-vision_foodres_20220101` (main dataset) and
`colasa_3d-vision_revisions` (multi-day, multi-depth and treadmill recordings).

Raw imaging data (two-photon TIFFs) is not part of the package; the package starts at the
suite2p outputs.

### Pointing the code at the data

Data is looked up through [flexiznam](https://github.com/znamlab/flexiznam), the lab's
database package, used here in offline mode: it reads `offline_database.json` and needs no
account or network access.

Copy `config_template.yml` from the package to `~/.flexiznam/config.yml` and replace every
`/path/to/v1_depth_map_data` in it with the location of the package.

Both projects must point to the same `raw` and `processed` folders. The notebooks read and
write their caches in the `analysis/` folder next to `processed/`; to use another location,
set the environment variable `V1_DEPTH_MAP_ANALYSIS_ROOT` to an absolute path.

## Level 1: drawing the figures

Run everything, in dependency order, and compile the manuscript statistics:

```bash
uv run python run_figures_pipeline.py
```

or selected notebooks:

```bash
uv run python run_figures_pipeline.py --notebooks figure1_depth_selectivity.ipynb
```

`--dry-run` prints the order without running anything. Figures are saved under `analysis/`
as SVG and PDF, and a status table with the run time and peak memory of each notebook is
written to `analysis/FIGURES_README.md`. You can also open a notebook in Jupyter and run it
by hand (`uv run jupyter lab`).

All notebooks are in `v1_depth_map/figures/` unless noted:

| Figure | Notebook |
|---|---|
| Fig. 1 | `figure1_depth_selectivity.ipynb` |
| Fig. 2 | `figure2_rsof_integration.ipynb`, `figure2_openloop.ipynb` |
| Fig. 3 | `figure3_depth_cells.ipynb` |
| Fig. 4 | `figure4_receptive_fields.ipynb` |
| Fig. S1 | `figsupp1_vis_stim_sync.ipynb` |
| Fig. S2 | `figsupp2_speed.ipynb` |
| Fig. S3 | `figsupp3_depth_pop.ipynb` |
| Fig. S4 | `figsupp4_size_control.ipynb` |
| Fig. S5 | `figsupp5_rsof.ipynb` |
| Fig. S6 | `figsupp6_openloop.ipynb` |
| Fig. S7 | `figsupp7_simulation_control.ipynb` |
| Fig. S8 | `figsupp8_multidepth_receptive_fields.ipynb` |
| Fig. S9 | `figsupp9_v1_depth_map.ipynb`, `figsupp9_rf_distance.ipynb` |
| Fig. S10 | `figsupp10_gratings.ipynb` |
| Table S1 | `figure_utils/methods_dataset_tables.ipynb` |

Even at this level, the notebooks that show example sessions re-synchronise those
recordings on every run, so they read a few files from `raw/` and `processed/` as well as
`analysis/`.

### Manuscript statistics

Each notebook exports the numbers quoted in the text to
`v1_depth_map/stats/generated/stats_<notebook>.yaml`. The pipeline then compiles them into
`MANUSCRIPT_STATS.md` and `manuscript_stats.tex` (see `v1_depth_map/stats/README.md`). To
compile or validate them without running notebooks:

```bash
uv run python -m v1_depth_map.stats.build_manuscript_stats [--check]
```

## Level 2: regenerating the caches

The caches in `analysis/` are tables that the notebooks assemble from the per-session
results in `processed/` (and, for the example panels, from the stimulus logs in `raw/`).
`analysis/` is entirely disposable: delete it, run level 2, and it comes back. Nothing at
this level writes to `raw/` or `processed/`.

```bash
uv run python run_figures_pipeline.py --reload                    # rebuild analysis/
uv run python run_figures_pipeline.py --reload --recompute-deep   # also per-session PSTHs and running-speed stats
```

- `--reload` concatenates the per-session results into the cached tables and reruns the
  bootstraps. This is the normal way to regenerate `analysis/`.
- `--recompute-deep` also recomputes the per-session PSTHs and running-speed statistics
  (`psth_crossval.pkl`, `rs_stats.pkl`) of all 84 sessions from the stimulus logs and dF/F
  traces, instead of reading the copies shipped in `processed/`. The recomputed files go to
  `analysis/per_session/`. About 3.5 hours on a laptop, with a peak of about 6 GB of memory.

The flags are set in a temporary copy of each notebook; the notebooks on disk are not
modified. To rebuild one cache by hand, set the flag at the top of the notebook
(`recompute_summary`, `regenerate_cached_files`, `RELOAD`, ...; the full list is
`RELOAD_FLAGS` in `run_figures_pipeline.py`) to `True`. Each notebook has a time limit:
2 hours by default, 4 hours with `--reload` and 6 hours with `--recompute-deep`; change it
with `--timeout SECONDS`.

Some caches are shared between notebooks, so the pipeline runs them in a fixed order
(`--dry-run` prints it).

<details>
<summary>Shared caches</summary>

| Cache in `analysis/` | Written by | Also read by |
|---|---|---|
| `fig1/neurons_df_all.pickle` | `figure1_depth_selectivity` | `figsupp3_depth_pop`, Table S1 |
| `fig2/results_all_rsof_closedloop.parquet` | `figure2_rsof_integration` | `figsupp5_rsof`, `figsupp10_gratings` |
| `fig2/results_all_rsof_openloop.pickle` | `figure2_openloop` | Table S1 |
| `supp/results_all_rs_supp.pickle` | `figsupp6_openloop` | `figsupp2_speed` |
| `rf_neurons_df_slim_spks.pkl`, `rf_all_sig*_spks.pkl`, `rf_gradient_bootstraps_spks.pkl` | `figsupp9_v1_depth_map` | `figsupp8_multidepth_receptive_fields`, `figsupp9_rf_distance` |
| `cache_gratings/` | `figsupp10_gratings` | Table S1 |

</details>

### Time and memory

Most notebooks finish in a few minutes and need less than 10 GB of RAM. The exceptions:

- `figsupp1_vis_stim_sync` and `figure4_receptive_fields` load whole sessions on every
  run, so they need about 20 GB even on the shipped caches.
- `figure2_openloop` takes about 35 minutes at both levels, because it recomputes four
  hierarchical bootstraps of 20,000 resamples on every run.
- Under `--reload`, `figsupp5_rsof` takes about 2.5 hours because it recomputes five
  hierarchical bootstraps of 20,000 resamples (the pipeline runs it last), and
  `figsupp9_v1_depth_map` takes about 1 hour and 20 GB.

A full `--reload` run took about 5.5 hours.

<details>
<summary>Run time and peak memory of each notebook</summary>

Measured by the pipeline on a Linux cluster node with the data on a network file system.
Level 1 times given as "under 15 min" are upper bounds, because they depend mostly on how
fast the files can be read; the other times are as measured.

| Notebook | Level 1 | Level 2 (`--reload`) |
|---|---|---|
| `figure1_depth_selectivity` | under 15 min, 9.6 GB | 6 min, 9.6 GB |
| `figure2_openloop` | 35 min, 9.0 GB | 35 min, 8.7 GB |
| `figure2_rsof_integration` | under 15 min, 2.0 GB | 3 min, 2.6 GB |
| `figure3_depth_cells` | under 15 min, 4.0 GB | 11 min, 5.0 GB |
| `figure4_receptive_fields` | under 15 min, 18.0 GB | 35 min, 19.0 GB |
| `figsupp1_vis_stim_sync` | under 15 min, 20.2 GB | 3 min, 20.5 GB |
| `figsupp2_speed` | under 15 min, 4.9 GB | 8 min, 5.4 GB |
| `figsupp3_depth_pop` | under 15 min, 1.9 GB | 3 min, 1.7 GB |
| `figsupp4_size_control` | under 15 min, 1.4 GB | 3 min, 1.4 GB |
| `figsupp5_rsof` | under 15 min, 1.9 GB | 145 min, 1.9 GB |
| `figsupp6_openloop` | under 15 min, 3.8 GB | 2 min, 3.3 GB |
| `figsupp7_simulation_control` | under 15 min, 5.0 GB | 2 min, 5.1 GB |
| `figsupp8_multidepth_receptive_fields` | under 15 min, 7.2 GB | 9 min, 8.8 GB |
| `figsupp9_rf_distance` | under 15 min, 0.8 GB | 1 min, 0.8 GB |
| `figsupp9_v1_depth_map` | under 15 min, 1.4 GB | 65 min, 20.2 GB |
| `figsupp10_gratings` | under 15 min, 0.6 GB | 9 min, 2.2 GB |
| `methods_dataset_tables` | under 15 min, 0.6 GB | 1 min, 0.6 GB |

</details>

## Level 3: re-analysis

The per-session files in `processed/` (suite2p traces, `neurons_df*.pickle` with the depth,
running speed / optic flow and receptive field fits, decoder results, simulations) were
produced by the scripts in `v1_depth_map/batch_analysis/` and
`v1_depth_map/precompute_data/`. They are slow model fits written for the Crick Slurm
cluster, and they write into `processed/`. They are provided to document how the processed
data was made; [REANALYSIS.md](REANALYSIS.md) lists which script made which file and how
to run them.

## Repository layout

```
run_figures_pipeline.py          run the figure notebooks (levels 1 and 2)
v1_depth_map/
  figures/                       one notebook per manuscript figure, and schematics/
  figure_utils/                  helper modules used by the notebooks; Table S1 notebook
  stats/                         manuscript statistics: export, compiler, tests, generated files
  paths.py                       location of the analysis folder
  revision_sessions.py           sessions of the revision experiments
  batch_analysis/                per-session pipelines (level 3)
  precompute_data/               fits and data preparation scripts (level 3)
  revisions/rf_spike_tau/        deconvolution time constant and RF lag
```

## Tests

The statistics tests need no data:

```bash
uv run pytest v1_depth_map/stats/tests/
```

## Licence and citation

Code: MIT, see [LICENSE](LICENSE). Please cite the paper if you use this code or data:

> Blot A, He Y, Colas Nieto A, Cypranowska C, Znamenskiy P. A depth map of visual space
> in the primary visual cortex. *bioRxiv* 2024.09.27.615442.
> https://doi.org/10.1101/2024.09.27.615442

## Note on exact numbers

Fits and bootstraps are seeded, but regenerated values can differ from the shipped ones across platforms and library versions.

<details>
<summary>Frame matching between stimulus log and monitor frames</summary>

The stimulus shown on each monitor frame is read from the frame log by row number. In a
few recordings the log starts at `FrameIndex` 1, so the stimulus is read one monitor frame
(7 ms) late. `v1_depth_map` keeps this matching (`ROW_MATCHED_PROJECTS`, set in
`v1_depth_map/__init__.py`) so that the shipped results are reproduced exactly; recordings
of the PZAG22 mouse, whose logs start hundreds of frames in, are matched by `FrameIndex`.
A few shipped results were computed with `FrameIndex` matching and change marginally when
recomputed: the depth decoders of 13 original sessions and, for
`PZAG16.3c_S20250401`, the trial-average running speed and optic flow fits and the fits
on simulated responses (counts change by one or two cells).

</details>
