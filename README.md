# v1_depth_map

Analysis and figure code for "A depth map of visual space in the primary visual cortex"
(Znamenskiy lab, The Francis Crick Institute).

This repository draws every figure and computes every number of the manuscript from the
accompanying data package (`<DATA_PACKAGE_URL>`). The core analysis library (fits,
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
| 1. Draw the figures | Runs the figure notebooks on the shipped caches | `analysis/` | figures in `analysis/`, statistics in this repository | About 1 hour for all figures on a laptop, about 6 GB of RAM |
| 2. Regenerate the caches | Rebuilds the notebook caches from the per-session data | `processed/`, `raw/` | `analysis/` | Laptop; under an hour, or several hours with the deep option (see below) |
| 3. Re-analyse | Reruns preprocessing and the model fits | `raw/`, `processed/` | `processed/` | Slow; written for a Slurm cluster |

**Level 1, drawing the figures.** Each notebook in `v1_depth_map/figures/` draws one
manuscript figure. With the default flags a notebook only reads its cache from `analysis/`,
so this level needs no fitting and no per-session data. The data package ships `analysis/`
already filled, so this works straight after download.

**Level 2, regenerating the caches.** The caches in `analysis/` are tables that the
notebooks assemble from the per-session results in `processed/` (and, for the example
panels, from the stimulus logs in `raw/`). Every notebook has a reload flag that rebuilds
its cache; `analysis/` is entirely disposable: delete it, run level 2, and it comes back.
Nothing at this level writes to `raw/` or `processed/`. It runs on a laptop but can be
slow, and has two depths:

- `--reload`: concatenates the per-session results into the cached tables and reruns the
  bootstraps. This is the normal way to regenerate `analysis/`.
- `--recompute-deep`: also recomputes the per-session PSTHs and running-speed statistics
  (`psth_crossval.pkl`, `rs_stats.pkl`) from the raw stimulus logs and dF/F traces of all
  84 sessions, instead of reading the copies shipped in `processed/`. The recomputed files
  go to `analysis/per_session/`. About 3.5 hours on a laptop (2.2 h for the PSTHs, 1.1 h
  for the running-speed statistics), with a peak of about 6 GB of memory.

**Level 3, re-analysing.** The per-session files in `processed/` (suite2p traces,
`neurons_df*.pickle` with the depth, running speed / optic flow and receptive field fits,
decoder results, simulations) were produced by the scripts in
`v1_depth_map/batch_analysis/` and `v1_depth_map/precompute_data/`. They are slow model
fits, were run as Slurm jobs, and write their results into `processed/`. They are provided
to document exactly how the processed data was made; see
[Re-analysis](#level-3-re-analysis) before running them.

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

Download and unpack the data package (`<DATA_PACKAGE_URL>`). Its layout is:

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

Copy `config_template.yml` from the package to `~/.flexiznam/config.yml` and replace
`/path/to/v1_depth_map_data` with the location of the package:

```yaml
offline_mode: true
offline_yaml: /path/to/v1_depth_map_data/offline_database.json
project_ids:
  hey2_3d-vision_foodres_20220101: 624ae21a73245b6d9992a1f5
  colasa_3d-vision_revisions: 67ae2a36b99b5006b4e2a8dd
project_paths:
  hey2_3d-vision_foodres_20220101:
    raw: /path/to/v1_depth_map_data/raw
    processed: /path/to/v1_depth_map_data/processed
  colasa_3d-vision_revisions:
    raw: /path/to/v1_depth_map_data/raw
    processed: /path/to/v1_depth_map_data/processed
```

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

```bash
uv run python run_figures_pipeline.py --reload                    # rebuild analysis/
uv run python run_figures_pipeline.py --reload --recompute-deep   # also per-session PSTHs and running-speed stats
```

The flags are set in a temporary copy of each notebook; the notebooks on disk are not
modified. To rebuild one cache by hand, set the flag at the top of the notebook
(`recompute_summary`, `regenerate_cached_files`, `RELOAD`, ...; the full list is
`RELOAD_FLAGS` in `run_figures_pipeline.py`) to `True`. Each notebook has a time limit:
15 minutes by default, 1 hour with `--reload` and 4 hours with `--recompute-deep`; change it
with `--timeout SECONDS`.

Some caches are shared, which is why the pipeline runs the notebooks in a fixed order:

| Cache in `analysis/` | Written by | Also read by |
|---|---|---|
| `fig1/neurons_df_all.pickle` | `figure1_depth_selectivity` | `figsupp3_depth_pop`, Table S1 |
| `fig2/results_all_rsof_closedloop.parquet` | `figure2_rsof_integration` | `figsupp5_rsof`, `figsupp10_gratings` |
| `fig2/results_all_rsof_openloop.pickle` | `figure2_openloop` | Table S1 |
| `supp/results_all_rs_supp.pickle` | `figsupp6_openloop` | `figsupp2_speed` |
| `rf_neurons_df_slim_spks.pkl`, `rf_all_sig*_spks.pkl`, `rf_gradient_bootstraps_spks.pkl` | `figsupp9_v1_depth_map` | `figsupp8_multidepth_receptive_fields`, `figsupp9_rf_distance` |
| `cache_gratings/` | `figsupp10_gratings` | Table S1 |

Expected cost on a laptop (Apple M1 Pro, data on an external hard drive):

| Step | Time | Peak RAM |
|---|---|---|
| Fig. 2 closed-loop table | about 5 min | 1.4 GB |
| Fig. S2 eye-tracking cache | about 11 min | 4.6 GB |
| Fig. S10 gratings tables (`RELOAD`) | 25 to 30 min | under 3 GB |

Fits and bootstraps are seeded, but regenerated values can differ from the shipped ones in
the last digits across platforms and library versions.

## Level 3: re-analysis

The scripts below produced the contents of `processed/`. They are kept so that every
processed file can be traced to the code that made it. They were written for the Crick
cluster: the `.sh` and `.sbatch` files are Slurm wrappers that need adapting (partition,
modules, environment, paths), and several scripts take hours per session.

**These scripts write into `processed/`**, replacing or adding to the shipped results. Work
on a copy if you want to keep the original.

Most of the revision fits work in two steps. The fit writes its result, with a provenance
`.json`, to a working folder given with `--out-dir`, and leaves the session folder alone. A
second call with `--merge` (try `--dry-run` first) copies the fitted columns into the
session's `neurons_df.pickle`: it matches neurons by `roi`, makes a byte-for-byte backup,
writes atomically, then reloads the file and checks that the merged columns equal the fit
and that every other column is unchanged.

```bash
S=v1_depth_map/precompute_data/refit_treadmill_depth.py
uv run python $S SESSION plateau_single_frames --out-dir WORK            # fit
uv run python $S SESSION plateau_single_frames --out-dir WORK --merge    # write into neurons_df.pickle
```

`fit_revision_treadmill.py` writes its fit files into the session folder and has its own
`--merge`. `refit_rsof_batched.py` writes the same files to `--out-dir`; copy them into the
session folder and run `fit_revision_treadmill.py --merge --conflicts overwrite`.

| Product in `processed/` | Script |
|---|---|
| suite2p ROIs and traces, dF/F | `batch_analysis/batch_preprocess/`, `batch_analysis/batch_preprocess_revisions/` (uses [2p-preprocess](https://github.com/znamlab/2p-preprocess)) |
| `neurons_df.pickle`: depth tuning, running speed / optic flow and RF fits | `batch_analysis/batch_analysis/` (the `cottage_analysis` analysis pipeline) |
| Size-control fits | `batch_analysis/batch_size_control/` |
| `decoder_results.pickle` | `batch_analysis/batch_depth_decoder/` |
| Eye tracking | `batch_analysis/eye_tracking/` |
| Deconvolved spikes and `neurons_df_spks.pickle` (RF fits on spikes) | `precompute_data/deconvolve_tau.py`, `precompute_data/rf_fit_spikes.py` |
| Grating fits: `neurons_df_sftf.pickle` | `batch_analysis/batch_SFTF/`, `precompute_data/rerun_depth_sftf.py` |
| Treadmill depth tuning (columns `*_treadmill`, `*_treadmill_model` of `neurons_df.pickle`) | `precompute_data/refit_treadmill_depth.py` |
| Closed-loop depth tuning of the revision sessions | `precompute_data/refit_closedloop_depth.py` |
| Treadmill and closed-loop running speed / optic flow fits (columns `*_treadmill*` of `neurons_df.pickle`) | `precompute_data/fit_revision_treadmill.py`; `precompute_data/refit_rsof_batched.py` runs the same fits as a Slurm array over batches of neurons |
| Simulated responses (`simulated_*.parquet`) | `precompute_data/fit_revision_simulation.py`, `precompute_data/rerun_simulation_tdecay2_areanorm.py` |
| `psth_crossval.pkl`, `rs_stats.pkl` | `precompute_data/make_depth_tuning_raster.py`, `precompute_data/calculate_rs_stats.py` |

`fit_revision_treadmill.py` and `fit_revision_simulation.py` check which copy of the data
they are about to write to. Outside the Crick, pass `--site other`, which accepts any data
root and prints it before running:

```bash
uv run python v1_depth_map/precompute_data/fit_revision_treadmill.py --site other --only treadmill --method plateau
```

**Frame matching.** The stimulus shown on each monitor frame is read from the frame log,
matched by row number. A few recordings have a frame log that starts at `FrameIndex` 1, so
there the stimulus is read one monitor frame (7 ms) late. The shipped results were
computed that way, and `v1_depth_map` keeps this matching so that they are reproduced
exactly (`ROW_MATCHED_PROJECTS` in `cottage_analysis.preprocessing.synchronisation`, set
in `v1_depth_map/__init__.py`). The effect is small: with `FrameIndex` matching, the PSTH
of a cell correlates at 0.95 or more with the shipped one, and none of 60 cells refitted in
the most affected session changed depth-selective status. The PZAG22 mouse is the
exception: their frame logs start hundreds of frames in, and they are matched by
`FrameIndex`.

`v1_depth_map/revisions/rf_spike_tau/` contains the analysis behind the deconvolution time
constant and the 2-frame lag used for the receptive field fits on spikes.

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

## Conventions

- All fits use traces with `anatomical_only=3` and `ast_neuropil=False` (main dataset) or
  the curated `annotated` masks (revision dataset).
- Depth-selective neurons: `iscell` and one-sided Spearman p < 0.05 on the cross-validated
  depth tuning fit.
- Receptive field panels and statistics use the fits on deconvolved spikes
  (`neurons_df_spks.pickle`), with a 2-frame stimulus lag. Everything else uses dF/F with
  no lag.

## Licence and citation

Code: MIT, see [LICENSE](LICENSE). Please cite the paper if you use this code or data: `<CITATION>`.
