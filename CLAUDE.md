# CLAUDE.md

Analysis and figure code for the manuscript "A depth map of visual space in the primary visual
cortex" (Znamenskiy lab, Crick), currently in revision for *Science*.

## Git rules

- **Never commit unless the user explicitly asks for a commit.** Finishing a task is not a request
  to commit.
- **Ask for confirmation before every commit**, even when commits were requested earlier in the
  session: show the files to be staged and the proposed message, and wait for a yes. One approval
  covers one commit.
- Commit messages use a bracketed tag: `[revisions]`, `[figures]`, `[precompute]`, `[stats]`,
  `[feature]`, `[refactor]`, `[minor]`, `[analysis]`, `[style]`.
- Stage specific files, never `git add -A`: the working tree often holds unrelated
  in-progress notebooks and generated files.

## Related repositories

- `../cottage_analysis` (editable dependency via `[tool.uv.sources]`): all core analysis,
  fitting (`analysis/`, `pipelines/`) and plotting (`plotting/`, including
  `plotting.style`). Generic logic belongs there; this repo holds project-specific glue,
  notebooks and batch scripts.
- `../2p-preprocess`: suite2p / dF/F / deconvolution preprocessing (`2p calcium`,
  `2p reextract`).
- `flexiznam` / `flexilims`: the lab database. Sessions, recordings and datasets are looked up
  through it; the data root comes from `flz.get_data_root(...)`. Offline mode uses
  `offline_database.json` (see README.md).

## Environment and commands

- Local env: `uv` with `.venv` (`uv run ...`). On NEMO: conda env `v1_depth_map`.
- Tests (no data needed): `uv run pytest v1_depth_map/stats/tests/`
- Lint/format: ruff (pre-commit also runs nbstripout, codespell, end-of-file/whitespace fixers).
  Notebooks are committed with outputs stripped, except the two `rf_spike_tau` notebooks
  that keep their figures.
- Run all figure notebooks with memory supervision, then compile stats:
  `python run_figures_pipeline.py [--notebooks NAME.ipynb] [--timeout S]`. Executed copies are
  discarded; status goes to `FIGURES_README.md`.
- Compile manuscript stats: `python -m v1_depth_map.stats.build_manuscript_stats [--check]`.

## Layout

- `v1_depth_map/paths.py`: `FIGURES_VERSION` (currently `"_rev1"`) and `get_figures_roots()`.
  Notebooks read/write under `<processed>/v1_manuscript_figures/ver_rev1/`. Never hardcode
  output paths.
- `v1_depth_map/figures/`: one notebook per manuscript figure (`figure1..4_*`, `figsupp1..9_*`)
  plus reviewer figures (`fig_reviewer_*`). `unused_*` notebooks are skipped by the pipeline.
  Notebooks start with `style.setup_figure_fonts()` and save via `style.savefig(..., fig=...)`;
  sizes come from `style.FONTSIZE_DICT`. A `reload` flag controls rebuilding cached pickles.
- `v1_depth_map/figure_utils/`: helper modules used by the notebooks (RF, RF distance, RS/OF,
  treadmill, SF/TF examples, V1 map, etc.).
- `v1_depth_map/stats/`: manuscript numbers pipeline. Notebooks call
  `export_figure_stats("<notebook_stem>", stats_dict, figure="figN")`, which writes one YAML per
  notebook to `stats/generated/`. `build_manuscript_stats.py` compiles them into
  `MANUSCRIPT_STATS.md` and `manuscript_stats.tex`. Rules (see `notes/05_manuscript_statistics.md`):
  stats are computed in the notebook that draws the panel, export cells only read existing
  variables (never re-run a test), and the compiler has no analysis logic. Copying the `.tex` to
  Overleaf is manual.
- `v1_depth_map/precompute_data/`: heavy fits and data prep, usually a Python script plus a
  `.sbatch` for NEMO Slurm arrays (manifest of argument lines, `src/` checkout on `PYTHONPATH`,
  results in a run directory). Production `neurons_df*` files are only read by revision
  scripts; new fits go to new files or run directories, with backups before overwriting.
- `v1_depth_map/batch_analysis/`: original per-session pipelines (preprocess, depth/RS-OF/RF
  fits, decoders, size control, SF/TF, eye tracking) run as Slurm jobs.
- `v1_depth_map/revisions/`: analyses for the reviewer response. Subfolders (`sftf/`,
  `rf_spike_tau/`, `rsof_spikes/`) have their own README with status, decisions and data
  locations; read them before working on that topic. `revision_sessions.py` lists sessions.
- `v1_depth_map/presentations/`, `additional_analysis/`: frozen talk/exploratory notebooks;
  they may read old column names. Don't update them unless asked.
- Root scripts (`run_full_cut.py`, `run_grid_subsets.py`, `run_trial_average_fits.py`,
  `submit_rf_only.py`): decoder and fit batch submissions.
- `notes/01..05_*.md`: tracking docs for preprocessing, fits and `neurons_df` lineage, notebook
  status, figure assembly and stats. Keep them in sync when a figure or fit changes.

## Data and conventions

- Projects: `hey2_3d-vision_foodres_20220101` (main dataset), `colasa_3d-vision_revisions`
  (revision recordings, curated Cellpose masks + `2p reextract`).
- Local data on `/Volumes/BlackPasspo/v1_depth_map/{raw,processed}`; on NEMO under
  `/camp/lab/znamenskiyp/home/shared/projects/`. Heavy computation runs on NEMO (see the
  `crick-nemo` skill), never on login nodes.
- Storage & caches: Never create large data caches (`.pkl`, `.npz`, `.parquet`) on the OS boot
  drive. If `/Volumes/BlackPasspo` is mounted, save scratch/intermediate caches to
  `/Volumes/BlackPasspo/agent_cache/` (or under `/Volumes/BlackPasspo/v1_depth_map/processed/`).
- All fits use traces with `filter_datasets={"anatomical_only": 3, "ast_neuropil": False}`.
- Per-session results live in `neurons_df.pickle`. Variants have suffixes, e.g.
  `neurons_df_spks.pickle` (RFs refit on deconvolved spikes, used by the manuscript for all RF
  panels and stats), `neurons_df_sftf[_spks].pickle` (gratings). Notebooks select the variant with
  a constant (e.g. `RF_FILE`) and add a matching suffix to outputs.
- Depth-selective neurons: `iscell` and one-sided Spearman p < 0.05 on the cross-validated depth
  tuning fit (`depth_tuning_test_spearmanr_*_closedloop`).
- Deconvolved spikes analysis of RF use a 2-frame stimulus lag; The rest use dF/F no lag.
- Generated figures (`*.pdf`, `*.svg`, `*.png`, `*.pickle`) are gitignored; the stats YAML/MD/TeX
  are tracked.
