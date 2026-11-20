# Re-analysis (level 3)

This page complements the [README](README.md).

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

`v1_depth_map/revisions/rf_spike_tau/` contains the analysis behind the deconvolution time
constant and the 2-frame lag used for the receptive field fits on spikes.
