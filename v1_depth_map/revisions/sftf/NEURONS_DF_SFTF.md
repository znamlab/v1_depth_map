# `neurons_df_sftf.pickle`: how it is made and what it contains

Per-session fits of drifting-grating responses, for the 6 sessions of `hey2_3d-vision_foodres_20220101`
that have `SFTF` recordings (PZAH10.2d S20230920, S20230922; PZAH10.2f S20230817, S20230912, S20230914,
S20230924).

## Where

`<processed>/hey2_3d-vision_foodres_20220101/<mouse>/<session>/neurons_df_sftf.pickle`, next to
`neurons_df.pickle`. It is written with `to_pickle` and is **not** registered on flexilims.

| file | date, author | notes |
|---|---|---|
| `<session>/neurons_df_sftf.pickle` + `.json` (all 6 sessions) | 2026-10-15, blota (job 60867152, run folder `/camp/home/blota/code/sftf_skip8_run`) | current version: trial window `[onset + 8, next_onset)` (`skip_first_n_volumes=8`, in the `.json`), otherwise as the 2026-10-08 version; copied to BlackPasspo (md5 checked) |
| `/camp/home/blota/code/sftf_skip8_run/backups/<session>/neurons_df_sftf.pickle` | 2026-10-08, blota (job 60117930) | previous version: `ast_neuropil=False` `dff.npy`, 5-fold leave-one-recording-out cross-validation (`sftf_test_*` columns), window `[onset + 2, next_onset + 2)` |
| `/camp/home/blota/code/sftf_crossval_run/backups/<session>/neurons_df_sftf.pickle` | 2026-02-24, hey2 | previous version; ast-neuropil-corrected dF/F (see Known issues) |
| `PZAH10.2f/S20230817/results_temp/neurons_df_sftf.pickle` | 2023-09-11 | early version |

The thesis summary figure (`yiran-phd-thesis/yh_thesis/depth/SFTF.ipynb`, February 2025) read the
version that existed then, which has since been overwritten.

## How it is generated

1. **Batch**: `v1_depth_map/batch_analysis/batch_SFTF/analyze_all_SFTF.py` submits one job per session
   (`pipeline_utils.sbatch_session` -> `cottage_analysis/sbatch/run_depth_SFTF.sh`), with photodiode protocol
   2 for PZAH6.4b / PZAG3.4f and 5 otherwise (5 for all grating sessions).
2. **Pipeline**: `cottage_analysis/pipelines/depth_SFTF.py::main(project, session_name, conflicts,
   photodiode_protocol)`.
3. **Trials**: `analysis/gratings.analyze_grating_responses(..., protocol_base="SFTF")`, for each recording
   whose name contains `SFTF`:
   - `synchronisation.generate_vs_df` (monitor frames from the photodiode, `FrameLog.csv`, parameter log
     `NewParams.csv`) and `synchronisation.generate_imaging_df` (one row per imaging volume);
   - `synchronisation.load_imaging_data`: dF/F of the recording (volumes × ROIs);
   - `gratings.generate_trials_df(img_df, dff, skip_first_n_volumes=8)`: a trial starts at each volume where
     `SpatialFrequency`, `TemporalFrequency` or `Angle` changes. The response window runs from
     8 volumes (0.53 s) after that change to the next one, and the response is the mean dF/F over the
     window. Until 2026-10-15 it ran from 2 volumes after the change to 2 volumes after the next one, which
     is the window of the 2026-10-08 file above. The last trial of each recording is dropped. ROI responses are stored in
     integer-named columns. Two consecutive trials with identical parameters would be merged into one
     (rare, since the order is randomised).
4. **Fit**: `analysis/fit_gaussian_blob.fit_sftf_tuning(trials_df, niter=10, min_sigma=0.25)`, one fit per ROI
   on the single-trial responses of all SFTF recordings of the session (~1440 trials):

   response = A · G(log SF, log TF) · D(α) + offset

   - G: 2D Gaussian (`gaussian_2d` with offset 0), centre (`sf0`, `tf0`), variances
     exp(`log_sigma_x2`) + `min_sigma` and exp(`log_sigma_y2`) + `min_sigma`, rotated by `theta`,
     amplitude A = exp(`log_amplitude`).
   - D: normalised von Mises direction tuning (`direction_tuning`), peak 1 at α = `alpha0`:
     D(α) = [exp(κ cos(α − α0)) + (1 − dsi) · exp(κ cos(α − α0 − π))] / [exp(κ) + (1 − dsi) · exp(−κ)],
     with κ = exp(`log_kappa`).
   - Bounds: `sf0` and `tf0` within 1 log unit beyond the tested grid ([log 0.01 − 1, log 0.32 + 1] and
     [log 0.5 − 1, log 16 + 1]); `theta` in [0, π/2]; `alpha0` in [0, 2π]; `dsi` in [0, 1]; others
     unbounded.
   - Initialisation: `sf0`, `tf0` and `alpha0` at the best marginal stimulus value, the rest random.
     `common_utils.iterate_fit` keeps the best of `niter` starts (seeded with 42).
5. Saved as `neurons_ds.path_full.parent / "neurons_df_sftf.pickle"`.

## Contents

One row per ROI, in suite2p order (same rows as `neurons_df.pickle`; there is no `roi` column), 11 columns
(15 with cross-validation):

| column | meaning | units |
|---|---|---|
| `log_amplitude` | log of the Gaussian amplitude | log dF/F |
| `sf0` | preferred spatial frequency | log(cycles/deg); `exp(sf0)` in cpd |
| `tf0` | preferred temporal frequency | log(Hz); `exp(tf0)` in Hz |
| `log_sigma_x2`, `log_sigma_y2` | log variances of the Gaussian along its axes (before adding `min_sigma`) | log(log-units²) |
| `theta` | rotation of the Gaussian in the log SF × log TF plane | rad, [0, π/2] |
| `offset` | baseline response | dF/F |
| `alpha0` | preferred direction | rad, [0, 2π] |
| `log_kappa` | log of the von Mises concentration (tuning width) | |
| `dsi` | direction selectivity: 0 = same response to both directions of an orientation, 1 = no response to the opposite direction | [0, 1] |
| `rsq` | R² of the best fit, **in-sample** on single trials | |
| `sftf_test_rsq` | R² of the held-out predictions, concatenated across folds (one fold per recording) | |
| `sftf_test_spearmanr_rval`, `sftf_test_spearmanr_pval` | Spearman correlation (two-sided p) between held-out responses and predictions | |
| `sftf_test_popts` | fit parameters of each fold (n_folds × 10) | |

Derived quantities used in the figures: preferred speed = exp(`tf0` − `sf0`) deg/s; preferred
orientation = `alpha0` mod π.

Direction convention (thesis notes): 0° = vertical bars moving forward, 90° = horizontal bars moving down,
180° = vertical bars moving backward, 270° = horizontal bars moving up.

## Known issues

- **The 2026-02-24 files use ast-neuropil-corrected dF/F, not the `ast_neuropil=False` traces of the
  depth `neurons_df`.** This was tested on 2026-10-07 by rerunning the pipeline on PZAH10.2f_S20230914
  on NEMO (job 60113469, folder `/camp/home/blota/code/sftf_repro_tmp`, which can be deleted), with
  three inputs:

  | input | vs production |
  |---|---|
  | A: `ast_neuropil=False` dataset, `dff_ast.npy` (pipeline as committed before `ead762b`) | identical (all 548 ROIs, all columns) |
  | B: `ast_neuropil=False` dataset, `dff.npy` (the actual non-ASt dF/F) | different (`rsq` r = 0.98, parameters r = 0.0-0.6) |
  | C: `ast_neuropil=True` dataset (2023), `dff_ast.npy` | identical |

  A and C are identical because the `dff_ast.npy` in the `ast_neuropil=False` folder (2025-09-15) is
  byte-identical (md5) to the 2023 ASt `dff_ast.npy`. The non-ASt dF/F is `dff.npy` (2025-10-06). The fit
  is deterministic (seeded), so a rerun of the pipeline as committed reproduces the files exactly.
- `load_imaging_data` always read `dff_ast.npy`, even from `ast_neuropil=False` folders, which caused the
  point above. It was fixed in cottage_analysis `ead762b` (2026-10-07), so a new run uses `dff.npy`.
- `fit_sftf_tuning` failed with the current `common_utils.iterate_fit` (response passed as a pandas
  Series). This was fixed in `325d16b` (2026-10-07).
- `rsq` is in-sample, so a threshold on it does not control false positives the way the cross-validated
  depth criterion does.

## Regenerating

[`precompute_data/rerun_depth_sftf.py`](../../precompute_data/rerun_depth_sftf.py) and its sbatch file
rerun `depth_SFTF.main` per session. They back up the existing file and check it first, and write a
`neurons_df_sftf.json` with the provenance next to the new file. On NEMO, from a run folder containing
`src/` (copies of cottage_analysis and v1_depth_map), `backups/`, `logs/` and a manifest with one
session per line:

    sbatch --array=0-5 src/v1_depth_map/precompute_data/rerun_depth_sftf.sbatch manifest.txt

Run folder for the October 2026 rerun: `/camp/home/blota/code/sftf_crossval_run`. It uses cottage_analysis
`23a5b63`, and `backups/` holds the 2026-02-24 files.

## Reanalysis for the reviewer (2026-10-07)

`figures/figsupp10_gratings.ipynb` does not read `neurons_df_sftf.pickle`. It refits with the same
function and settings and caches the results in
`<processed>/v1_manuscript_figures/ver_rev1/cache_gratings/cache/<session>_sftf_fit.pickle`, with
the same 11 columns plus `session` and `roi`. Its trial table differs from `generate_trials_df` in three
ways:
- traces are loaded through `generate_imaging_df` with `{"anatomical_only": 3, "ast_neuropil": False}`,
  i.e. `dff.npy`;
- trials are split where the parameter-log time changes, so identical consecutive trials stay separate;
- the last trial of each recording is kept, with the median trial length.
