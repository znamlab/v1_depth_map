# Plan: a simple model of grating responses with running speed, chosen on 10 good cells

Status: plan only (2026-10-10). To be executed by another agent. Background and results so far are in
[README.md](README.md) and [NEURONS_DF_SFTF.md](NEURONS_DF_SFTF.md); read them first.

## Goal

Find the simplest model that explains, frame by frame, how a V1 neuron's response to drifting gratings
depends on direction, spatial frequency (SF), temporal frequency (TF) and running speed (RS). Use 10 cells
with strong, well-fit responses, where the model choice is not dominated by noise. Main question: does
running **shift** the preferred TF (or SF), or does it mainly **scale** the response (gain) or add to it
(offset)? These are confounded in the current analysis.

## What exists (reuse, do not rewrite)

- **Data**: 6 sessions of `hey2_3d-vision_foodres_20220101` (PZAH10.2d S20230920, S20230922; PZAH10.2f
  S20230817, S20230912, S20230914, S20230924), 5 SFTF recordings each: 6 SF (0.01-0.32 cpd) x 6 TF
  (0.5-16 Hz) x 8 directions, 4 s per grating, gratings in open loop, mice mostly running (median ~0.33 m/s).
- **Frames**: `v1_depth_map/precompute_data/sftf_running_models.py::session_frames(flm, session)` returns
  running frames (RS > 1 cm/s) with `X = (log SF, log TF, direction rad, log RS - median log RS)`,
  dF/F of all ROIs (`ast_neuropil=False`, `dff.npy`, shifted by `LAG = 5` frames), recording names, and
  the median log RS. It runs locally (data on `/Volumes/BlackPasspo`, offline flexiznam) in ~1 min per
  session. The same frames for the SF/TF-tuned ROIs are saved on NEMO in
  `/camp/home/blota/code/sftf_running_models/results/frames/<session>.npz`.
- **Base model**: `cottage_analysis.analysis.fit_gaussian_blob.grating_tuning` = offset + A * G(log SF,
  log TF) * D(direction), G a rotated 2D Gaussian (`sf0`, `tf0`, `log_sigma_x2`, `log_sigma_y2`, `theta`,
  `min_sigma=0.25`), D a von Mises with a direction selectivity index (`alpha0`, `log_kappa`, `dsi`).
  Parameters in `GratingParams`.
- **Running-speed models already fitted** (all 1455 SF/TF-tuned iscell cells): base, TF shift (`a_tf`),
  SF shift (`a_sf`), both. Results:
  `/Volumes/BlackPasspo/v1_depth_map/processed/v1_manuscript_figures/ver_rev1/fig_reviewer_gratings/running_models/<session>_sftf_running_models.pickle`
  (columns `<model>_<param>`, `<model>_rsq`, `<model>_test_rsq`, `<model>_test_spearmanr_*`). Finding so
  far: median TF slope +0.18 (TF0 ∝ RS^0.18), consistently positive; SF slope ~0; held-out R² gains tiny.
  Model code (`rs_tuning`, `fit_models`, `fit_roi`, `bounds`, `random_p0`) is in the same script.
- **Notebook**: `v1_depth_map/figures/fig_reviewer_gratings.ipynb`. Speed-binned SF/TF maps and TF
  curves per running-speed bin: `v1_depth_map/figure_utils/sftf_examples.py`
  (`speed_binned_responses`, `tf_curves_by_speed`, `plot_sftf_by_speed`, `plot_tf_curves`), using the
  figure 3 RS bins `[2.83, 5.66, 11.31, 22.63, 45.25, 90.51]` cm/s.

## Step 1: select 10 cells

From the running-model results (base model, frame level):

- base held-out R² (`base_test_rsq`) > 0.1 and amplitude `exp(base_log_amplitude)` > 0.5 dF/F;
- preferred SF and TF inside the tested grid (`base_sf0` within log [0.01, 0.32], `base_tf0` within log
  [0.5, 16]), so the shift parameters are constrained by data;
- no collapsed Gaussian axis (`base_log_sigma_x2`, `base_log_sigma_y2` > -10);
- at least 1 cell per session and both mice; mix of depth-selective and other cells (`depth_tuned` in
  `grating_tuning_cells.csv` next to the figure outputs); rank by `base_test_rsq` within these constraints.

87 cells pass the first three criteria (11-20 per session). Top ones include S20230920 ROI 220,
S20230922 ROI 171, S20230912 ROI 391, S20230924 ROIs 206 and 269, S20230914 ROI 8, S20230817 ROIs 62, 259,
214 and 544. Record the final list (session, roi, why) in the results file.

## Step 2: candidate models

Keep every model a small extension of the base model. With `r` = log RS - median log RS (frame level):

| name | change from base | new parameters |
|---|---|---|
| `base` | none | 0 |
| `gain` | amplitude A * exp(g · r) (multiplicative running gain) | 1 |
| `offset` | offset + b · r (additive running modulation, stimulus independent) | 1 |
| `gain_offset` | both | 2 |
| `tf` | log TF0 = tf0 + a_tf · r (existing) | 1 |
| `gain_offset_tf` | gain + offset + TF shift | 3 |
| `gain_offset_sf` | gain + offset + SF shift | 3 |
| `gain_offset_tf_sf` | gain + offset + TF and SF shifts | 4 |
| `gain_offset_speed` | gain + offset + shift along the speed axis: log TF0 = tf0 + a · r, log SF0 = sf0 - a · r (preferred speed TF/SF scales as RS^2a) | 3 |

Optional, only if a cell's residuals show it: TF width changing with RS (`log_sigma` + c · r), direction
tuning sharpening with RS (`log_kappa` + k · r). Do not add more than one extra term at a time.

The key comparison is `gain_offset` vs `gain_offset_tf` (vs `_sf`, `_tf_sf`, `_speed`): does a preference
shift remain once gain and offset are allowed?

## Step 3: fitting

- Extend `rs_tuning` (or write a small `make_model(terms)` returning the function, parameter names,
  bounds and the base-to-model start) in a new module, e.g. `v1_depth_map/precompute_data/sftf_running_model_selection.py`.
  Do not change `fit_gaussian_blob` or the production `neurons_df_sftf`.
- Bounds: as `sftf_running_models.bounds`, plus **lower bound -20 on `log_sigma_x2`, `log_sigma_y2`**
  (the model is flat below that, and leaving it free made some fits take hours), upper bound ~10 on
  `log_kappa`; RS terms within ±5 (g, a), b free.
- Starts: nested warm starts. Fit `base` with 5 random starts (as now), then each model from the best
  simpler model it contains with the new terms at 0, plus 3 random starts. Keep the best in-sample R².
  Seed per fit for reproducibility. Cap `maxfev` (e.g. 5000) and log fits that hit it.
- Cross-validation: leave one recording out (5 folds), held-out R² of the concatenated predictions, and
  held-out Spearman, as in `fit_roi`. Also report in-sample R², and AIC/BIC from the residual sum of
  squares as a secondary criterion (frames are autocorrelated, so treat them as indicative).
- Robustness (on the 10 cells only): repeat the comparison with (a) LAG 2 and 8 frames, (b) RS smoothed
  over ~0.5 s before taking the log, (c) trial-level responses (mean dF/F over running frames of each
  trial, mean log RS of those frames). The winning model should not depend on these.
- Compute: ~10 models x 6 fits x 10 cells is small. Run locally with joblib, or on NEMO with the
  prepare/fit/merge pattern of `sftf_running_models.sbatch` (one task per cell x model). On NEMO follow the
  crick-nemo skill (no computation on the login node, preflight the sbatch, run folder under
  `/camp/home/blota/code/`, write only there).

## Step 4: choose the model

- For each cell, held-out R² of every model relative to `base` and to `gain_offset`.
- Pick the simplest model whose held-out R² is within a small margin (e.g. 10% of the best model's gain
  over `base`) of the best, for most cells. Report how many of the 10 cells each model wins.
- Check the fitted RS terms are consistent across cells (sign and order of magnitude of g, b, a_tf).
- Sanity check on simulated data: simulate frames from the fitted `gain_offset` model of 2 cells (with
  their real stimulus and RS sequence and Gaussian noise of the residual variance) and confirm the TF-shift
  model then finds a_tf ≈ 0; and the reverse (simulate with a TF shift, check it is recovered). This shows
  gain and shift are separable with this stimulus.

## Step 5: figures (notebook section, exploratory)

Add a section to `fig_reviewer_gratings.ipynb` (keep it short; put plotting helpers in
`figure_utils/sftf_examples.py`):

- per cell: TF curves per figure 3 RS bin, data (`tf_curves_by_speed`) with the predictions of
  `gain_offset` and of the winning model overlaid (same SF / direction pairs);
- held-out R² gain over `base` per model, one line per cell;
- fitted RS parameters (g, b, a_tf, a_sf) per cell for the winning model.

## Deliverables

- Code: the new script (+ sbatch if NEMO is used), committed with `[revisions] ...` messages ending with the
  `Co-Authored-By` line; pre-commit runs ruff-format and nbstripout (re-stage and commit again if they
  modify files).
- Results: `<processed>/v1_manuscript_figures/ver_rev1/fig_reviewer_gratings/running_model_selection/`
  (per-cell fits, held-out R² table, chosen cells).
- A short section in `revisions/sftf/README.md`: cells, models, held-out R² table, chosen model, RS
  parameter values, robustness checks, caveats. Not exported to the manuscript stats.

## Pitfalls already met

- Frame-level R² is small (median 0.001 for all SF/TF-tuned cells), hence the selection of good cells.
- Collapsed Gaussian axes (`log_sigma` → -∞) make `curve_fit` wander: bound them.
- The mice rarely stand still and running varies within trials (30-64% of the log RS variance is between
  trials), so trial-level analyses lose information; use frames for the main fit.
- Running speed is instantaneous (encoder difference per imaging volume) and not lagged; dF/F is lagged
  by `LAG` frames relative to stimulus and RS.
- `PZAH10.2f_S20230817`'s traces dataset index is swapped (`suite2p_traces_0` is the `ast_neuropil=False`
  one); always select traces with `{"anatomical_only": 3, "ast_neuropil": False}`, never by index.
