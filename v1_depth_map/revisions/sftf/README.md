# Drifting gratings: is depth tuning multiplexed with orientation / SF / TF tuning?

Reviewer comment:

> One major open question in my mind is how this sensitivity to depth relates to other sensitivities,
> such as orientation. Are these multiplexed, or are they separate sub-populations? [...] it seems possible
> to derive orientation selectivity of the recorded neurons during (near) stationary epochs.

Status as of 2026-10-07: analysis notebook written and run; conclusions below are preliminary.
How the per-session grating fits (`neurons_df_sftf.pickle`) are made is in
[NEURONS_DF_SFTF.md](NEURONS_DF_SFTF.md).

## Data

- 6 figure sessions, 2 mice (GCaMP6f): PZAH10.2d S20230920, S20230922; PZAH10.2f S20230817, S20230912,
  S20230914, S20230924. No other session of `hey2_3d-vision_foodres_20220101` has gratings.
- 5 `SFTF` recordings per session, after the depth protocol. Each recording shows the full grid once:
  6 SF (0.01-0.32 cpd) × 6 TF (0.5-16 Hz) × 8 directions, ~4 s per grating (61 imaging volumes at 15.2 Hz),
  no blank. 5 repeats per condition per session (one trial short in PZAH10.2f_S20230817).
- Same suite2p ROIs as the depth recordings (ROI counts match in every session), so tuning can be
  compared cell by cell.
- The gratings are separate recordings, not interleaved with the depth stimulus, so **all grating trials
  are used** (no stationary-epoch selection). Response = mean dF/F (`ast_neuropil=False` traces).
- 3012 `iscell` ROIs in total.

## Analysis

Notebook: [`figures/fig_reviewer_gratings.ipynb`](../../figures/fig_reviewer_gratings.ipynb).
Outputs in `<processed>/v1_manuscript_figures/ver_rev1/fig_reviewer_gratings/` (figure PDFs,
`grating_tuning_cells.csv` with one row per cell, and `cache/` with the trial tables and fits).

- **Depth selective**: as in the manuscript figures, i.e. `iscell` and one-sided Spearman p < 0.05 between
  held-out and predicted responses of the cross-validated depth tuning fit
  (`depth_tuning_test_spearmanr_*_closedloop`, via `figure_utils.rf_distance.load_session_neurons`).
- **SF/TF tuned**: `fit_sftf_tuning` R² > 0.05 (criterion of the thesis figure, see below). Note this R²
  is in-sample on single trials, unlike the cross-validated depth criterion.
- **RS/OF tuned** (panels B, C only): `rsof_test_rsq_closedloop_g2d` > 0.02 (thesis criterion).
- Summary figure `fig_sftf_summary.pdf` reproduces the layout, selections and stats of the thesis figure
  (`hey2` `yiran-phd-thesis/yh_thesis/depth/SFTF.ipynb`, commit 46a6fae, 2025-02-07), except that depth
  selectivity uses the current one-sided test instead of `rval > 0.1 & two-sided p < 0.05`. Fits that
  end on a parameter bound are kept, as in the thesis.
  - A: Venn of depth-selective and SF/TF-tuned cells.
  - B, C: preferred grating speed (TF0 / SF0) vs preferred optic flow speed and preferred virtual depth,
    for SF/TF-tuned and RS/OF-tuned cells. Spearman ρ, p from a hierarchical bootstrap over mice then
    sessions (20000 resamples).
  - D, E: fitted direction selectivity (`dsi`) and preferred orientation, SF/TF-tuned and depth-selective
    cells. F: preferred direction of those with `dsi` > 0.6.
- Model-free check (later sections of the notebook): vector-sum gOSI / gDSI and preferred orientation /
  direction at each cell's preferred SF/TF; responsive = one-way ANOVA across the 288 conditions
  (p < 0.01); orientation / direction selective = responsive and gOSI / gDSI shuffle p < 0.05.

## Results

| | |
|---|---|
| Depth selective | 660 / 3012 (22%) |
| SF/TF tuned | 1167 / 3012 (39%) |
| Both | 305 (10%); 8.5% expected if independent |
| SF/TF tuned among depth-selective / other cells | 46% / 37% (χ² p = 1e-5) |
| RS/OF tuned (B, C) | 75; SF/TF and RS/OF tuned: 31 |
| B: grating speed vs preferred OF | ρ = 0.18, 95% CI [-0.49, 0.68], bootstrap p = 0.43 (n = 31) |
| C: grating speed vs preferred depth | ρ = -0.07, 95% CI [-0.59, 0.43], bootstrap p = 0.67 (n = 31) |
| D: `dsi` of depth-selective, SF/TF-tuned cells | 44% below 0.5; median 0.57 vs 0.59 in non-depth-selective SF/TF-tuned cells (p = 0.89) |
| E: preferred orientation (0/45/90/135°) | 82 / 61 / 100 / 62 cells, similar to non-depth-selective cells (229 / 176 / 252 / 205) |

Model-free check:

| | depth selective | other | p |
|---|---|---|---|
| Responsive (ANOVA) | 60% | 50% | 3e-6 (χ²) |
| Orientation selective | 25% | 26% | 0.76 (χ²) |
| Direction selective | 14% | 12% | 0.14 (χ²) |
| Preferred orientation distribution (orientation-selective cells) | | | 0.87 (χ², 4 bins) |
| Preferred direction distribution (direction-selective cells) | | | 0.32 (χ², 8 bins) |
| gOSI (responsive cells) | | | 0.13 (Mann-Whitney) |
| gDSI (responsive cells) | slightly higher | | 1.5e-5 (Mann-Whitney) |
| Preferred depth across orientation bins (depth & orientation selective, n = 167) | | | 0.66 (Kruskal-Wallis) |

## Conclusions (preliminary)

1. **Multiplexed, not separate sub-populations.** Depth-selective cells are as likely as other cells to be
   SF/TF tuned or orientation selective (slightly more often SF/TF tuned / grating responsive). The
   overlap in A is close to what independent assignment predicts.
2. **Orientation and direction preferences of depth-selective cells match the rest of the population**,
   and preferred orientation is unrelated to preferred depth.
3. **No significant relation between preferred grating speed and preferred optic flow speed or depth**
   with the thesis selection and stats (n = 31, two mice). An earlier variant (depth-selective instead of
   RS/OF-tuned cells, R² > 0.1, fits on bounds removed, Spearman across cells) gave ρ = 0.31 for OF and
   -0.29 for depth (p < 0.001, n = 114 / 137), so this relation depends on selection and is not robust.
4. Caveats: only 2 mice (the hierarchical bootstrap resamples between two mice, so its CIs are wide);
   the SF/TF criterion is in-sample; the thesis numbers (Venn 9 / 16 / 42 / 33%) came from older traces
   (ast-neuropil dF/F) and an older fit file, so they are not expected to match.

## Open points

- `P_RESPONSIVE` (0.01) differs from `P_SELECTIVE` (0.05) in the model-free section without a principled
  reason; suggested to set both to 0.05 (not done yet).
- A cross-validated SF/TF criterion (fit on 4 recordings, predict the 5th, one-sided Spearman as for depth)
  would put both selections on the same footing.

## Changes made along the way

- 2026-10-07, copied to `/Volumes/BlackPasspo` from NEMO for the 30 SFTF recordings: `FrameLog.csv`,
  `NewParams.csv` (raw) and `harpmessage.npz`, `monitor_frames_df.pickle` (processed). Local dF/F traces
  match NEMO (md5).
- cottage_analysis (`reviews`):
  - `ead762b`: `synchronisation.load_imaging_data` reads `dff.npy` or `dff_ast.npy` according to the
    dataset's `ast_neuropil` attribute (it always read `dff_ast.npy`).
  - `325d16b`: `fit_gaussian_blob.fit_sftf_tuning` passes the response as a numpy array (it failed with the
    current `common_utils.iterate_fit`).
  - uncommitted: cross-validation in `fit_sftf_tuning` (`k_folds`, `fold_col`, `n_jobs`; `sftf_test_*`
    columns), `depth_SFTF.main(k_folds=5, n_jobs=1)` with one fold per recording, and
    `tests/tests_analysis/test_fit_sftf_tuning.py`.
- 2026-10-07, reproduction test on NEMO (job 60113469, PZAH10.2f_S20230914): the production
  `neurons_df_sftf.pickle` is reproduced exactly from the ast-neuropil-corrected dF/F (`dff_ast.npy`), not
  from the non-ASt `dff.npy` used for the depth `neurons_df`. Details in
  [NEURONS_DF_SFTF.md](NEURONS_DF_SFTF.md#known-issues). This README's results use the notebook's own refit
  on `dff.npy`, so they are not affected.
