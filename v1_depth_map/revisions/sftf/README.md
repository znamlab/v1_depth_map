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
- Example neurons (`fig_sftf_examples.pdf`, `figure_utils/sftf_examples.py`, layout of the thesis example
  figure): PZAH10.2f_S20230914 ROI 447 (depth selective, preferred depth 2.7 m, SF/TF tuned, dsi 0.90) and
  PZAH10.2f_S20230817 ROI 544 (thesis example; SF/TF tuned, not depth selective: depth r = 0.06). The thesis
  depth-selective example, ROI 229 of PZAH10.2f_S20230817, is not `iscell`: its 82-pixel mask touches the edge
  of the registered field of view. The `iscell` labels of these sessions were rewritten on 5-9 January 2024,
  presumably by manual curation: they do not follow the suite2p classifier probability. They are the same
  labels as in the thesis.
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

## The thesis figure (`fig_sftf_summary.pdf`): criteria verified

Applying the criteria of `yiran-phd-thesis/yh_thesis/depth/SFTF.ipynb` to the tables it saved
(`/nemo/lab/znamenskiyp/home/shared/presentations/yiran_thesis/ver{1,2}/rsof/sftf/neurons_df_all.pickle` and
`neurons_df_sftf_all.pickle`, November 2024 and January 2025, identical results) reproduces the PDF exactly
(checked 2026-10-09, NEMO job 60363540):

| panel | cells | criterion | n |
|---|---|---|---|
| A | all neurons | `iscell` | 3012 |
| A | depth selective | `iscell` & `depth_tuning_test_spearmanr_rval_closedloop` > 0.1 & `..._pval_closedloop` (two-sided) < 0.05 | 758 |
| A | SF/TF tuned | `iscell` & in-sample `rsq` > 0.05 | 1727 |
| A | result | depth only 9%, both 16%, SF/TF only 42%, neither 33% | |
| B, C | SF/TF tuned & RS/OF tuned (`rsof_test_rsq_closedloop_g2d` > 0.02) | not restricted to depth-selective cells | 43 (Spearman ρ = 0.38 with OF, −0.32 with depth, across cells) |
| D, E | SF/TF tuned & depth selective | | 477 |
| F | D, E cells with `dsi` > 0.6 | | 193 |

The thesis grating fits (`neurons_df_sftf_all.pickle`) are value-for-value identical to the 2026-02-24
`neurons_df_sftf.pickle` files (now in the backups), i.e. fits on ast-neuropil-corrected dF/F.

## Depth-selective counts: criterion vs data (checked 2026-10-09)

Same 3012 `iscell` neurons, depth columns from the thesis-era `neurons_df` (in the thesis `neurons_df_all`,
ver2, January 2025) or the current `neurons_df.pickle` (2026-02-13):

| | old rule (r > 0.1 & two-sided p < 0.05) | new rule (one-sided p < 0.05) |
|---|---|---|
| thesis-era `neurons_df` | 758 | 906 |
| current `neurons_df` | 549 | 660 |

The new rule adds cells on both (as it must: the old rule implies the new one). The drop comes from the
data: held-out Spearman r is lower in the current `neurons_df` (median 0.015 vs 0.055, 63% of cells lower,
r between versions 0.77), in every session (fraction depth tuned with the new rule: 0.13→0.08, 0.34→0.18,
0.27→0.24, 0.23→0.17, 0.44→0.31, 0.39→0.33). With the new rule, 538 cells are depth tuned in both, 368
only in the thesis-era data and 122 only in the current data. The thesis-era fits predate the
`ast_neuropil=False` traces (2025-09-15), so the difference is ASt vs non-ASt dF/F and/or pipeline changes
since 2024; separating the two needs a rerun of the current depth fit on the ASt traces.

## Trace variants (2026-10-10)

A quick check, removed afterwards: the figure was recomputed from two other traces, with grating fit and
depth tuning refitted by the current pipeline functions. RS/OF (B, C selection) and `iscell` were kept from
the main `neurons_df`. The script (`precompute_data/sftf_trace_variants.py`) is in
v1_depth_map `9e642cb` and was removed in the next commit, with the notebook switch and the outputs. Its `dff`
variant reproduced the production fits exactly.

| | dF/F, `ast_neuropil=False` (main) | ASt dF/F | spikes (`ast_neuropil=False`) |
|---|---|---|---|
| depth selective | 660 (22%) | 888 (29%) | 470 (16%) |
| SF/TF tuned (cross-validated) | 1455 (48%) | 2127 (71%) | 1033 (34%) |
| both (expected if independent) | 13% (11%) | 23% (21%) | 6% (5%) |
| B: grating speed vs OF (n) | ρ = 0.26, p = 0.39 (35) | ρ = 0.41, p = 0.11 (51) | ρ = 0.12, p = 0.55 (26) |
| C: grating speed vs depth | ρ = −0.19, p = 0.45 | ρ = −0.32, p = 0.14 | ρ = 0.04, p = 0.95 |
| responsive, depth / other | 60% / 50%, p < 0.0001 | 75% / 69%, p = 0.003 | 54% / 46%, p = 0.001 |
| orientation selective, depth / other | 25% / 26%, p = 0.76 | 29% / 30%, p = 0.56 | 23% / 26%, p = 0.18 |
| direction selective, depth / other | 14% / 12%, p = 0.14 | 15% / 13%, p = 0.11 | 14% / 11%, p = 0.09 |
| preferred orientation, depth vs other (χ²) | p = 0.87 | p = 0.019 | p = 0.47 |
| preferred direction (χ²) | p = 0.32 | p = 0.22 | p = 0.36 |
| gOSI (Mann-Whitney) | p = 0.13 | p = 0.019 | p = 0.009 |
| gDSI (Mann-Whitney) | p < 0.0001 | p < 0.0001 | p = 0.74 |
| preferred depth across orientation bins (KW) | p = 0.66 (167) | p = 0.92 (257) | p = 0.22 (108) |

- The conclusion holds for all three: depth and grating tuning overlap about as expected if independent
  (slightly more), orientation selectivity does not differ between depth-selective and other cells, and
  preferred orientation is unrelated to preferred depth. No B or C correlation is significant.
- Small differences that are not consistent across traces: preferred orientation distribution
  (p = 0.019 with ASt only), gOSI (p = 0.019 ASt, 0.009 spikes, 0.13 main), gDSI (main and ASt only).
- **The depth-selective count depends on the traces, not on pipeline changes**: the current pipeline on
  ASt traces gives 888 depth-selective cells, close to the thesis-era 906 (see the section above), against
  660 on `ast_neuropil=False` dF/F.
- ASt dF/F has a few large negative trial responses (down to −70 in PZAH10.2f_S20230912), presumably
  neuropil over-subtraction.

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
