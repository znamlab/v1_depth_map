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
- Trial response window (2026-10-15): mean dF/F over `[onset + 8, next_onset)` volumes, i.e. skipping the
  first 0.53 s (the volume closest to 0.5 s), in all trial-averaged SF/TF analyses
  (`skip_first_n_volumes=8` default of `gratings.generate_trials_df`, so `neurons_df_sftf`; `LAG = 8` in the
  notebook; `SKIP = 8` in `precompute_data/sftf_trial_average_subsets.py`).
  Chosen from [`sftf_onset_profile.ipynb`](sftf_onset_profile.ipynb): gratings are back to back, there is no
  onset transient, the response reaches 50% of its plateau at 0.33 s and the previous grating's response
  decays to 50% 0.34 s after its offset (540 ROIs, ANOVA p < 0.01 and preferred-condition dF/F > 0.3).
  Figures in `<processed>/v1_manuscript_figures/ver_rev1/sftf_onset_profile/`.

## Results

| | |
|---|---|
| Depth selective | 660 / 3012 (22%) |
| SF/TF tuned | 1167 / 3012 (39%) |
| Both | 305 (10%); 8.5% expected if independent |
| SF/TF tuned among depth-selective / other cells | 46% / 37% (χ² p = 1e-5) |
| RS/OF tuned (B) | figure 2 empirical-null threshold (test R² > 0.0014): 527; SF/TF and RS/OF tuned: 358 |
| B: grating speed vs preferred OF (SF/TF and RS/OF tuned) | ρ = 0.21, 95% CI [-0.18, 0.35], bootstrap p = 0.21 (n = 358) |
| C: grating speed vs preferred depth (SF/TF tuned and depth selective) | ρ = -0.19, 95% CI [-0.37, -0.06], bootstrap p = 0.005 (n = 390) |
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

## RF shape vs grating tuning (2026-10-10)

Does the sphere RF shape predict grating tuning? (A) does a red centre with blue flanks on either side
prefer gratings with bars parallel to the flanks; (B) are asymmetric RFs more direction selective?
Notebook `figures/fig_reviewer_rf_sftf.ipynb`, helpers `figure_utils/rf_shape.py`, outputs in
`<processed>/v1_manuscript_figures/ver_rev1/fig_reviewer_rf_sftf/`. Exploratory, no stats exported.

- RF fits checked first: closed-loop dF/F and spike RFs (`neurons_df_spks.pickle`, made 5-6 Oct after the τ
  re-deconvolution) are present and filled for every ROI of the 6 sessions.
- Spike RFs, fold mean at the RF peak depth (16 × 24 map, 5° pixels). Metrics from circular moments of the
  negative surround in a 20° disk around the peak, in z of the ipsilateral RF: flank axis, anisotropy,
  asymmetry (0 symmetric, 1 single flank) and its direction, flank depth per side; centre elongation;
  2D Gabor fit.
- Grating convention, verified in the stimulus workflow (`vis-stim-2d/run_SFTF.bonsai`, same in the
  2023-09-11 version `fa05b40` used for these sessions) and BonVision: `Angle` is the wave vector (bars at
  `Angle` + 90°, 0 = vertical bars), the phase grows with time so bars drift towards `Angle` + 180°. The
  grating is drawn in an orthographic view (x 0-120°, y ±50°) wrapped on the sphere (`SphereMapping`):
  x = azimuth from the front increasing to the right, y = elevation up, the same axes as the RF
  reconstruction. So grating and RF angles share one frame ("texture y = elevation" below is the correct
  convention; the mirrored one is a control). E.g. `Angle` 0 drifts towards lower azimuth (back to front).
  This matches the thesis (methods 2.4.10: "0° represented vertical gratings moving forward"; Fig. 5.10F
  caption: 0/180° vertical drifting forward/backward, 90/270° horizontal drifting downward/upward). The
  thesis and `cottage_analysis.plotting.grating_plots` polar plots are labelled in `Angle` units drawn
  counterclockwise from the right, i.e. they draw the wave vector: in the RF-map frame (azimuth to the right,
  elevation up) the drift points the opposite way (90° drawn up = drifting down). Panels E-F of
  `fig_reviewer_gratings` follow the same labels and need the thesis caption wording.
- Cells: 679 / 3012 have a significant RF, 396 with the peak ≥ 4 pixels from the map edges (needed so the
  surround is not truncated). Of these, 214 flanked (deepest flank ≥ 2 z), 108 bilateral, 95
  orientation-selective, 50 direction-selective, 186 SF/TF tuned. Bilateral and orientation selective: 23.

| Test | n | Result |
|---|---|---|
| A1 flank axis vs preferred orientation (weighted mean cos 2Δ, within-session permutation) | 68 | 0.16, p = 0.11 (mirrored control: 0.11, p = 0.35) |
| A2 centre long axis + 90° vs preferred orientation | 68 | 0.19, p = 0.06 / 0.17, p = 0.07 |
| A3 Gabor orientation (R² ≥ 0.5) | 55 | p = 0.31 / 0.42 |
| A4 flank axis vs fitted orientation, SF/TF-tuned | 131 | p = 0.13 / 0.05 |
| A5 bilateral → horizontal bars | 23 vs 72 | 17% vs 28%, Fisher p = 0.41 |
| A5 flanks above and below → vertical bars | 12 vs 83 | 58% vs 25%, Fisher p = 0.037 (one of six uncorrected A5 tests) |
| B1 asymmetry vs fitted DSI (hierarchical bootstrap) | 103 | ρ = 0.09 [-0.34, 0.34], p = 0.68 |
| B1 asymmetry vs gDSI, responsive | 119 | ρ = -0.01, p = 0.98 |
| B2 asymmetry, direction selective vs not | 29 vs 90 | medians 0.36 vs 0.42, p = 0.26 |
| B3 preferred motion vs asymmetry direction (asym ≥ 0.3, direction selective) | 17 | p = 0.36 / 0.52 |
| B4 asymmetry direction vs local optic flow (V-test) | 145 | V = 0.03, p = 0.61 |
| B4 asymmetry along azimuth vs RF azimuth | 145 | ρ = -0.53, p = 1e-11 |
| B4 asymmetry along elevation vs RF elevation | 145 | ρ = -0.31, p = 0.0002 |

Conclusions so far:
1. **No reliable link between flank geometry and preferred orientation.** All alignment stats are positive
   but none survives correction. The one nominal hit points the other way from the side-band hypothesis:
   cells with flanks above and below the centre prefer *vertical* bars (7 of 12), which would fit end
   inhibition, not horizontal-grating tuning. With 12 cells this is not evidence.
2. **RF asymmetry does not predict direction selectivity** (B1-B3 null).
3. **The asymmetry is largely set by RF position.** The deeper flank points towards the middle of the
   mapped field (low azimuth for RFs at large azimuth, upwards for low RFs and downwards for high ones),
   which suggests stimulus geometry (sphere density / size across the screen, or the ridge fit) rather than
   a property of the cell. Any use of RF asymmetry needs to control for this.
4. Limits: the RF maps are coarse (5° pixels resolve ≤ 0.1 cpd, and Laplacian smoothing favours low SF;
   median Gabor SF ≈ 0.01 cpd, so most RFs are blobs); blue regions are suppression by spheres, which
   need not be OFF subregions (sphere polarity not checked); the fold-split "reliability" in the notebook is not independent (fold
   models share most of their training data).

## Frame-level analyses moved to lag 0 (2026-10-14)

All frame-level analyses (running-speed models, speed-binned responses, model-free TF shift) now pair the stimulus
and running speed of a frame with the dF/F of the **same frame** (`FRAME_LAG = 0` in the notebook, `LAG = 0` in
`precompute_data/sftf_running_models.py`). That is the convention of every manuscript RS/OF fit. Before this change
they used a 5-frame lag, taken from the exploratory RS/OF lag sweep. **The sections below dated before 2026-10-14
report lag-5 results.** Those results and their NEMO run directories have been deleted.

**Lag sweep.** The grating lag sweep is in `revisions/rf_spike_tau/lag_choice.ipynb`, run with
`sftf_lag_sweep.py`. It covers the 100 best trial-level-fitted SF/TF-tuned cells and fits `base`, `rsg` and
`rsg_tf` at lags −2 to 15 frames. On the first 68 cells:
- The best lag is 4 frames (median; IQR 3–5), for all three models.
- Held-out R² as a fraction of the cell's best is 0.90 at lag 0 and 0.99 at lags 4–5.
- Preferred SF and TF do not depend on the lag (median range over lags 0–8: 0.02 and 0.08 octaves).
- `a_tf` is positive at every lag (median 0.08 at lag 0, 0.09 at lag 5).

**Running-speed models at lag 0.** These are all 3012 `iscell` cells, refitted on NEMO
(`/camp/home/blota/code/sftf_running_models_lag0`), with results in `running_models/`. 722 cells are significant in
at least one model (731 at lag 5).

| | All | Depth selective | Other | Depth > other |
|---|---|---|---|---|
| Best model `rsg_tf` | 17.9% | 22.9% | 16.0% | p = 0.10 (lag 5: 30.5% vs 20.0%, p = 0.025) |
| `a_tf`, `rsg_tf` significant (median) | 0.095 (n = 566, p < 0.001) | 0.168 (p < 0.001) | 0.075 (p = 0.001) | **p = 0.002** (lag 5: p = 0.19) |
| `a_sf`, `rsg_sf` significant (median) | −0.014 (p = 0.005) | −0.014 | −0.013 | p = 0.41 |
| Held-out R² `rsg_tf` − `rsg` (median) | −1e-4 (worse, p < 0.001) | −1e-4 (p = 0.20) | −2e-4 (p < 0.001) | p = 0.005 |

The p-values against 0 come from a hierarchical bootstrap. The depth vs other p-values permute depth labels within
session. The lag-5 values are on the same cells (the lag-0 vs lag-5 comparison of 2026-10-14).

**Model-free TF shift at lag 0** (`running_models/tf_shift.pickle`, regenerated). Shifters among the 1454 SF/TF-tuned
cells:

| | Depth selective | Other |
|---|---|---|
| Lag 0 | 55/389 (14.1%) | 132/1065 (12.4%) |
| Scrambled running speed (sham) | 9.0% | 7.4% |

Fisher one-sided p = 0.21 for lag 0 and 0.19 for the sham. At lag 5 the fractions were 18.5% vs 12.2%, and the
within-session permutation gave p = 0.001.

**What changes with the lag:**
- **Holds at both lags:**
  - Running shifts the preferred TF upwards: `a_tf` > 0 in depth-selective and other cells.
  - The SF shift is small (about −0.014) and does not differ between groups.
  - Adding the TF shift does not improve held-out R² overall, and it costs less in depth-selective cells.
  - The trial-level results are not lag dependent (panels A–D, preferences, exported stats).
- **Depends on the lag:**
  - At lag 0 the larger `a_tf` of depth-selective cells becomes significant.
  - At lag 0, two lag-5 results are lost: more depth-selective cells best fit by `rsg_tf`, and more model-free
    TF shifters among depth-selective cells.
  - The depth-selective vs other comparison should therefore be reported as lag sensitive.

## Running-speed models: gain vs preference shift (10 chosen cells, 2026-10-10)

To test whether running primarily **shifts** grating tuning (TF or SF preferences) or mainly modulates responses via multiplicative **gain** or additive **offset**, 9 candidate models were evaluated on 10 high-SNR cells across all 6 sessions and both mice (3 depth-tuned, 7 other, including thesis example PZAH10.2f_S20230817 ROI 544).
Fits used nested warm starts and 5-fold leave-one-recording-out cross-validation (`precompute_data/sftf_running_model_selection.py`).

| Session | ROI | Depth | base | gain | offset | gain_offset | tf | gain_offset_tf | gain_offset_sf | gain_offset_tf_sf | gain_offset_speed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PZAH10.2d_S20230920 | 220 | False | 0.6633 | 0.6836 | 0.6652 | 0.6836 | 0.6751 | 0.6829 | 0.6827 | 0.6825 | 0.6826 |
| PZAH10.2d_S20230922 | 171 | False | 0.6618 | 0.6603 | 0.6617 | 0.6603 | 0.6632 | 0.6614 | 0.6597 | 0.6606 | 0.6596 |
| PZAH10.2f_S20230817 | 62 | True | 0.5283 | 0.5276 | 0.5285 | 0.5277 | 0.5317 | 0.5306 | 0.5294 | 0.5303 | 0.5282 |
| PZAH10.2f_S20230817 | 259 | True | 0.5113 | 0.5288 | 0.5142 | 0.5288 | 0.5239 | 0.5279 | 0.5287 | 0.5256 | 0.5285 |
| PZAH10.2f_S20230817 | 214 | False | 0.4748 | 0.5072 | 0.4792 | 0.5072 | 0.5011 | 0.5141 | 0.5118 | 0.5158 | 0.5144 |
| PZAH10.2f_S20230817 | 544 | False | 0.4858 | 0.5016 | 0.4881 | 0.5016 | 0.5006 | 0.5053 | 0.5075 | 0.5081 | 0.5098 |
| PZAH10.2f_S20230912 | 391 | False | 0.5908 | 0.5759 | 0.5855 | 0.5754 | 0.5741 | 0.5701 | 0.5735 | 0.5674 | 0.5757 |
| PZAH10.2f_S20230914 | 8 | True | 0.5532 | 0.5559 | 0.5538 | 0.5558 | 0.5549 | 0.5559 | 0.5551 | 0.5589 | 0.5532 |
| PZAH10.2f_S20230924 | 206 | False | 0.5694 | 0.5773 | 0.5705 | 0.5773 | 0.5700 | 0.5771 | 0.5778 | 0.5774 | 0.5779 |
| PZAH10.2f_S20230924 | 269 | False | 0.5659 | 0.5820 | 0.5667 | 0.5822 | 0.5715 | 0.5808 | 0.5810 | 0.5801 | 0.5807 |

Held-out Test R² Gain over Base (× 1000):

| Session | ROI | gain | offset | gain_offset | tf | gain_offset_tf | gain_offset_sf | gain_offset_tf_sf | gain_offset_speed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PZAH10.2d_S20230920 | 220 | +20.34 | +1.97 | +20.32 | +11.84 | +19.65 | +19.41 | +19.20 | +19.36 |
| PZAH10.2d_S20230922 | 171 | -1.50 | -0.11 | -1.44 | +1.39 | -0.39 | -2.05 | -1.12 | -2.17 |
| PZAH10.2f_S20230817 | 62 | -0.68 | +0.23 | -0.57 | +3.48 | +2.34 | +1.15 | +2.01 | -0.06 |
| PZAH10.2f_S20230817 | 259 | +17.55 | +2.89 | +17.56 | +12.58 | +16.66 | +17.40 | +14.31 | +17.22 |
| PZAH10.2f_S20230817 | 214 | +32.43 | +4.36 | +32.42 | +26.32 | +39.30 | +37.02 | +40.97 | +39.60 |
| PZAH10.2f_S20230817 | 544 | +15.87 | +2.34 | +15.86 | +14.87 | +19.51 | +21.74 | +22.33 | +23.99 |
| PZAH10.2f_S20230912 | 391 | -14.95 | -5.34 | -15.43 | -16.77 | -20.70 | -17.30 | -23.39 | -15.15 |
| PZAH10.2f_S20230914 | 8 | +2.66 | +0.53 | +2.59 | +1.63 | +2.68 | +1.83 | +5.64 | -0.03 |
| PZAH10.2f_S20230924 | 206 | +7.92 | +1.14 | +7.85 | +0.63 | +7.70 | +8.40 | +7.95 | +8.54 |
| PZAH10.2f_S20230924 | 269 | +16.12 | +0.80 | +16.23 | +5.62 | +14.90 | +15.07 | +14.14 | +14.78 |

Conclusions:
1. **Multiplicative gain dominates**: In all cells where running modulates responses, `gain` alone produces large, positive held-out R² gains over `base` (median +16.1 × 10⁻³ in modulated cells). Additive offset (`offset`) contributes minimally (median +1.5 × 10⁻³; offset slope median $b = +0.002$).
2. **Preference shift is largely an artifact confounded by gain**: The shift-only model (`tf`) showed apparent positive slopes (median $a_{\text{tf}} = +0.33$). However, once gain is allowed (`gain_offset_tf`), the fitted TF slope drops by more than half (median $a_{\text{tf}} = +0.17$), and held-out cross-validated R² fails to improve over `gain_offset` (median $\Delta R^2 = -0.6 \times 10⁻³$ due to overfitting to training noise).
3. **Simulation sanity check**: Synthetic frames generated from `gain_offset` (ground truth $a_{\text{tf}} = 0$) recover $\hat{a}_{\text{tf}} \approx 0$ ($-0.007$ and $+0.002$), while synthetic frames with a true shift ($a_{\text{tf}} = 0.5$) recover $\hat{a}_{\text{tf}} = 0.497$ and $0.491$, confirming that gain and shift parameters are fully separable with this stimulus.
4. **Robustness checks**: The conclusion holds across LAG 2, LAG 8, running speed smoothed over 0.5 s, and trial-level aggregated responses. In all variants, `gain_offset` reliably improves over `base` (median $+7.2$ to $+13.1 \times 10⁻³$), whereas adding `tf` shift on top of `gain_offset` yields negative median held-out gain ($-0.26$ to $-4.13 \times 10⁻³$).
5. **Caveats**: Evaluated on 10 high-SNR cells to avoid the population noise floor; gratings were open-loop.

## Full-population running-speed models: gain vs preference shift (1,455 cells, 2026-10-11)

Following the 10-cell detailed selection, all 1,455 SF/TF-tuned `iscell` ROIs across all 6 sessions were fitted on NEMO (Job `60485107`, `precompute_data/sftf_running_models.py`) with 5-fold leave-one-recording-out cross-validation comparing:
- `base`: unmodulated grating tuning
- `gain`: multiplicative gain modulation ($A \propto \text{RS}^g$)
- `tf`: temporal frequency shift ($\text{TF}_0 \propto \text{RS}^{a_\text{tf}}$)
- `gain_tf`: combined multiplicative gain and temporal frequency shift
- `sf`: spatial frequency shift ($\text{SF}_0 \propto \text{RS}^{a_\text{sf}}$)
- `both`: combined spatial and temporal frequency shifts

Outputs are merged into `<processed>/v1_manuscript_figures/ver_rev1/fig_reviewer_gratings/running_models/*_sftf_running_models.pickle` and plotted in `running_speed_models.pdf`.

### Model comparison (% best-fitting model by held-out test $R^2$)

| Model | Free parameters | All cells ($n=1455$) | Selected ($n=564$) | Selected, not depth selective ($n=399$) | Selected, depth selective ($n=165$) |
| --- | --- | --- | --- | --- | --- |
| **`base`** | None | 29.7% | 23.0% | 25.1% | 18.2% |
| **`gain`** | $g$ | **22.1%** | **23.8%** | **25.3%** | 20.0% |
| **`tf`** | $a_\text{tf}$ | 14.4% | 15.4% | 14.8% | 17.0% |
| **`gain_tf`** | $g, a_\text{tf}$ | 13.7% | 18.4% | 15.3% | **26.1%** |
| **`sf`** | $a_\text{sf}$ | 12.6% | 10.8% | 10.5% | 11.5% |
| **`both`** | $a_\text{sf}, a_\text{tf}$ | 7.4% | 8.5% | 9.0% | 7.3% |

### Parameter distributions and generalization (selected cells, $n = 564$)

Hierarchical bootstrap across mice and sessions (1000 resamples):

| Parameter / metric | Model | Median | 95% CI | Bootstrap $p$ | % Positive |
| --- | --- | --- | --- | --- | --- |
| Gain slope $g$ | `gain` | **+0.101** | [+0.030, +0.168] | $p < 0.001$ | 68.1% |
| Gain slope $g$ | `gain_tf` | **+0.070** | [+0.012, +0.128] | $p = 0.004$ | 61.4% |
| TF slope $a_\text{tf}$ | `tf` (alone) | **+0.183** | [+0.040, +0.244] | $p < 0.001$ | 72.7% |
| TF slope $a_\text{tf}$ | `gain_tf` (co-fitted) | **+0.109** | [+0.048, +0.175] | $p < 0.001$ | 66.0% |
| SF slope $a_\text{sf}$ | `sf` | -0.007 | [-0.024, +0.007] | $p = 0.240$ | 45.6% |
| Held-out $\Delta R^2$ over base | `gain` | **+2.0 × 10⁻⁴** | [-1.0 × 10⁻⁴, +8.0 × 10⁻⁴] | $p = 0.352$ | 56.6% |
| Held-out $\Delta R^2$ over base | `tf` | +0.2 × 10⁻⁴ | [-2.0 × 10⁻⁴, +6.0 × 10⁻⁴] | $p = 1.000$ | 50.5% |
| Held-out $\Delta R^2$ over base | `gain_tf` | **+2.3 × 10⁻⁴** | [-2.0 × 10⁻⁴, +11.0 × 10⁻⁴] | $p = 0.386$ | 55.5% |
| Held-out $\Delta R^2$ over base | `sf` | -0.9 × 10⁻⁴ | [-2.0 × 10⁻⁴, -0.0 × 10⁻⁴] | $p = 0.010$ | 40.2% |

### Key population conclusions

1. **Multiplicative gain is the principal running-related modulation**: `gain` is the most common winning modulated model across the population (22.1% all, 23.8% selected), outperforming both `tf` (14.4%) and `sf` (12.6%).
2. **Co-fitting gain attenuates apparent TF preference shifts by 40%**: In the shift-only model `tf`, the population exhibits an apparent median shift $a_\text{tf} = +0.183$. Once multiplicative gain is co-fitted (`gain_tf`), the temporal frequency slope drops to $a_\text{tf} = +0.109$, confirming that a large fraction of apparent TF shifts in uncorrected models arises from running-dependent gain.
3. **The residual TF shift is significant as a parameter but does not generalise overall**: in `gain_tf` the median $a_\text{tf} = +0.109$ ($p < 0.001$, ×1.08 per doubling of running speed, vs ×1.13 in `tf`), but adding the TF shift to `gain` lowers held-out $R^2$ in the selected cells (better in 40%, Wilcoxon $p = 0.0008$). The shift is concentrated in depth-selective cells (next section). *(Corrected 2026-10-11; first version said the residual shift was significant without the held-out comparison.)*
4. **Spatial frequency is completely unaffected by running speed**: Median $a_\text{sf} \approx 0$ ($p = 0.24$), and adding SF shift degrades held-out $R^2$ ($p = 0.010$).

### Depth-selective cells: gain in all cells, TF shift in depth-selective cells (2026-10-11)

Hypothesis (from figure 3): every cell is gain modulated, and depth-selective cells also shift their preferred TF
with running speed. `gain_tf` model (gain $g$ and TF shift $a_\text{tf}$ fitted together), selected cells (significant in
at least one model): 165 depth-selective, 399 other. Depth > other tested with permutations of the depth labels within
session, and within session × base-fit-quality tertile (one-sided, 10 000 permutations); within-group values against 0
with Wilcoxon (cells) and hierarchical bootstrap (mice, sessions). Notebook section "Hypothesis: all cells are gain
modulated, depth-selective cells also shift their TF preference", figure `depth_cells_tf_shift.pdf`.

| Measure | Depth selective | Other | p (within session) | p (within session × fit quality) |
|---|---|---|---|---|
| gain $g$ (median) | 0.037 (Wilcoxon p = 0.013, bootstrap p = 0.13) | 0.082 (p < 1e-4, bootstrap p = 0.016) | 0.88 (depth not larger) | 0.71 |
| TF shift $a_\text{tf}$ (median) | 0.161 (p = 0.0002, bootstrap 0.002) | 0.101 (p < 1e-4, bootstrap 0.002) | **0.015** | **0.010** |
| held-out $R^2$, `gain_tf` − `gain` (median) | +0.00000 (Wilcoxon p = 0.29) | −0.00022 (p < 1e-4, worse) | **0.003** | **0.008** |
| best model `tf` or `gain_tf` | 43% | 30% | **0.018** | **0.038** |

Per session, the TF shift is larger in depth-selective cells in 4 of 6 sessions (S20230920 0.33 vs 0.15, S20230922
0.16 vs 0.02, S20230817 0.19 vs 0.14, S20230914 0.25 vs 0.18; reversed slightly in S20230912 and S20230924).

Interpretation:
- **Gain is present in both groups** and not larger in depth-selective cells (median $g$ 0.04 vs 0.08; in
  depth-selective cells it is significant across cells but not in the hierarchical bootstrap).
- **The TF shift is larger in depth-selective cells**, and it is the part that generalises there: adding it to gain does
  not hurt held-out prediction in depth-selective cells (median 0), while it overfits in the other cells (median
  −2.2e-4). Controlling for fit quality does not change this (depth-selective cells are not simply better fitted).
- So the data fit the hypothesis that all cells are gain modulated and depth-selective cells additionally shift their
  TF preference with running speed, with two caveats: in depth-selective cells the shift still does not *improve*
  held-out $R^2$ over gain alone (median gain ≈ 0), and $g$ and $a_\text{tf}$ trade off within a fit (both raise the
  high-TF responses at fast running), so a smaller $g$ in depth-selective cells may partly be absorbed by $a_\text{tf}$.
  2 mice, 6 sessions.


### Deconvolved spikes at neural timescales: lag 2 frames (~131 ms) (2026-10-12)

To test whether the gain vs shift separation holds on instantaneous neural event estimates without GCaMP decay dynamics, the four symmetric running-speed models (`base`, `gain`, `gain_tf`, `gain_sf`) were fitted to Suite2p deconvolved spikes (`spks.npy`, `ast_neuropil=False`) using the peak response lag from the receptive field analysis (`LAG = 2` frames, ~131 ms at 15.2 Hz; matching `revisions/rf_multidepth/`). All 1,455 SF/TF-tuned cells across the 6 sessions were fitted on NEMO with both 5-fold leave-one-recording-out cross-validation (held-out $R^2$) and full-dataset fits (best parameter estimates). Results in `running_models_spikes/` and `running_speed_models_spikes.pdf`.

#### Winning model breakdown (held-out cross-validated $R^2$)

| Model | Terms | All cells ($n = 1455$) | Running-modulated ($n = 757, 52\%$) | Depth selective ($n = 390$) | Other cells ($n = 1065$) |
|---|---|---|---|---|---|
| **`base`** | None | 48.0% (698) | — | 39.2% (153) | 51.2% (545) |
| **`gain`** | $g$ | **24.3%** (354) | **46.8%** (354) | **26.9%** (105) | **23.4%** (249) |
| **`gain_tf`** | $g, a_\text{tf}$ | **16.2%** (236) | **31.2%** (236) | **21.0%** (82) | 14.5% (154) |
| **`gain_sf`** | $g, a_\text{sf}$ | 11.5% (167) | 22.1% (167) | 12.8% (50) | 11.0% (117) |

#### Parameter distributions (full-dataset fits, all $n = 1455$ cells)

| Parameter / metric | Model | Median | Mean | Wilcoxon $p$ vs 0 | dF/F correlation (Spearman $r$, $n=1455$) |
|---|---|---|---|---|---|
| Gain slope $g$ | `gain` | **+0.0215** | **+0.2483** | $p = 2.2 \times 10^{-19}$ | **$r = +0.790$** ($p < 10^{-300}$) |
| TF slope $a_\text{tf}$ | `gain_tf` | **+0.1089** | **+0.1599** | $p = 1.6 \times 10^{-8}$ | **$r = +0.470$** ($p = 1.1 \times 10^{-80}$) |
| SF slope $a_\text{sf}$ | `gain_sf` | -0.0147 | +0.0140 | $p = 0.253$ (not sig.) | **$r = +0.493$** ($p = 4.4 \times 10^{-90}$) |

#### Key findings on spikes:
1. **Multiplicative gain remains the dominant running modulation**: exactly as in dF/F (24.5%), `gain` is the winning running model in 24.3% of all cells (46.8% of modulated cells).
2. **Robust temporal frequency shift ($a_\text{tf} > 0$)**: preferred TF shifts significantly rightward towards higher frequencies during locomotion (median $+0.109$, mean $+0.160$, $p = 1.6 \times 10^{-8}$), closely matching the dF/F median ($+0.103$).
3. **Flat spatial frequency control ($a_\text{sf} \approx 0$)**: $a_\text{sf}$ is centered at zero ($p = 0.253$), confirming that the positive TF shift is not a non-specific fitting drift.
4. **Tight cell-by-cell agreement**: cell-by-cell slopes correlate strongly between dF/F and spikes ($r = 0.79$ for gain, $r = 0.47$ for TF shift, $r = 0.49$ for SF shift).
5. **Depth-selective enrichment**: `gain_tf` is significantly more likely to win in depth-selective cells (21.0% vs 14.5%, $\chi^2$ $p = 0.003$), with higher median shift ($a_\text{tf} = +0.142$ vs $+0.103$).


### Model-free per-cell test: which cells shift their preferred TF beyond gain? (2026-10-11)

Recommended test of the hypothesis (more direct than the model parameters above, where `g` and `a_tf` trade off).
`precompute_data/sftf_tf_shift.py`, results `running_models/tf_shift.pickle` (+ `tf_shift_sham.pickle`), notebook
section "Which cells shift their preferred TF beyond gain?", figures `tf_shift_enrichment.pdf`, `tf_shift_examples.pdf`.

- Per cell: running frames split at the session's median running speed; TF curves of the slow and fast halves over the
  SF / direction pairs that drive the cell (≥ 0.5 × best pair, same pairs in both halves); preferred TF of each half =
  centre of mass in log2 TF of (curve − minimum), which a gain or an offset does not change; shift = fast − slow.
- Null per cell: 200 surrogates from the cell's fitted `gain` model plus its own residuals circularly shifted within
  each recording (keeps the noise structure and the gain-induced SNR difference between halves, no TF shift).
  Shifter = observed shift above the surrogates, one-sided p < 0.05 (TF increasing with running).
- Calibration: same analysis with the running-speed trace scrambled in time → 8.3% "shifters" in both groups (slightly
  above the nominal 5%: per-cell p values are a little liberal, equally for both groups).

| | Depth selective | Other | Enrichment p (within session) | p (within session × fit quality × preferred TF) |
|---|---|---|---|---|
| All 1454 SF/TF-tuned cells with a defined shift | **72/389 (18.5%)** | 130/1065 (12.2%) | **0.001** | **0.0007** |
| Selected cells (564) | 52/165 (31.5%) | 94/399 (23.6%) | 0.015 | 0.018 |
| Scrambled running speed (all cells) | 8.2% | 8.3% | 0.65 | 0.56 |

- More shifters in depth-selective cells in 5 of 6 sessions. Robust to the pair threshold (0.3: 15.6% vs 10.8%, p = 0.002;
  0.7: 18.7% vs 13.6%, p = 0.002; exploratory runs, not saved).
- The shift is real and not a noise artefact: median +0.083 octaves (fast − slow) against +0.008 for the gain-only
  surrogates; unrelated to the gain `g` (ρ = 0.02), related to `a_tf` of `gain_tf` (ρ = 0.21).
- Depth-selective cells prefer higher TF (median 3.0 vs 1.6 Hz); stratifying on it does not change the enrichment.

Are the shifters particular? Shifters have stronger, better-fitted and more selective grating responses than the other
cells (amplitude, held-out R², gOSI, gDSI; p < 0.001), as expected since a shift is only detectable in strong responses.
Among depth-selective cells, shifters have RS/OF Gaussians closer to the diagonal (median distance to 45°: 27° vs 37°,
Mann-Whitney p = 0.02) and prefer faster running (p = 0.049), but neither survives controlling for fit quality
(permutation within session × fit tertile: p = 0.26 and 0.26). Preferred depth, optic flow and RS/OF elongation do not
differ. So no RS/OF property identifies the shifters beyond depth selectivity itself; the diagonal trend is suggestive.

Relation to the population TF curves ("Population TF and SF tuning per running-speed bin"): both use the same TF curves
over the same SF / direction pairs. The population curves average cells, aligned on the full-fit preferred TF and
normalised by each cell's maximum over bins, so they keep the gain and show the shift only as an asymmetric growth of
the high-TF flank. This test removes the gain from each curve, gives a statistic and a null for every cell, and so says
which cells shift and whether they are enriched in depth-selective cells; the population curves are its group-average
view.

**Conclusion on the hypothesis.** The data support it: all cells are gain modulated, a subset additionally increases
its preferred TF with running speed, and this subset is enriched among depth-selective cells (about 1.5 × the
fraction of other cells; ~10 vs ~4 percentage points above the scrambled-speed rate). Caveats: other cells shift too;
2 mice, 6 sessions; per-cell p values are slightly liberal (calibration 8%), so the comparison between groups, not the
per-cell counts, is the robust result; the shift is modest (median ~0.1-0.2 octave between the slow and fast halves).

## Running-speed Gaussian (`rsg`) vs a quadratic running-speed term (2026-10-13)

The notebook's running-speed models (`rsg`, `rsg_tf`, `rsg_sf`; all 3012 `iscell` cells, dF/F) use a Gaussian in log
running speed, parameterised and bounded as the RS/OF fits of figure 2: preferred speed 0.5 cm/s - 5 m/s,
σ² = exp(log σ²) + 0.25. A monotonic cell needs a wide Gaussian peaking far outside the speeds analysed. With the
peak capped at 5 m/s, steeply increasing cells keep some curvature, and 36% of the cells fitted significantly had
their `rsg` peak at a bound.

**Alternative tested.** The same curves were written as a quadratic in log running speed,
$\ln A = \ell + g \cdot r - c \cdot r^2$, with $0 \leq c \leq 2$ (the same minimum width, σ ≥ 0.5) and |g| ≤ 20.
- There is no bound on the peak: $c = 0$ is exactly the power-law gain.
- Models `quad`, `quad_tf` and `quad_sf` were fitted on all cells (NEMO job 60631747).
- They were started from the gain fit and from peaks at quantiles of running speed, not from `rsg`.
- 2996 of the 3012 cells were done when the job was stopped; 651 were significant in `gain`, `rsg` or `quad`.

**Result: the quadratic term is not better.**
- **Same solution wherever `rsg` is not at a bound.** For the 404 cells where both models peak:
  - preferred speed Spearman 1.000, 98% of cells within 5%;
  - width Spearman 0.999, 98% within 5%.
- **Cells where `rsg` was at its bound (235).** `quad` becomes monotonic ($c = 0$) for 79% of them. Its held-out R² is
  better in 59% of these cells, but only by a median of +4e-5, against a typical held-out R² of about 0.02.
- **Held-out R² over all cells.**

  | Comparison | Median difference | Mean difference |
  |---|---|---|
  | quad - rsg | -7e-7 | +4e-5 |
  | quad_tf - rsg_tf | -2e-6 | +5e-5 |
  | quad_sf - rsg_sf | -6e-7 | +2e-4 |

  These are within optimiser noise.
- **Preference shifts unchanged.**

  | Shift | Spearman between models | Median, `quad` | Median, `rsg` |
  |---|---|---|---|
  | `a_tf` | 0.94 | 0.10 | 0.12 |
  | `a_sf` | 0.95 | -0.015 | -0.015 |

- **`gain` still beats the curved model in about 2/3 of the cells**: 65% against `quad`, 69% against `rsg`. That
  happens because the extra curvature parameter overfits and costs on held-out data. The 5 m/s cap is not the
  reason.

**Decision.** Keep `rsg`, which matches the RS/OF fits. The quadratic models, their code and their results were
removed.

The `gain` vs `rsg` comparison (shape of the running-speed curve, preferred speed vs the RS/OF fit, `a_tf` under
both models; `running_speed_tuning_vs_gain.pdf`) was moved on 2026-10-14 from `fig_reviewer_gratings.ipynb` to
`running_speed_gain_vs_preferred_rs.ipynb` in this folder (personal notes, not for the paper).

## Open points

- `P_RESPONSIVE` (0.01) differs from `P_SELECTIVE` (0.05) in the model-free section without a principled
  reason; suggested to set both to 0.05 (not done yet).
- A cross-validated SF/TF criterion (fit on 4 recordings, predict the 5th, one-sided Spearman as for depth)
  would put both selections on the same footing.

## Changes made along the way

- 2026-10-15, response window bug: the window was `[onset + 2, next_onset + 2)` in both
  `gratings.generate_trials_df` and the notebook's `recording_trials`, so each trial skipped only 0.13 s and
  included the first 0.13 s of the next grating. `sftf_trial_average_subsets.py` shifted both edges by its
  frame lag (0 for dF/F: no skip) and read RS and response from the same frames. Now `[onset + 8, next_onset)`
  everywhere (see Analysis), with the subsets script pairing each response frame with the RS `lag` frames
  earlier. Done the same day on NEMO (run folder `/camp/home/blota/code/sftf_skip8_run`):
  `neurons_df_sftf.pickle` refit (job 60867152, previous files in `backups/`) and trial-average subsets
  (job 60867153, previous local results in `fig_reviewer_gratings/trial_average_subsets_before_skip8/`), copied
  to BlackPasspo; `fig_reviewer_gratings` rerun with rebuilt caches.
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
