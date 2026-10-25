# Drifting gratings: is depth selectivity multiplexed with SF / TF / direction tuning?

Reviewer comment:

> One major open question in my mind is how this sensitivity to depth relates to other sensitivities,
> such as orientation. Are these multiplexed, or are they separate sub-populations? [...] it seems possible
> to derive orientation selectivity of the recorded neurons during (near) stationary epochs.

The figure answering it is `figures/fig_reviewer_gratings.ipynb`; this folder holds the exploratory analyses, the
scripts that only they use, and the notes. Results are in the notebook outputs and in
`stats/generated/stats_fig_reviewer_gratings.yaml`, not here. The review of 2026-10-24 and its decisions are tracked
in [REVIEW_FOLLOWUP.md](REVIEW_FOLLOWUP.md) (temporary).

## Data

- Project `hey2_3d-vision_foodres_20220101`, the only sessions with gratings: PZAH10.2d S20230920, S20230922 and
  PZAH10.2f S20230817, S20230912, S20230914, S20230924 (GCaMP6f).
- 5 `SFTF` recordings per session, after the depth protocol. Each recording shows every condition once:
  6 SF (0.01–0.32 cpd) × 6 TF (0.5–16 Hz) × 8 directions, about 4 s per grating, back to back, no blank.
- Same suite2p ROIs as the depth recordings, so tuning is compared cell by cell. The gratings are separate
  recordings, so all trials are used (no stationary selection for the trial-level analyses).
- Traces: dF/F, `filter_datasets={"anatomical_only": 3, "ast_neuropil": False}`, as every manuscript fit.

## Pipeline (rerun order)

All outputs are under `<processed>/v1_manuscript_figures/ver_rev1/fig_reviewer_gratings/`.

1. `neurons_df_sftf.pickle` per session: `precompute_data/rerun_depth_sftf.py` (+ `.sbatch`, NEMO). How it is made
   and what it contains: [NEURONS_DF_SFTF.md](NEURONS_DF_SFTF.md).
2. `precompute_data/sftf_grating_cells.py [--rebuild]` (local): trial tables (`cache/<session>_sftf_trials.pickle`),
   slim neurons_df (`cache/<session>_neurons_slim.pickle`) and `grating_tuning_cells.csv` (one row per `iscell`
   ROI: model-free tuning, depth and RS/OF tuning, the trial-level fit and the selections).
3. `precompute_data/sftf_running_models.py` (NEMO): `prepare --trace dff --all-cells` (running frames) and
   `prepare --trace dff --all-cells --slow` (frames below 2.83 cm/s, not fitted) → `running_models/frames{,_slow}/<session>.npz`;
   `fit` and `merge` → `running_models/<session>_sftf_running_models.pickle`. The figure needs only the `base`
   model (`fit SESSION ROI --models base`: selection of the population curves); the other models feed the
   exploratory notebook (`--models gain gain_tf gain_sf rsg rsg_tf rsg_sf`, added to the same ROI results).
4. `figures/fig_reviewer_gratings.ipynb`: builds `cache/speed_curves_v8.pickle`, writes `fig_sftf_analysis.{pdf,png}`
   and the stats YAML.
5. `revisions/sftf/sftf_exploratory_analysis.ipynb` (optional).

Code: `figure_utils/sftf.py` (figure helpers, held-out speed curves), `revisions/sftf/sftf_plots.py` (exploratory
plots). Panel A schematic: `figures/schematic_gratings.pdf` (vector, cropped from the Illustrator figure) and its PNG
preview, merged into the figure PDF by `sftf.save_with_vector_image`.

## Definitions and choices

**Trial response.** Mean dF/F over `[onset + 8, next_onset)` imaging volumes (the first 0.53 s are skipped; onset =
change of the parameter-log time `stimulus_harptime`). Chosen from `sftf_onset_profile.ipynb` (2026-10-15): the
gratings are back to back with no onset transient; the response reaches half its plateau about 0.33 s after onset and
the previous grating's response decays to half about 0.34 s after its offset.

**Frame-level analyses** (running-speed models, speed-binned curves): stimulus and running speed of a frame paired
with the dF/F of the same frame (lag 0, the convention of the manuscript RS/OF fits), and the same 8-frame onset skip
as the trial window (`SKIP` in `sftf_running_models.py`, since 2026-10-24). The lag sweep
(`sftf_lag_sweep.py`, `revisions/rf_spike_tau/lag_choice.ipynb`) found a best lag of about 4 frames but preferred
SF and TF independent of the lag, so lag 0 was kept.

**Trial-level SF/TF fit** (`fit_gaussian_blob.fit_sftf_tuning`): a 2D Gaussian in log SF × log TF times a two-lobed
von Mises direction tuning, plus an offset, fitted to single-trial responses; cross-validated leaving one recording
out. Preferred speed = TF0 / SF0. SF0 and TF0 may extrapolate one natural-log unit beyond the tested range and fits on
a bound are kept: on the cells with in-range preferences the figure correlations keep their sign and significance
(checked 2026-10-24), so the extrapolated fits are kept and the figure is unchanged.

**Selections.**
- Depth selective: `iscell` and one-sided Spearman p < 0.05 of the cross-validated depth tuning
  (`depth_tuning_test_spearmanr_*_closedloop`).
- SF/TF tuned: one-sided Spearman p < 0.05 between held-out responses and predictions of the trial-level fit. The
  stricter held-out R² above the empirical null (as RS/OF in figure 2) was checked as a robustness test
  (2026-10-24): all correlations keep their sign.
- RS/OF tuned: RS/OF held-out R² above the threshold of figure 2: 95th percentile of the mirrored negative tail of
  the iscell ROIs of all figure 2 sessions (`fig2/results_all_rsof_closedloop.parquet`, 0.0015 on 2026-10-24),
  not of the grating sessions alone (0.0009, which passed 648 instead of 501 grating cells).

**Direction selectivity** (panels E, F). The fitted `dsi` parameter only scales the null-direction lobe: when the
von Mises concentration κ goes to 0 the tuning is flat whatever `dsi` is, and the parameter drifts to its bounds. The
figure uses the selectivity implied by the fit, (pref − null) / (pref + null) = `dsi` · tanh κ / (2 − `dsi`).
Kim, Znamenskiy et al. (2018) used the same model but a model-free DSI and a width criterion (σ_dir < 40°); both
alternatives were compared (`agent_cache/sftf_dsi_comparison/`) and give the same conclusion. Panel F keeps the
cells with an implied DSI above `DSI_THR` (0.3: preferred response at least 1.86 times the null one; 0.5, a 3:1
ratio, until 2026-10-24). χ² p of the depth vs other comparison: 0.056 at 0.3, 0.034 at 1/3 (2:1), 0.084 at 0.5,
0.14 at 0.6.
The E test is stratified by session (`sftf.stratified_median_difference`): difference of the medians, depth
selective minus other, within each session, averaged over sessions weighted by their number of cells, with the depth
labels permuted within sessions. A pooled Mann-Whitney test was confounded: the session with the most depth-selective
cells (PZAH10.2f_S20230924) has the lowest DSI, so the pooled medians went the other way from 4 of the 6 sessions
(2026-10-24). On synthetic data with a session effect only, the stratified test gave 5.5% false positives at
p < 0.05 and the pooled test 100%.

**Direction convention.** Grating `Angle` is the wave vector; the bars drift towards `Angle` + 180° (stimulus
workflow `vis-stim-2d/run_SFTF.bonsai`, same in the 2023-09-11 version `fa05b40`, and BonVision). The grating is drawn
in an orthographic view (x 0–120°, y ±50°) wrapped on the sphere, with the axes of the RF reconstruction (x = azimuth
increasing to the right, y = elevation up). The figure plots drift direction (`sftf.drift_direction`): 0° is towards
higher azimuth, i.e. front to back, the optic flow of forward running; 90° is up. The thesis polar plots were labelled
in `Angle` units.

**Correlations** (panels G–J). Spearman ρ, 95% CI and p from a hierarchical bootstrap over sessions only (with 2
mice, resampling mice is degenerate: half the resamples contain one mouse). Five pairs are tested and Holm-corrected
(decided 2026-10-24): preferred grating speed vs preferred depth (G), the question; TF and SF vs depth (H, I), to see
which of the two drives it; then, as SF does, SF vs preferred optic flow (J) and running speed (not shown). The
other combinations (TF or grating speed vs optic flow or running speed) are not tested. Log-log regression line where
the Holm p < 0.05.

**Overlap of depth-selective and SF/TF-tuned cells** (panel D, added 2026-10-25): one square per neuron,
column by column by group, magenta (depth) and blue (SF/TF) of davidmathlogic.com/colorblind at alpha 0.5 on white
and their overlap for the cells that are both, count and percentage on each block. The count expected if independent is computed per
session and summed, because the fractions of tuned cells differ between sessions; Mantel-Haenszel test of the pooled
odds ratio.

**Running speed** (panels K–P). Bins of 2.8–5.7, 5.7–11.3, 11.3–22.6, 22.6–45.3 and > 45.3 cm/s: the treadmill bins
of figure 3 (motor speeds 4–61 cm/s, one octave apart), the last with no upper limit. One more curve (black) for all
the frames below 2.8 cm/s, whatever the running speed before or after (`prepare --slow`). Decided 2026-10-24; before,
the bins were 5–11.3, 11.3–22.6, 22.6–45.2 and 45.2–90.5 cm/s and the black curve was the still frames (below 5 cm/s
for about 1 s, as figure 1), which left out ~10% of the grating frames (slow but not still for 1 s). Curves over the
SF (TF) / direction pairs where the fitted tuning is at least half its best pair. Population curves
(`sftf.build_speed_curves`): cells significant on the frame-level `base` fit and with a significant TF (SF) effect;
each cell's peak comes from all the frames (running frames in the speed bins and slow frames, pooled) of half of the
recordings (alternate recordings, i.e. alternate presentations of each condition), and the curves of the other half,
per speed bin and for the slow frames, are aligned on that peak, so that no speed range defines the alignment (until
2026-10-24 the peak came from the running frames only). Responses are the dF/F divided by the maximum of the
trial-level SF/TF fit over the tested stimuli (offset + peak of `sftf.sftf_fit_peak`; decided 2026-10-25, no offset
subtracted; from 2026-10-24 to 2026-10-25 the fit offset was subtracted first and the peak above it was the
denominator). Normalising by the selection half's peak, with the kept pairs also chosen on that half (no fit at
all), was tried on 2026-10-25 and gave the same ordering of the speed bins, flanks at ~0.4 (baseline dF/F) and held-out
peaks below 1 (regression to the mean); the fit is the better estimate of the peak. Until 2026-10-24 they
were divided by the peak value of the selection half, which blew up for a cell whose peak was just above 0 (ΔF/F/peak
of about 1300 after the onset-skip refit); the fit is on all recordings, so the normalisation is not held out, but it
is never close to 0.

**Running-speed models** (exploratory). The amplitude depends on running speed through a Gaussian in log running
speed, parameterised and bounded as the RS/OF fits (`rsg`), optionally with a TF or SF preference shift. A
quadratic in log running speed (no bound on the peak) was tested on 2026-10-13 and gave the same solution wherever
`rsg` was not at its bound, the same preference shifts and no better held-out R², so `rsg` was kept and the quadratic
models removed.

## Exploratory analyses (`sftf_exploratory_analysis.ipynb` unless stated)

- Thesis summary figure (`fig_sftf_summary.pdf`): overlap of depth-selective and SF/TF-tuned cells, grating speed vs
  optic flow and depth, fitted `dsi`, preferred orientation and direction, with the thesis selections.
- Example neurons (`sftf_plots.plot_example_neuron`, layout of the thesis example figure) and near / mid / far
  depth-preferring examples.
- Population SF × TF maps by preferred depth (near < 20 cm, mid, far > 100 cm).
- SF × TF responses per running-speed bin, running-speed distribution, TF and SF curves by speed.
- Running-speed model comparison (`base`, `rsg`, `rsg_tf`, `rsg_sf`; held-out R², shift slopes, depth-selective vs
  other with within-session permutations).
- Running gain vs TF / SF preference shift (`shift_summary_overlay.{pdf,png}`, added 2026-10-25): slopes of the
  log-linear `gain`, `gain_tf`, `gain_sf` models (octave per octave) for the cells of panels M–P, depth selective
  over other, medians with a hierarchical bootstrap over sessions. Candidate summary panel for N and P, not in the
  figure. On 2026-10-25 `running_models/` was replaced by the full 7-model skip-8 run (identical `base` columns; the
  base-only copy is in `running_models_base_only_20261024/`).
- Proportions of responsive / orientation- / direction-selective cells and their preferred orientation and direction.
- TF × running-speed slope against preferred direction and against the shape of the RS/OF Gaussian.
- RF shape vs grating tuning: `fig_reviewer_rf_sftf.ipynb` with `figure_utils/rf_shape.py` (flank geometry and
  asymmetry of the spike RFs against preferred orientation and direction selectivity).
- Onset profile of the grating responses: `sftf_onset_profile.ipynb` (choice of the trial window).
- Frame-level lag sweep: `sftf_lag_sweep.py` (+ `.sbatch`), evaluated in `revisions/rf_spike_tau/lag_choice.ipynb`.

`PLAN_running_speed_models.md` is the 2026-10-10 plan of the 10-cell model selection; its script was removed on
2026-10-24.

## History

- 2026-10-07: analysis started. Sync files of the 30 SFTF recordings copied to BlackPasspo. cottage_analysis
  (`reviews`): `load_imaging_data` reads `dff.npy` or `dff_ast.npy` according to `ast_neuropil` (`ead762b`; it always
  read `dff_ast.npy`), `fit_sftf_tuning` passes the response as a numpy array (`325d16b`), cross-validation in
  `fit_sftf_tuning`. The February 2026 `neurons_df_sftf.pickle` was reproduced exactly from ASt dF/F, not from the
  `dff.npy` of the depth fits.
- 2026-10-08: `neurons_df_sftf.pickle` refitted on `dff.npy` with leave-one-recording-out cross-validation (NEMO job
  60117930; previous files in `/camp/home/blota/code/sftf_crossval_run/backups/`).
- 2026-10-09: the thesis figure criteria were verified on the thesis tables. The thesis grating fits are identical to
  the February 2026 (ASt) files. The lower depth-selective counts of the current data come from the traces (ASt vs
  non-ASt dF/F), not from pipeline changes.
- 2026-10-10: the figure recomputed on ASt dF/F and on deconvolved spikes (script in `9e642cb`, removed in the next
  commit): the conclusions held for all three traces.
- 2026-10-13: quadratic running-speed models tested and removed (see Definitions).
- 2026-10-14: frame-level analyses moved from a 5-frame lag to lag 0; lag-5 results deleted.
- 2026-10-15: trial-window bug fixed. The window was `[onset + 2, next_onset + 2)` in `gratings.generate_trials_df`
  and in the notebook (it included the start of the next grating); now `[onset + 8, next_onset)` everywhere.
  `neurons_df_sftf.pickle` refitted (job 60867152, run folder `/camp/home/blota/code/sftf_skip8_run`, previous files
  in `backups/`).
- 2026-10-19: PZAH10.2d_S20230920 FrameLog restored and the session refitted; its SFTF recordings were not affected.
- 2026-10-24: review of the figure and refactoring (decisions in REVIEW_FOLLOWUP.md): correlations bootstrapped
  over sessions with Holm correction, implied DSI, drift-direction convention, per-session overlap test, onset skip
  in the frame-level analyses, first running bin from 5 cm/s, static curve normalised like the running curves; code
  split between `figure_utils/sftf.py`, `precompute_data/sftf_grating_cells.py` and this folder. Removed analyses and
  their data: trial-average fits on speed subsets, gain vs preferred running speed notebook, 1D TF tuning by speed,
  10-cell model selection, running models on spikes and at lag 4.
