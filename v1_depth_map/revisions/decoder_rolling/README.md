# Depth decoder: with vs without rolling average (Nov 2026)

How much the depth decoder accuracy changed when the 0.5 s rolling average before
downsampling was removed, per session and for the population. Both runs used the libsvm
`SVC`. Production has since moved to `LinearSVC` (`notes/07_decoder_rerun_2026-11-16.md`), so
this comparison isolates the effect of the rolling average only.

## What changed between the two runs

| | Rolling (Oct 31) | No rolling (Nov 16-17) |
|---|---|---|
| Date of `decoder_results.pickle` | Oct 31 2026 (Still-bin fix, see `notes/02`) | Nov 16-17 2026 |
| Preprocessing | 0.5 s rolling mean of dF/F and RS, then 0.5 s downsampling | 0.5 s downsampling only (`rolling_window=None`, cottage_analysis `9165c80`) |
| C grid, Still rule, speed bins | `logspace(-3, 3, 7)`, max raw RS over 1 s < 5 cm/s, 0.05-2 m/s | unchanged |

The rolling average is the only analysis difference. The FrameIndex sync fix does not matter
for these sessions: the hey2 sessions have a 0-1 frame offset.

**The stored `params` is wrong in the rerun files.** `decoder_results.pickle` from the rerun
still has `params["rolling_window"] == 0.5`. That key is left over from the Oct 31 file: the
current pipeline never writes `params`, and the orchestrator updates the existing dict. The
rerun really ran without rolling: NEMO `~/code/cottage_analysis` pulled `9357cf8` (which
includes `9165c80`) at 17:50 on Nov 16, and the batch was submitted at 17:52. The sbatch
`run_depth_decoder_pipeline.sh` passes no `rolling_window`, and the default is `None`. Do not
use `params` to tell the two runs apart.

## Where the data are

- **Rolling results:** `decoder_results.pickle.rolling_20261031_backup` in each session folder
  on BlackPasspo (`/Volumes/BlackPasspo/v1_depth_map/processed/hey2_3d-vision_foodres_20220101/<mouse>/<session>/`).
  These are the only full copies. The NEMO `.pre_still_fix_20261031_backup` files have rolling
  but the old Still rule, so their Still bin is not comparable.
- **Closed-loop rolling values** are also in the Jul 13 Figure 1 cache
  `ver_rev1/decoder_df_all.pickle.bak`. Open loop is not in the cache.
- **No-rolling libsvm results:** NEMO
  `/nemo/lab/znamenskiyp/home/users/blota/decoder_libsvm_norolling_backup_2026-11-17/`. The
  live session files now hold the `LinearSVC` results.
- **Summaries:** `old_rolling_summary.pickle`, `new_norolling_summary.pickle` and
  `unfinished_sessions_at_submit.txt` in `/Volumes/BlackPasspo/agent_cache/decoder_rolling_comparison/`.

## Files in this folder

- `sessions.txt`: the 84 sessions of Figure 1, copied once from the `session` column of the
  Oct 31 Figure 1 cache `decoder_df_all.pickle` (BlackPasspo). It is a fixed list so that the
  NEMO job, which cannot read that cache, uses exactly the sessions of the compared run.
- `extract_decoder_summary.py`: reads `decoder_results.pickle` for each session. It does not
  write to the decoder files.
  - **Kept per loop:** accuracy, confusion matrix, best C, accuracy and confusion matrix per
    speed bin, and the number of test samples. Small output; the input files are ~1 GB each.
  - **Checks recorded (from notes/07):**
    - the stored accuracy equals `accuracy_score(y_test, y_preds)`;
    - there are 5 best-C values;
    - the speed-bin results are present.
- `extract_decoder_summary.sbatch`: the same extraction, run on NEMO.
- `compare_decoder_rolling.py`: pairs the old and new results, then writes one PNG per
  session, a population summary, and CSV/pickle tables. A session is compared only if all of
  these hold:
  - it is not listed in the unfinished file;
  - its new file is newer than 2026-11-16 17:52;
  - both loops pass the checks.

## Results (2026-11-17)

Compared for the 76 sessions whose no-rolling libsvm run had finished. The other 8
(PZAH10.2d_S20230704, PZAH10.2d_S20230821, PZAH10.2f_S20230707, PZAH8.2f_S20230214,
PZAH8.2h_S20230113, PZAH8.2h_S20230127, PZAH8.2i_S20230110, PZAH8.2i_S20230330) never
finished before the switch to `LinearSVC` and were not compared. Output in
`/Volumes/BlackPasspo/v1_depth_map/processed/v1_manuscript_figures/ver_rev1/decoder_rolling_comparison/`:
`sessions/*.png`, `population_summary.png`, `comparison.csv`, `status.csv`.

- **Closed loop (n = 76):** median accuracy 0.346 -> 0.341, median difference -0.003
  (IQR -0.008 to +0.006), range -0.025 to +0.019. Wilcoxon p = 0.08. Best C is identical in
  72/76 sessions.
- **Open loop (n = 33):** median accuracy 0.226 -> 0.220, median difference -0.001 (IQR
  -0.009 to +0.001), range -0.041 to +0.009. Wilcoxon p = 0.046. Best C is identical in all
  sessions.
- Largest drop: PZAH10.2d_S20230613, open loop, 0.266 -> 0.225. Its depth-5 row loses
  accuracy.
- Still bin: median difference -0.003 (closed loop) and 0.000 (open loop). The other bins
  change by less than 0.01 wherever at least 12 sessions contribute.
- Speed bins with fewer than 20 test samples in either run are left out. The speed bins use
  the downsampled running speed, which the rolling average also changed, so bin membership
  shifts slightly. The Still bin uses raw speed, so its membership does not change.
- The old closed-loop accuracy equals the Figure 1 cache exactly for all 84 sessions. Four old
  PZAG3.4f closed-loop files (S20220426, S20220503, S20220505, S20220511) list 10 best-C
  values, apparently appended twice by an old run; their accuracy checks out.

## Fitting time and the C grid (rerun, no rolling)

All numbers below come from the same rerun (`decoder_inputs_<type>.pickle` built without
rolling). Details of the batch itself are in `notes/07_decoder_rerun_2026-11-16.md`.

### 1. Where the time goes

- Each fold job fits a linear `SVC` (libsvm, single-threaded) for the 7 values of C on the
  training set, picks the best on the validation set, and refits on train+val.
- On slow folds the C=1000 fit alone takes 1.7-2.2 h. The other six Cs take about 30 min in
  total, and the final refit at the usual best C (0.1) takes about 1 min.
- The slowest sessions have fold jobs of 15-19 h. PZAH10.2d_S20230821 was still at C=10 after
  17 h (both loops).
- The libsvm kernel cache (`cache_size` 200, 2000 or 4000 MB) changes neither the speed nor the
  results.
- Two things make a session slow (NEMO run dir `session_sizes.py`, `outlier_neurons.py`):
  - **Many frames.** For example, PZAH8.2f_S20230313 has 19,540 training frames, against 3,000-5,000
    in fast sessions.
  - **A few neurons with extreme dF/F.** These sessions are small, but one or a few
    `iscell` ROIs have dF/F values hundreds or thousands of times larger than the other neurons.
    That makes the SVM problem badly conditioned at high C:
    - PZAH10.2d_S20230821: ROI 55 has SD 821 and max 5334 (the median neuron has SD 0.11 and
      max 0.9); ROIs 561 and 41 have SD of about 10. This is the slowest session.
    - PZAH8.2i_S20230110: ROI 14 has SD 54 and max 373.
    - PZAH10.2f_S20230707: ROI 312 has SD 81 and max 577, plus 7 more ROIs above 20.
    - PZAH8.2i_S20230330: 8-10 ROIs have max 37-71.
    - PZAH8.2h_S20230113: ROI 744 has SD 7.6 (the median is 0.14).

    These look like broken dF/F traces (F0 close to zero), not real activity. They also weigh on
    the decoder itself, not only its speed. Not yet checked against the suite2p traces.

### 2. Speed-up that keeps the results identical

`cottage_analysis` `1a3287b` (branch `reviews`): `fit_each_fold` fits the C grid in parallel (joblib, `SLURM_CPUS_PER_TASK` workers, 8 by
default). The final train+val fits for every C except 1000 start alongside the grid, so the
final model is usually ready when the grid ends. Each fit's duration is logged.

Benchmark (NEMO `/flask/scratch/znamenskiyp/blota/svm_bench_2026-11-17`), two slow closed-loop
folds of the rerun:

| Fold | Current code | Parallel | Identical to live outputs |
| --- | --- | --- | --- |
| PZAH8.2f_S20230126 fold 1 | 8116-8276 s | 6182-6888 s | yes |
| PZAH8.2h_S20230302 fold 4 | 9460-9571 s | 7761-7851 s | yes |

- Best C, validation accuracy and predictions at every C, test predictions and decision values
  are bit-identical across all variants and equal the live `decoder_outputs_*_fold*.pickle`.
- The gain is about 20 %: the C=1000 fit cannot be split, so it sets the minimum fold time.
  Running the Cs as separate Slurm jobs would not shorten a fold either. It would only reduce
  idle CPUs, at the cost of seven times more jobs.

### 3. Validation accuracy across C

From the logs of the 555 fold jobs of this batch that tried all 7 Cs (NEMO
`~/c_curves_2026-11-17.json`, one `[session, job name, {C: accuracy}]` entry per fold):

- Best C: 0.001 in 22 folds, 0.01 in 112, 0.1 in 242, 1 in 88, 10 in 40, 100 in 31 and
  1000 in 20.
- Only 252/555 curves have a single peak. At high C the curve is a flat plateau where
  neighbouring Cs differ by about 0.001-0.005.
- Rule "fit C=1000 only if C=100 beats C=10": skips 400 of the 555 C=1000 fits and changes the
  best C in 5 folds, where C=1000 had won by 0.001-0.006:
  PZAH8.2f_S20230214 openloop fold 2, PZAH10.2d_S20230725 closedloop fold 1,
  PZAH8.2f_S20230223 openloop fold 4, PZAH10.2d_S20230623 openloop fold 4,
  PZAH10.2f_S20230606 openloop fold 3.
- Stricter rule "fit C=1000 only if C=100 is the best so far": skips 511 fits, changes 7.
- Not applied.

### 4. Alternative fitting methods

NEMO array `64158655`, run dir `/flask/scratch/znamenskiyp/blota/svm_variants_2026-11-17`
(`variant_fit.py`, `variant_fit.sbatch`, `manifest.txt`, `compare_variants.py`). Read-only on
the decoder files; writes only to the run dir.

- **Sessions:** 10 sessions with both loops finished: the 5 with the slowest folds
  (PZAH8.2f_S20230313, PZAH8.2i_S20230203, PZAH8.2i_S20230209, PZAH8.2f_S20230223,
  PZAH8.2h_S20230224) and 5 spread over the rest (PZAH8.2h_S20230303, PZAH10.2f_S20230606,
  PZAH10.2d_S20230920, PZAH10.2f_S20230615, PZAH10.2f_S20230623). All 5 folds of both loops.
- **Variants:** same C grid and selection rule as production; 1 CPU each:
  - A: z-score each neuron (fitted on the training data) + current `SVC(kernel="linear")`;
  - C: `OneVsOneClassifier(LinearSVC(loss="hinge"))`, the liblinear version closest to the
    current model;
  - D: `LinearSVC()` defaults (squared hinge, one-vs-rest);
  - E: z-score + D.
  - The skip rule combined with z-scoring is not run separately; it can be read from A's
    per-C accuracies.
- **Compared with the live run:** time per fold, best C, session-level test accuracy (folds
  combined as in `calculate_acc_conmat`), agreement of predictions, confusion matrices, the
  closed - open loop difference across sessions, convergence warnings, and how often the best C
  is at the bottom of the grid (0.001).
- **Smoke test** (PZAH10.2f_S20230623 closed loop, fold 0): every variant ran the full grid
  plus refit in 16-28 s. Best C was 0.001 for E and 1 for C. C gives convergence warnings at
  C >= 1 with liblinear's default `max_iter`.
- **Results, 2026-11-17 17:00.** C, D and E were complete (20 session x loop); A was partial
  (8, only fast sessions). Output in the run dir: `comparison.pickle`. Live CPU time is the
  sum of the 100 live fold jobs.

  | Variant | Accuracy - live, mean [range] | Predictions agreeing with live | Total time vs live (347.7 h) | Slowest fold (live 18.1 h) | Folds with best C = 0.001 | Convergence warnings | Closed - open, corr. with live |
  | --- | --- | --- | --- | --- | --- | --- | --- |
  | A (n=8) | -0.027 [-0.063, -0.001] | 0.61 | 0.9 h (fast sessions only) | 9.7 min | 12/40 | 0 | - |
  | C | +0.004 [-0.012, +0.016] | 0.75 | 6.4 h | 58 min | 3/100 | 9359 | 0.97 |
  | D | +0.003 [-0.025, +0.032] | 0.64 | 3.1 h | 14 min | 25/100 | 13 | 0.96 |
  | E | -0.004 [-0.062, +0.035] | 0.57 | 4.4 h | 36 min | 35/100 | 76 | 0.95 |

  The live run has best C = 0.001 in 6/100 folds.

  - **A (z-score + SVC)** is not a speed-up: on the slow sessions its C=10 and C=100 fits
    already take 2-3 h each. It also loses accuracy, and its best C often sits at the bottom
    of the grid.
  - **C (one-vs-one liblinear, hinge)** is the closest to the current model. It is about 55x
    less CPU time, the best-C distribution is similar to live, and the closed - open loop
    difference is preserved (mean 0.054 vs 0.053 live). But most of its fits at C >= 1 stop at
    liblinear's default 1000 iterations without converging.
  - **D (`LinearSVC` defaults)** is the fastest, about 110x, and barely warns. It matches the
    accuracy and the closed - open difference about as well as C. But 25 % of its folds pick
    the lowest C, so the grid would need to go lower.
  - **E** brings no benefit over D. Z-scoring pushes the best C even lower.
  - **Agreement with live** is 0.6-0.9 even when the accuracy is the same. At 15-40 % accuracy,
    individual predictions are noisy. We have no baseline for how much they would move under a
    negligible change, so read the accuracy and confusion-matrix columns rather than agreement.
- **A cancelled** (17:05, 55 tasks still running; 45/100 folds done): too slow and less
  accurate, so finishing it would not change the conclusion.
- **Follow-up, array `64180470`** (`manifest_fg.txt`, `variant_fit_fg.sbatch`), same sessions
  and folds:
  - F: C with `max_iter=10000`, to test whether it converges and still matches the live run;
  - G: D with the C grid extended down to 1e-5 (`logspace(-5, 3, 9)`), to test whether the
    best C stops sitting at the bottom of the grid.
- **Follow-up results (2026-11-17 ~20:00, all 200 tasks completed):**

  | Variant | Accuracy - live, mean [range] | Total time | Slowest fold | Best C at grid bottom | Unconverged fits | Closed - open, corr. with live |
  | --- | --- | --- | --- | --- | --- | --- |
  | F (C, `max_iter=10000`) | +0.004 [-0.010, +0.017] | 30.2 h | 156 min | 2/100 at 0.001 | 5257 | 0.97 |
  | G (D, C from 1e-5) | -0.0005 [-0.025, +0.032] | 2.0 h | 20 min | 5/100 at 1e-5 (9 below 0.001) | 13 | 0.96 |

  - **F:** ten times more iterations still leaves about half of its fits unconverged. It is
    5x slower than C, and the results are the same as C. liblinear's hinge-loss solver does not
    converge on this unscaled data, but the lack of convergence does not change the accuracy.
  - **G:** extending the grid down fixes most of D's bottom-of-grid problem. The mean accuracy
    difference from live drops to -0.0005, and it has the smallest confusion-matrix difference
    of all variants (mean absolute difference 0.021). It uses about 170x less CPU time than the
    live run, and its slowest fold takes 20 min, against 18 h live. Per-session differences
    still range from -0.025 to +0.032.
- **Confusion matrices, live vs C (one-vs-one liblinear) vs G (one-vs-rest).**
  `conmat_diagnostics.py`, plots in
  `/Volumes/BlackPasspo/v1_depth_map/processed/v1_manuscript_figures/ver_rev1/decoder_rolling_comparison/svm_variants_conmat/`.
  Mean over the 10 sessions.
  - **Overall:** the distribution of error distances (|decoded - true| in depth steps) is the
    same for all three. Fraction exact is 0.207 live, 0.212 C and 0.209 G in closed loop, and
    0.177 / 0.181 / 0.178 in open loop.
  - **Live already has a U-shape:** recall is highest at 5 and 640 cm, and those two depths are
    also decoded most often.
  - **G (one-vs-rest)** shifts this slightly towards the extremes:
    - closed loop: recall +0.017 at 5 cm and +0.028 at 640 cm, -0.029 at 10 cm, -0.012 at
      40 cm;
    - closed loop: more frames decoded as 5 cm (0.209 vs 0.194), fewer as 10 cm (0.129 vs
      0.160);
    - open loop: +0.029 at 640 cm, -0.015 at 40 and 80 cm.
  - **C (one-vs-one)** does the opposite: slightly better at the middle depths (+0.01 to +0.02)
    and worse at 5 cm (-0.02).
  - **Size:** these shifts are within about one SEM. Per-session differences span about
    ±0.1 per depth.
- **Adopted:** G, `LinearSVC()` with default settings (squared hinge loss, one-vs-rest) and C
  from 1e-5 to 1e3, on raw dF/F, is the production decoder since 2026-11-18 (notes/07).
  - The methods text would need updating: the model becomes squared hinge with one-vs-rest
    instead of `SVC` one-vs-one.
  - Fix or exclude the outlier ROIs (`notes/09_outlier_dff_rois_2026-11-17.md`) first, because
    they affect every variant.


## Reproducing the comparison

```bash
O=/Volumes/BlackPasspo/agent_cache/decoder_rolling_comparison
.venv/bin/python -W ignore v1_depth_map/revisions/decoder_rolling/compare_decoder_rolling.py \
  --old $O/old_rolling_summary.pickle --new $O/new_norolling_summary.pickle \
  --unfinished $O/unfinished_sessions_at_submit.txt \
  --out-dir /Volumes/BlackPasspo/v1_depth_map/processed/v1_manuscript_figures/ver_rev1/decoder_rolling_comparison
```

- `sessions/<session>.png`: one figure per session, one row per loop: accuracy per speed bin
  (old in grey, new in red, overall accuracy dashed, chance dotted), both row-normalised
  confusion matrices with best C, and their difference.
- `population_summary.png`: old vs new scatter, and per-bin differences.
- `comparison.csv` / `.pickle` and `status.csv` (why each session was or was not compared).

`new_norolling_summary.pickle` was extracted on NEMO from the live files before they were
overwritten by the `LinearSVC` run, so rerunning `extract_decoder_summary.sbatch` now would
read the `LinearSVC` results instead.
