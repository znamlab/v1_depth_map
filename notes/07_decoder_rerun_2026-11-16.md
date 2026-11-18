# Depth decoder rerun without rolling average (2026-11-16 to 11-18)

All 84 Figure 1 sessions were refit. The results are copied back to BlackPasspo, and the
figure caches and stats were rebuilt on 2026-11-18.

## 1. What changed

| | Before (Oct 31) | Now (Nov 18) |
| --- | --- | --- |
| Preprocessing | 0.5 s rolling mean of dF/F and RS, then 0.5 s downsampling | 0.5 s downsampling only (`rolling_window=None`, cottage_analysis `9165c80`) |
| Classifier | `SVC(kernel="linear")` (libsvm, one-vs-one, hinge) | `LinearSVC()` defaults (liblinear, one-vs-rest, squared hinge), cottage_analysis `516df73` |
| C grid | `logspace(-3, 3, 7)` | `logspace(-5, 3, 9)` |
| Still bin, speed bins | max raw RS over 1 s < 5 cm/s; 0.05-2 m/s | unchanged |

- **Why LinearSVC:** the libsvm rerun took up to 24 h per fold, and some sessions never
  finished. `LinearSVC` gives the same accuracy on average and uses about 170x less CPU time
  (variant G in [08](./08_decoder_fitting_speed_2026-11-17.md)).
- **Methods text:** must describe the new model (one-vs-rest, squared hinge, C from 1e-5
  to 1e3).

## 2. Bugs fixed on the way

The first batch failed with `pickle data was truncated` on `decoder_results.pickle`, because
several jobs of a session read and wrote that file at the same time. The orchestrator could
also silently overwrite results written meanwhile by another job. Fixed in cottage_analysis
(branch `reviews`):

- `c3756bb`: the fold jobs no longer read `decoder_results.pickle`, and closed-loop
  `calculate_acc_conmat` waits for all open-loop jobs.
- `ad8bb41`: `decoder_results.pickle` is loaded just before writing and written atomically
  (temp file + `os.replace`).
- `1a3287b`: each fold job fits the C grid in parallel (about 20 % faster, identical results).

If a session ever needs fixing, the per-fold outputs (`decoder_outputs_<type>_fold<i>.pickle`)
are valid on their own: rerun only `calculate_acc_conmat` → `find_acc_speed_bins` (open loop,
then closed loop) → `plot_decoder_session`.

## 3. Checks

`check_linearsvc.py` (NEMO backup folder below) passed for all 84 sessions (120 session ×
loop), with fresh fold outputs and results, 5 best Cs per loop, accuracy equal to
`accuracy_score(y_test, y_preds)`, and speed-bin entries present.

- **Best C over 600 folds:** 1e-5: 21, 1e-4: 25, 1e-3: 96, 0.01: 233, 0.1: 99, 1: 41,
  10: 47, 100: 24, 1000: 14.
- **LinearSVC − libsvm accuracy (no rolling, sessions finished in both):** closed loop
  −0.004 mean [−0.030, +0.032] (n = 76), open loop +0.002 [−0.035, +0.031] (n = 33).
- **Effect of removing the rolling average (libsvm, both runs):** see
  `v1_depth_map/revisions/decoder_rolling/README.md`. The median accuracy change is −0.003
  (closed loop) and −0.001 (open loop).
- **Figure 2 open vs closed loop decoding error:** 0.35 → 0.31, p 0.0004 → 0.0021
  (stats commit `515e6ce`).

## 4. Where the data are

- **Current results:** `decoder_results.pickle` and `decoder_outputs_*_fold*.pickle` in each
  session folder, on NEMO and BlackPasspo.
- **Rolling results (Oct 31):** `decoder_results.pickle.rolling_20261031_backup` next to each
  file on BlackPasspo. These are the only copies, since the NEMO ones were overwritten.
- **libsvm results without rolling:** NEMO
  `/nemo/lab/znamenskiyp/home/users/blota/decoder_libsvm_norolling_backup_2026-11-17/`
  (fold outputs and `decoder_results_slim.pickle`, without `trials_df_*`). 8 sessions there
  are incomplete: PZAH10.2d_S20230704, PZAH10.2d_S20230821, PZAH10.2f_S20230707,
  PZAH8.2f_S20230214, PZAH8.2h_S20230113, PZAH8.2h_S20230127, PZAH8.2i_S20230110 and
  PZAH8.2i_S20230330.
- **Figure 1 cache:** `ver_rev1/decoder_df_all.pickle`, rebuilt 2026-11-18. The Jul 13 version
  is kept as `decoder_df_all.pickle.bak`.

## 5. Still to do

- [ ] Update the methods text for `LinearSVC` (§1).
- [ ] Update the decoder section of `notes/02_processing_fits.md`.
- [ ] Decide on the ROIs with extreme dF/F ([09](./09_outlier_dff_rois_2026-11-17.md)), which
      affect every decoder variant.
