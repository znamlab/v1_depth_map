# Depth decoder: fitting time and choice of classifier (2026-11-17)

Why the decoder now uses `LinearSVC` (see [07](./07_decoder_rerun_2026-11-16.md)). All
numbers come from the libsvm rerun without rolling average. Full tables, run directories and
scripts are in
[`v1_depth_map/revisions/decoder_rolling/README.md`](../v1_depth_map/revisions/decoder_rolling/README.md),
section "Fitting time and the C grid".

## 1. Why libsvm was slow

- Each fold job fit a linear `SVC` (libsvm, single-threaded) for 7 values of C on the
  training set, picked the best C on the validation set, and refit on train+val.
- The C=1000 fit dominated: 1.7-2.2 h on slow folds, against about 30 min for the other six
  Cs together. The slowest sessions had fold jobs of 15-24 h.
- Two things made a session slow:
  - **Many frames:** for example, PZAH8.2f_S20230313 has 19,540 training frames, against
    3,000-5,000 in fast sessions.
  - **A few `iscell` ROIs with extreme dF/F** (SD up to 821, against 0.11 for the median
    neuron). They make the SVM badly conditioned, and libsvm converges very slowly at high C.
    These look like broken dF/F (F0 close to zero); see
    [09](./09_outlier_dff_rois_2026-11-17.md).
- Running the C grid in parallel (cottage_analysis `1a3287b`) saves about 20 % with identical
  results. The C=1000 fit sets the floor.

## 2. Validation accuracy across C (libsvm)

Over 555 folds, best C was 0.1 in 242 folds and 1000 in 20. At high C the curve is a flat
plateau where neighbouring Cs differ by 0.001-0.005, and only 252/555 curves have a single
peak. Skipping C=1000 when C=100 does not beat C=10 would skip 400 fits but change the best C
in 5 folds, so it was not used.

## 3. Alternative classifiers

Tested on 10 sessions (the 5 slowest + 5 spread), both loops, all folds, same selection rule.
Live = libsvm run (100 fold jobs, 348 CPU-hours, slowest fold 18.1 h).

| Variant | Accuracy − live, mean [range] | Total time | Slowest fold | Best C at grid bottom | Unconverged fits | Closed − open, corr. with live |
| --- | --- | --- | --- | --- | --- | --- |
| A: z-score + `SVC` (cancelled) | −0.027 [−0.063, −0.001] | slower on slow sessions | > 3 h per C | 12/40 | 0 | - |
| C: one-vs-one `LinearSVC(loss="hinge")` | +0.004 [−0.012, +0.016] | 6.4 h | 58 min | 3/100 | 9359 | 0.97 |
| D: `LinearSVC()` defaults | +0.003 [−0.025, +0.032] | 3.1 h | 14 min | 25/100 | 13 | 0.96 |
| E: z-score + D | −0.004 [−0.062, +0.035] | 4.4 h | 36 min | 35/100 | 76 | 0.95 |
| F: C with `max_iter=10000` | +0.004 [−0.010, +0.017] | 30.2 h | 156 min | 2/100 | 5257 | 0.97 |
| **G: D with C from 1e-5 to 1e3** | **−0.0005 [−0.025, +0.032]** | **2.0 h** | **20 min** | 5/100 at 1e-5 | 13 | 0.96 |

- **Chosen: G**, the closest to live on average, with the smallest confusion-matrix
  difference and about 170x less CPU time.
- **C and F** match libsvm best, but liblinear's hinge solver does not converge on unscaled
  data, even with 10x more iterations.
- **A and E:** z-scoring makes `SVC` slower and pushes the best C down.
- **Confusion matrices:** G shifts recall slightly towards the extreme depths (5 and 640 cm),
  within about one SEM.
- **Agreement of single predictions with live** is only 0.55-0.9, even at equal accuracy.
  Compare accuracy and confusion matrices instead.
