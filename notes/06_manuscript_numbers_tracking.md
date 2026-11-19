# 06 - Manuscript Numbers Tracking & Audit

This document lists every pipeline-tracked number cited in the manuscript (`v1_depth_map.tex`, Overleaf/Dropbox copy) next to its value in the stats pipeline (`stats/generated/*.yaml`, compiled into `MANUSCRIPT_STATS.md` / `manuscript_stats.tex`). How the pipeline works is described in [05_manuscript_statistics.md](./05_manuscript_statistics.md).

> [!IMPORTANT]
> **Status (2026-11-19)**
> - **Stats column**: matches the YAMLs regenerated on 2026-11-18, plus 28 Table S1 / Animals macros added to `methods_dataset_tables` on 2026-11-19 (281 metrics, 15 notebooks, 0 hand-entered, `build_manuscript_stats --check` passes).
> - **Paper column and line numbers**: from a full read of the `.tex` on 2026-11-19 (Overleaf copy saved 2026-11-18 18:28, 1,049 lines).
> - 🔴 **2 lines of the paper need updating** (Section 1): the Fig 3F caption $N$ and the decoder smoothing sentence in the Methods.
> - The 14 lines flagged on 2026-11-18 (Fig 2K, Fig 3, Fig S7, Fig S10J, Table S1) are all fixed in the `.tex`.
> - All numbers in the `.tex` are literals (no `\stat...` macros in use), so drift is only caught by audits like this one.

---

## 1. Paper needs updating

| Line(s) | Context | Value in paper | Current stats / code | Cause |
| :---: | :--- | :--- | :--- | :--- |
| L227 | Fig 3F caption, neurons | $N = 214$ | $N$ = 227 (`\statFigThreeMotorizedDepthNeurons`) | 214 is the elongated subset of Fig 3M. Fig 3F and G use all 227 neurons (L211 gives 100/227 and 88/227, and the 3G caption says "same neurons as in F") |
| L782 | Methods, population decoding | "smoothed with a 0.5 s moving average and downsampled by averaging in non-overlapping 0.5 s bins" | 0.5 s downsampling only, no moving average (`rolling_window=None`) | Decoder rerun without rolling average (2026-11-18, [07_decoder_rerun_2026-11-16.md](./07_decoder_rerun_2026-11-16.md)); the Fig 1L-M and Fig 2K-L numbers in the paper come from that rerun |

To consider (wording, not a number mismatch): L240 says "the majority of neurons were best explained by conjunctive fits oriented at $\sim 45^\circ$" next to "48.1% of the neurons in the $[22.5^\circ,67.5^\circ)$ interval". The 45° group is the largest but below 50% with this binning (the von Mises component at 43° has a 70% weight, L234).

---

## 2. Audit tables

Only numbers tracked by the stats pipeline are listed. Static parameters (setup geometry, stimulus parameters, fit bounds) are not tracked here; the ones checked against code are in Section 3.

### 2.1 Figure 1 (Depth selectivity)

| Line | Context | Value in Paper | Macro | Stats | Status |
| :---: | :--- | :---: | :--- | :---: | :---: |
| L148 | Fig 1J depth-selective neurons | $N$ = 27,127 | `\statFigOneDepthNeurons` | 27,127 | 🟢 |
| L150 | Fig 1L sessions | $N$ = 59 | `\statFigOneEightDepthSessions` | 59 | 🟢 |
| L151 | Fig 1M sessions | $N$ = 84 | `\statFigOneTotalSessions` | 84 | 🟢 |
| L152 | Fig 1N mice / sessions | 4 mice, 19 sessions | `\statFigOneMultidayMice` / `Sessions` | 4 / 19 | 🟢 |
| L154 | Fig 1P pairs / neurons | 374 pairs, 234 neurons | `\statFigOneMultidayPairs` / `TrackedNeurons` | 374 / 234 | 🟢 |
| L154, L169 | Fig 1P correlation | $r = 0.819$, $p < 0.0001$ | `\statFigOneMultidayPrefDepthR` / `Pval` | 0.819 / $p < 0.0001$ | 🟢 |
| L159 | 5-depth / 8-depth sessions | 25 sessions; 59 sessions | `\statFigOneFiveDepthSessions` / `EightDepthSessions` | 25 / 59 | 🟢 |
| L161 | Neurons / sessions / mice | 59,937 / 84 / 7 | `\statFigOneTotalNeurons` / `TotalSessions` / `TotalMice` | 59,937 / 84 / 7 | 🟢 |
| L161, L273 | Depth-selective | 45.3% (27,127 of 59,937) | `\statFigOnePctDepthNeurons` / `DepthNeurons` | 45.3% / 27,127 | 🟢 |

### 2.2 Figure 2 (RS/OF integration, open loop)

| Line | Context | Value in Paper | Macro | Stats | Status |
| :---: | :--- | :---: | :--- | :---: | :---: |
| L190 | Fig 2E sessions | $N$ = 84 | `\statFigTwoModelSessions` | 84 | 🟢 |
| L193, L194 | Fig 2H-J neurons | $N$ = 3,961 | `\statFigTwoOpenLoopNeurons` | 3,961 | 🟢 |
| L195 | Fig 2K sessions | $N$ = 34 | `\statFigTwoDecoderTotalSessions` | 34 | 🟢 |
| L196 | Fig 2L sessions | $N$ = 27 | `\statFigTwoDecoderEightDepthSessions` | 27 | 🟢 |
| L200 | Additive & conjunctive vs others | all $p < 0.0001$ | `\statFigTwoModelComparisonPval` | $p < 0.0001$ | 🟢 |
| L205 | Peak amplitude ratio | median 0.98, $p = 0.735$ | `\statFigTwoAmpRatioMedian` / `Pval` | 0.98 / 0.735 | 🟢 |
| L205 | Preferred RS closed vs open | $r = 0.470$, $p < 0.0001$ | `\statFigTwoRSCorrR` / `Pval` | 0.470 / $p < 0.0001$ | 🟢 |
| L205 | Preferred OF closed vs open | $r = 0.536$, $p < 0.0001$ | `\statFigTwoOFCorrR` / `Pval` | 0.536 / $p < 0.0001$ | 🟢 |
| L207 | Fig 2K open - closed decoding error | median 0.31 octaves, $p = 0.0021$ | `\statFigTwoDecoderErrDiffMedian` / `Pval` | 0.31 / 0.0021 | 🟢 |
| L647 | Model comparison neurons | 21,457 of 27,127 (84 sessions, 7 mice) | `\statFigTwoModelNeurons` / `\statFigOneDepthNeurons` | 21,457 / 27,127 | 🟢 |
| L653 | Open-loop neurons | 3,961 of 9,462 (34 sessions, 7 mice) | `\statFigTwoOpenLoopNeurons` / `TotalDepthNeurons` / `Sessions` / `Mice` | 3,961 / 9,462 / 34 / 7 | 🟢 |

### 2.3 Figure 3 (Motorized wheel) and Fig S7 (simulation)

| Line | Context | Value in Paper | Macro | Stats | Status |
| :---: | :--- | :---: | :--- | :---: | :---: |
| L211, L227 | Fig 3F correlation | $r = 0.778$, $p < 0.0001$ | `\statFigThreeMotorizedDepthR` / `Pval` | 0.778 / $p < 0.0001$ | 🟢 |
| L227 | Fig 3F neurons | $N$ = 214 | `\statFigThreeMotorizedDepthNeurons` | 227 | 🔴 |
| L211 | Fig 3G fastest RS bins | 44%, 100/227 | `\statFigThreePrefRSAboveRangePct` / `N` | 44% (100/227) | 🟢 |
| L211 | Fig 3G peaked at intermediate speeds | 39%, 88/227 | `\statFigThreePrefRSWithinRangePct` / `N` | 39% (88/227) | 🟢 |
| L232, L661 | Fig 3K neurons / sessions / mice | 227 / 4 / 4 | `\statFigThreePolarNeurons` / `Sessions` / `Mice` | 227 / 4 / 4 | 🟢 |
| L232, L661 | Elongation $> 1.4$ | 214/227 | `\statFigThreeElongationCutoffRatio` | 214/227 | 🟢 |
| L234 | Fig 3M neurons, $k$ | $N$ = 214, $k = 3$ | `\statFigThreeElongatedNeurons` / `VonMisesK` | 214 / 3 | 🟢 |
| L234 | Fig 3M components | 18% at 4°, 70% at 43°, 12% at 90° | `\statFigThreeVonMisesComponents` | 18% at 4°, 70% at 43°, 12% at 90° | 🟢 |
| L240 | Orientation near 0°, $[-22.5^\circ, 22.5^\circ)$ | 67/214, 31.3% (4 sessions, 4 mice) | `\statFigThreeOrientationZeroN` / `Pct` | 67/214, 31% (raw 31.31) | 🟢 |
| L240 | Orientation near 90°, $[67.5^\circ, 112.5^\circ)$ | 18.2% | `\statFigThreeOrientationNinetyPct` | 18% (raw 18.22) | 🟢 |
| L240 | Orientation near 45°, $[22.5^\circ, 67.5^\circ)$ | 48.1% | `\statFigThreeOrientationFortyFivePct` | 48% (raw 48.13) | 🟢 |
| L663 | Real data elongated | 94% (214/227) | `\statFigThreeElongationCutoffPct` / `Ratio` | 94% (214/227) | 🟢 |
| L925 | S7C neurons | $N$ = 326 | `\statSuppSimulFreeNeurons` | 326 | 🟢 |
| L663, L926 | Simulation, free running | 306/326 (94%; 93.9% in caption) | `\statSuppSimulFreeElongatedRatio` / `Pct` | 306/326 (94%, raw 93.87) | 🟢 |
| L663, L926 | Simulation, motorized | 22/227 (9.7%) | `\statSuppSimulMotorElongatedRatio` / `Pct` | 22/227 (10%, raw 9.69) | 🟢 |
| L926 | S7D neurons / sessions / mice | 227 / 4 / 4 | `\statSuppSimulMotorNeurons` / `Sessions` / `Mice` | 227 / 4 / 4 | 🟢 |

### 2.4 Figure 4, Fig S8, Fig S9 (Receptive fields, V1 map)

| Line | Context | Value in Paper | Macro | Stats | Status |
| :---: | :--- | :---: | :--- | :---: | :---: |
| L258 | Fig 4E correlation | $r = 0.518$, $p < 0.0001$ | `\statFigFourPopR` / `Pval` | 0.518 / $p < 0.0001$ | 🟢 |
| L258 | Fig 4E N | 1,009 neurons, 5 sessions, 4 mice | `\statFigFourPopNeurons` / `Sessions` / `Mice` | 1,009 / 5 / 4 | 🟢 |
| L260 | Fig 4I-K N | 16,974 neurons, 84 sessions | `\statFigFourRFDistNeurons` / `Sessions` | 16,974 / 84 | 🟢 |
| L265, L964 | Depth gradient (corrected) | $p < 0.0001$ | `\statSuppNineGradientPvalCorrected` | 4.9e-09 | 🟢 |
| L265, L966 | Depth gradient (uncorrected) | $p = 0.041$ | `\statSuppNineGradientPvalUncorrected` | 0.0406 | 🟢 |
| L945 | S8D | 1,009 neurons, 5 sessions, 4 mice; contra $r = 0.334$, ipsi $r = 0.031$, $p < 0.0001$ | `\statSuppEightRFCorr*` | same | 🟢 |
| L946 | S8E | 46.2%, $N$ = 89 sessions | `\statSuppEightSigRFMedianTotal` / `SessionsTotal` | 46.2% / 89 | 🟢 |
| L947 | S8F | 1,009 neurons; $r = 0.561$, $p < 0.0001$ | `\statSuppEightPeakDepthNeurons` / `R` / `Pval` | same | 🟢 |
| L963, L964 | S9B-E | $N$ = 16974 neurons, 84 sessions | `\statSuppNineRFNeuronsTotal` / `RFSessionsTotal` | 16,974 / 84 | 🟢 |
| L965 | S9F corrected | near 3315, mid 7288, far 6371 | `\statSuppNine{Near,Mid,Far}Corrected` | 3,315 / 7,288 / 6,371 | 🟢 |
| L966 | S9G-H uncorrected | 4132 / 8100 / 4742 | `\statSuppNine{Near,Mid,Far}Uncorrected` | 4,132 / 8,100 / 4,742 | 🟢 |

### 2.5 Gratings (main text, Discussion, Fig S10)

The main text (L269) quotes the Holm-corrected p-values as "$p$", rounded to 3 decimals; the captions give them as $p_\mathrm{Holm}$ with 4 decimals.

| Line | Context | Value in Paper | Macro | Stats | Status |
| :---: | :--- | :---: | :--- | :---: | :---: |
| L269 | "more than half of depth selective neurons" SF/TF tuned | (derived) | `\statRevGratBothN` / `DepthNeurons` | 390/693 = 56% | 🟢 Consistent |
| L269, L989 | S10G depth vs grating speed | $N$ = 390, $r = -0.16$, $p = 0.028$ (caption 0.0276) | `\statRevGratSpeedDepth{N,Rho,PHolm}` | 390 / -0.16 / 0.0276 | 🟢 |
| L269, L990 | S10H depth vs TF | $N$ = 390, $r = 0.06$, $p = 1.0$ | `\statRevGratTfDepth{N,Rho,PHolm}` | 390 / 0.06 / 1 | 🟢 |
| L269, L991 | S10I depth vs SF | $N$ = 390, $r = 0.36$, $p = 0.029$ (caption 0.0291) | `\statRevGratSfDepth{N,Rho,PHolm}` | 390 / 0.36 / 0.0291 | 🟢 |
| L269, L992 | S10J SF vs OF | $N$ = 333, $r = -0.41$, $p = 0.006$ | `\statRevGratSfOf{N,Rho,PHolm}` | 333 / -0.41 / 0.006 | 🟢 |
| L287 | Discussion: TF shift with running speed | $\sim 0.1$ octaves per octave | `\statRevGratTfShiftDepthMedian` / `OtherMedian` | 0.13 / 0.08 | 🟢 Consistent |
| L555, L985 | Gratings sessions / mice | 6 sessions, 2 mice | `\statRevGratSessions` / `Mice` | 6 / 2 | 🟢 |
| L985 | S10D overlap | 390 observed vs 345.2 expected, $p < 0.0001$ | `\statRevGratBothN` / `BothExpectedN` / `\statRevGratOverlapP` | 390 / 345.16 / 6.2e-05 | 🟢 |
| L986 | S10E DSI | median 0.14 ($N$ = 390) vs 0.19 ($N$ = 1,052); $p = 0.73$ | `\statRevGratDsiMedianDepth` / `Other`, `OtherSftfNeurons`, `DsiP` | 0.14 / 0.19 / 1,052 / 0.725 | 🟢 |
| L987 | S10F direction tuned | $N$ = 130 vs 393; $p = 0.056$ | `\statRevGratDirTunedN` / `OtherN`, `DirP` | 130 / 393 / 0.0561 | 🟢 |
| L779, L995 | S10M speed-modulated TF curves | $N$ = 218 | `\statRevGratSpeedCurvesTfN` | 218 | 🟢 |
| L779, L996 | S10N speed-modulated SF curves | $N$ = 342 | `\statRevGratSpeedCurvesSfN` | 342 | 🟢 |

### 2.6 Methods, Table S1, Fig S2-S4

| Line | Context | Value in Paper | Macro | Stats | Status |
| :---: | :--- | :---: | :--- | :---: | :---: |
| L553 | 5-depth cohort | 2 mice (25 sessions) | `\statSuppDepthPopFiveDepthMice`, `\statFigOneFiveDepthSessions` | 2 / 25 | 🟢 |
| L553 | 8-depth cohort | 5 mice (59 sessions) | `\statSuppDepthPopEightDepthMice` / `Sessions` | 5 / 59 | 🟢 |
| L553 | Frame timing | 91.5 ± 0.9% of frames at 144 Hz; 127 ± 2 Hz | `\statSuppFrameTargetPct` / `FrameRate` | 91.5 ± 0.9% / 127 ± 2 Hz | 🟢 |
| L553 | Display lag | 26.4 ± 0.1 ms | `\statSuppDisplayLag` | 26.4 ± 0.1 ms | 🟢 |
| L575, L854 | Eye tracking (S2K-L) | 16 sessions, 2 mice | `\statSuppEyeSessions` / `Mice` | 16 / 2 | 🟢 |
| L848 | S2A-B sessions | 5 depths $N$ = 25, 8 depths $N$ = 59 | `\statFigOneFiveDepthSessions` / `EightDepthSessions` | 25 / 59 | 🟢 |
| L874 | S3J sessions / mice | GCaMP6f 70 sessions, 6 mice; GCaMP6s 14 sessions, 1 mouse | `\statSuppDepthPopSixf*` / `Sixs*` | 70 / 6; 14 / 1 | 🟢 |
| L874 | S3J medians | 24.6% and 80.3% | `\statSuppDepthPopSixfMedianPctDepthTuned` / `Sixs...` | 24.6% / 80.3% | 🟢 |
| L875 | S3K | 25 sessions, 2 mice, 19,762 neurons | `\statSuppDepthPopFiveDepth*` | 25 / 2 / 19,762 | 🟢 |
| L875 | S3L | 59 sessions, 5 mice, 7,365 neurons | `\statSuppDepthPopEightDepth*` | 59 / 5 / 7,365 | 🟢 |
| L886 | S4C N | 314 neurons, 3 sessions, 2 mice | `\statSuppSizeControlNeurons` / `Sessions` / `Mice` | 314 / 3 / 2 | 🟢 |
| L886 | S4C 5 vs 10° | ratio 1.05, $p_{ratio} = 0.588$, $r = 0.769$, $p < 0.0001$ | `\statSuppSizeControl*FiveTen` | same | 🟢 |
| L886 | S4C 5 vs 20° | ratio 1.00, $p_{ratio} = 0.812$, $r = 0.754$, $p < 0.0001$ | `\statSuppSizeControl*FiveTwenty` | same | 🟢 |
| L886 | S4C 10 vs 20° | ratio 1.00, $p_{ratio} = 0.951$, $r = 0.766$, $p < 0.0001$ | `\statSuppSizeControl*TenTwenty` | same | 🟢 |

The total number of animals (L515) is exported by `methods_dataset_tables` (`\statTableSOneTotalMice`, 12): see Section 3.

Table S1 (L527-532). Every cell is exported by `methods_dataset_tables` (since 2026-11-19): `\statTableSOne<Row>{Mice,Sex,Indicators,Sessions,Neurons}` with `<Row>` = `ClosedLoop`, `Motorized`, `OpenLoop`, `Gratings`, `Multiday`, `Multidepth`. `Sex` and `Indicators` are the cell text (e.g. "1M, 6F", "GCaMP6f (6), GCaMP6s (1)"; indicators sorted by mouse count, then name). All 30 cells match the `.tex` on 2026-11-19. Mice, sex and sessions also agree with the figure metrics (`\statFigOneTotalMice`/`MaleMice`/`FemaleMice`/`TotalSessions`, `\statFigTwoOpenLoopMice`/`Sessions`, `\statFigThreePolarMice`/`Sessions`, `\statRevGratMice`/`Sessions`, `\statFigOneMultidayMice`/`Sessions`), and the closed loop indicators with `\statSuppDepthPopSixfMice` / `SixsMice` (6 / 1).

| Line | Experiment | Mice | Sex | Sessions | Neurons in paper | Neurons in stats | Macros | Status |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- | :---: |
| L527 | Closed loop | 7 | 1M, 6F | 84 | 59,937 | 59,937 | `\statTableSOneClosedLoopNeurons`; `\statFigOneTotalMice`, `MaleMice`/`FemaleMice`, `TotalSessions` | 🟢 |
| L528 | Motorized wheel | 4 | 3M, 1F | 4 | 2,573 | 2,573 | `\statTableSOneMotorizedNeurons`; `\statFigThreePolarMice`/`Sessions` | 🟢 |
| L529 | Open loop | 7 | 1M, 6F | 34 | 22,851 | 22,851 | `\statTableSOneOpenLoopNeurons`; `\statFigTwoOpenLoopMice`/`Sessions` | 🟢 |
| L530 | Drifting gratings | 2 | 2F | 6 | 3,012 | 3,012 | `\statTableSOneGratingsNeurons`; `\statRevGratMice`, `Sessions`, `Neurons` | 🟢 |
| L531 | Multi-day tracking | 4 | 3M, 1F | 19 | 11,518 | 11,518 | `\statTableSOneMultidayNeurons`; `\statFigOneMultidayMice`/`Sessions` | 🟢 |
| L532 | Multi-depth RF | 4 | 3M, 1F | 5 | 3,799 | 3,799 | `\statTableSOneMultidepthNeurons`; `\statFigFourPopMice`/`Sessions` | 🟢 |

The table has no total row. If one is added, it should read 77,827 (`\statTableSOneTotalRecordedNeurons`).

---

## 3. Static parameters and hand-checked values

These are not exported by the pipeline.

Checked against code on 2026-11-12 (line numbers refreshed 2026-11-19, text unchanged):

| Line | Parameter | Value in paper | Code |
| :---: | :--- | :--- | :--- |
| L549, L661 | Motorized wheel grid | 30 combinations | 5 speeds × 6 optic flow speeds |
| L579 | Saccade threshold | 70 deg/s, "a $\sim5$ degrees change in a frame" | `figsupp2_speed.ipynb`; 70 deg/s at ~15 Hz ≈ 4.7 deg/frame |
| L659 | Motorized analysis window | ~2 s at 61 cm/s, ~9.5 s at 3.8 cm/s | consistent with the 7.7 cm/s² ramp |
| L661 | Motorized fit bounds | 2.3-101 cm/s, 0.61-1688 °/s ($e^{0.5}$ padding) | recomputed |
| L773 | DSI permutation test | 20,000 permutations | `N_PERM = 20000` in `figsupp10_gratings.ipynb` |
| L777 | Grating correlation bootstrap | 20,000 iterations | `N_BOOTS = 20000` in `figsupp10_gratings.ipynb` |

Changed in the `.tex` on 2026-11-18 from Flexilims metadata (see `TODO_stats.md`); re-checked on 2026-11-19 against Flexilims and the Table S1 session list (112 sessions):

| Line | Parameter | Value in paper | Note |
| :---: | :--- | :--- | :--- |
| L515 | Animals | Twelve mice: 7 Emx1-Cre × Ai95D, 5 CamKII-tTA × TRE-GCaMP6s | 🟢 Now exported: `\statTableSOneTotalMiceWord` (Twelve), `\statTableSOneTotalMice` (12), `\statTableSOneEmxMice` (7), `\statTableSOneGsixsMice` (5), from `methods_dataset_tables`. `\statSuppEightSigRFMiceTotal` (11) is a Fig S8 count and does not include PZAG22.1a (treadmill only) |
| L538 | Age at imaging | 3 to 7 months (up to 11 months) | 🟡 Max 11.4 months (PZAH10.2f, Flexilims birth dates are DD/MM/YYYY), so "up to 11" is right. Sessions: min 3.8, 5-95% 4.1-10.6, median 7.7, IQR 6.6-8.4 months; only 63% of sessions are 3-7 months, so "typically 3 to 7" understates the original cohort (6.5-11 months) |
| L562 | Field of view | ~661 µm (7 mice) or ~860 µm (5 mice) | 🟡 Split matches the cohorts (7 original, 5 revision mice). No zoom/FOV in Flexilims and no raw TIFFs on BlackPasspo, so PZAG22.1a's ~860 µm still needs a ScanImage header check on NEMO |

Not checked: the 1,000 bootstrap iterations for the speed-modulated curves (L779). The decoder smoothing sentence (L782) is out of date, see Section 1.

---

## 4. Exported metrics not cited in the manuscript (99)

Values and descriptions are in [MANUSCRIPT_STATS.md](../v1_depth_map/stats/MANUSCRIPT_STATS.md); they are not copied here.

| Notebook | Macros | Note |
| :--- | :--- | :--- |
| `figure2_openloop` (6) | `\statFigTwoRSRatioMedian`, `RSRatioPval`, `DecoderFiveDepthSessions`, `DecoderMice`, `DecoderErrDiffCI`, `DecoderErrClosedLowerN` | `DecoderErrDiffCI` (0.13-0.47) and `DecoderErrClosedLowerN` (29 of 34 sessions) could support "consistently higher" at L207 |
| `figure2_rsof_integration` (12) | `\statFigTwoClosedLoopConjunctiveNeurons`, `ConjunctiveThreshold`, and the 10 pairwise `\statFigTwoPval<A>Vs<B>` | L646 describes the threshold procedurally; L200 cites the summary `ModelComparisonPval` |
| `figure3_depth_cells` (6) | `\statFigThreePrefRSBelowRangeN` / `Pct`; `\statFigThreeOrientationFortyFiveN`, `OrientationNinetyN`, `OrientationMinusFortyFiveN` / `Pct` | Third Fig 3G group (39/227, 17%) is not mentioned at L211. L240 gives the count only for the 0° group (103/214 at 45° and 39/214 at 90° are available), and says negative slopes are "almost entirely absent" without the number (5/214, 2%) |
| `figure4_receptive_fields` (2) | `\statFigFourRFDistMice`, `RFDistPairs` | Caption 4I-K gives neurons and sessions only |
| `figsupp3_depth_pop` (6) | `\statSuppDepthPop{FiveDepth,EightDepth}MedianPrefDepth` / `PrefDepthIQR`, `KsStat`, `KsPval` | |
| `figsupp5_rsof` (10) | `\statSuppRSOFCorrOFDepth{R,Pval}`, `CorrRSDepthAll{R,Pval}`, `CorrRSDepthBin{One,Two,Three}{R,Pval}` | |
| `figsupp8_multidepth_receptive_fields` (9) | `\statSuppEightSigRF{Sessions,Median}{Single,Multi}`, `SigRFNeuronsTotal`, `SigRFSigNeuronsTotal`, `SigRFMiceTotal`, `PeakDepthMedianRatio`, `PeakDepthRatioIQR` | `SigRFMiceTotal` (11) is the Fig S8 mouse count; the Animals sentence uses `\statTableSOneTotalMice` (12, Section 3) |
| `figsupp9_v1_depth_map` (3) | `\statSuppNineAPCorrR`, `APCorrPval`, `APCorrCI` | The commented-out sentence that cited the AP correlation is no longer in the `.tex`; p is a hierarchical bootstrap since 2026-11-18 |
| `figsupp10_gratings` (44) | `DepthNeurons`, `RsofThreshold`, `RsofNeurons`, `PctDepth`, `SftfNeurons`, `PctSftf`, `PctBoth`, `PctBothExpected`, `OverlapOR`, `DsiDiff`; uncorrected `{SpeedDepth,TfDepth,SfDepth,SfOf}P`; `SfRs{N,Rho,P,PHolm}`; `SpeedCurves{Tf,Sf}{Sessions,Mice}`; `{Tf,Sf}Shift{Depth,Other}{CiLow,CiHigh,N,PZero,POne}`; `SfShift{Depth,Other}Median` (all prefixed `\statRevGrat`) | `DepthNeurons` (693) is implicit in L269 "more than half" |
| `methods_dataset_tables` (3) | `\statTableSOneTotalRecordedNeurons`, `\statTableSOneEmxMice`, `\statTableSOneGsixsMice` | 77,827 neurons, Table S1 has no total row. The line counts (7, 5) match the literals at L515 (the total, 12, is matched in Section 3) |

---

## 5. Maintaining this document

- After regenerating stats (`python run_figures_pipeline.py`, then `python -m v1_depth_map.stats.build_manuscript_stats`), update the Stats column and Section 1 for any value that changed.
- `--check` currently flags 7 of 15 YAMLs as stale. This is an mtime check (notebook file saved after `generated_at`), and all 15 were regenerated on 2026-11-18. Re-run the notebooks before final submission.
- A full audit means reading the `.tex` from `\begin{document}`, extracting all numeric tokens and comparing them against the `raw` / `formatted` YAML values. Line numbers here shift whenever the `.tex` is edited; refresh them at the next full audit.
- The Overleaf/Dropbox folder is not readable from the agent sandbox; copy `v1_depth_map.tex` into the repo root (untracked) for an audit.
