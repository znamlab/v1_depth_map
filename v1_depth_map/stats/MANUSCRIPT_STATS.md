# Manuscript Statistics Manifest (Audit Dashboard)

*Compiled from figure notebook outputs at 2026-09-20 17:23:22.*

This document is compiled directly from statistics generated inside the figure notebooks.
Because each number is written by the exact code cell that generates the corresponding figure panel,
these values cannot drift out of sync with the figures.

> ✅ All 81 values were written by a notebook run.

---

## Figure 1: Virtual Depth Selectivity & Experience Independence

Source notebook: `figure1_depth_selectivity.ipynb` (notebook, generated 2026-09-19T16:41:49)

| Macro Name | Formatted Value | Raw Value | Source | Description / Manuscript Context |
| :--- | :---: | :---: | :---: | :--- |
| `\statFigOneTotalSessions` | `84` | 84 | notebook | Total two-photon imaging sessions for virtual depth experiment |
| `\statFigOneFiveDepthSessions` | `25` | 25 | notebook | Sessions with 5 virtual depths (6 cm to 600 cm) |
| `\statFigOneEightDepthSessions` | `59` | 59 | notebook | Sessions with 8 virtual depths (5 cm to 640 cm) |
| `\statFigOneTotalMice` | `7` | 7 | notebook | Total mice recorded in layer 2/3 of V1 |
| `\statFigOneTotalNeurons` | `59,937` | 59937 | notebook | Total layer 2/3 excitatory neurons recorded |
| `\statFigOneDepthNeurons` | `27,094` | 27094 | notebook | Neurons exhibiting significant depth selectivity |
| `\statFigOnePctDepthNeurons` | `45.2\%` | 45.2041310042211 | notebook | Percentage of layer 2/3 neurons with depth selectivity |
| `\statFigOneMultidayMice` | `4` | 4 | notebook | Number of mice in multiday tracking experiment |
| `\statFigOneMultidaySessions` | `19` | 19 | notebook | Number of sessions across consecutive days |
| `\statFigOneMultidayPairs` | `374` | 374 | notebook | Consecutive day neuron pairs tracked |
| `\statFigOneMultidayTrackedNeurons` | `234` | 234 | notebook | Unique neurons tracked across consecutive days |

## Figure 2: Visuomotor Integration Models & Open-Loop Replay

Source notebook: `figure_openloop.ipynb` (notebook, generated 2026-09-20T13:02:10)

| Macro Name | Formatted Value | Raw Value | Source | Description / Manuscript Context |
| :--- | :---: | :---: | :---: | :--- |
| `\statFigTwoOpenLoopTotalDepthNeurons` | `9,429` | 9429 | notebook | Total depth-selective neurons in open-loop sessions (Methods L624) |
| `\statFigTwoOpenLoopNeurons` | `1,233` | 1233 | notebook | Depth-selective neurons compared between closed and open loop (Fig 2H-J) |
| `\statFigTwoAmpRatioPval` | `$p = 0.651$` | 0.6514 | notebook | Hierarchical bootstrap p-value of peak response closed vs open loop ratio |
| `\statFigTwoRSCorrR` | `$r = 0.552$` | 0.5520813429666219 | notebook | Correlation r of preferred running speed (closed vs open loop) |
| `\statFigTwoRSCorrPval` | `$p < 0.0001$` | 0.0 | notebook | Correlation p-value of preferred running speed (closed vs open loop) |
| `\statFigTwoOFCorrR` | `$r = 0.711$` | 0.710987617320202 | notebook | Correlation r of preferred optic flow speed (closed vs open loop) |
| `\statFigTwoOFCorrPval` | `$p < 0.0001$` | 0.0 | notebook | Correlation p-value of preferred optic flow speed (closed vs open loop) |
| `\statFigTwoDecoderTotalSessions` | `34` | 34 | notebook | Sessions for closed vs open loop SVM decoding (Fig 2K) |
| `\statFigTwoDecoderEightDepthSessions` | `27` | 27 | notebook | Sessions with 8 depths for confusion matrices (Fig 2L) |

Source notebook: `figure_rsof_integration.ipynb` (notebook, generated 2026-09-20T12:49:21)

| Macro Name | Formatted Value | Raw Value | Source | Description / Manuscript Context |
| :--- | :---: | :---: | :---: | :--- |
| `\statFigTwoModelSessions` | `84` | 84 | notebook | Sessions included in model comparison analysis (Fig 2E) |
| `\statFigTwoModelNeurons` | `21,318` | 21318 | notebook | Depth-tuned neurons with at least one significant RS/OF model fit (Fig 2E) |
| `\statFigTwoClosedLoopConjunctiveNeurons` | `17,953` | 17953 | notebook | Depth-tuned neurons with significant conjunctive model fit in closed loop (Methods L621) |
| `\statFigTwoModelComparisonPval` | `$p < 0.0001$` | 0.0001 | notebook | Additive and Conjunctive models outperforming isolated speed and ratio models (Fig 2E) |

## Figure 3: Motorized Wheel & Visuomotor Gain Modulation

Source notebook: `figure_depth_cells.ipynb` (notebook, generated 2026-09-19T17:39:50)

| Macro Name | Formatted Value | Raw Value | Source | Description / Manuscript Context |
| :--- | :---: | :---: | :---: | :--- |
| `\statFigThreeMotorizedDepthNeurons` | `285` | 285 | notebook | Depth-tuned neurons compared between free locomotion and motorized wheel (Fig 3F) |
| `\statFigThreePolarNeurons` | `285` | 285 | notebook | Neurons with significant 2D-Gaussian fit on motorized wheel (Fig 3K) |
| `\statFigThreePolarSessions` | `4` | 4 | notebook | Sessions in motorized wheel elongation analysis (Fig 3K) |
| `\statFigThreePolarMice` | `4` | 4 | notebook | Mice in motorized wheel elongation analysis (Fig 3K) |
| `\statFigThreeElongationCutoffRatio` | `266/285` | 266/285 | notebook | Fraction of neurons with elongation ratio > 1.4 (Fig 3K) |
| `\statFigThreeElongatedNeurons` | `266` | 266 | notebook | Neurons with elongated ellipses for orientation distribution (Fig 3L) |
| `\statFigThreeVonMisesK` | `3` | 3 | notebook | Best axial von Mises mixture number of components (Fig 3L) |
| `\statFigThreeVonMisesComponents` | `$3^\circ$ (14\%), $43^\circ$ (76\%), $90^\circ$ (11\%)` | $3^\circ$ (14\%), $43^\circ$ (76\%), $90^\circ$ (11\%) | notebook | Component centers and mixture weights for orientation distribution (Fig 3L) |

## Figures 4 & 5: Three-Dimensional Receptive Fields & V1 Depth Map

Source notebook: `figure_rf.ipynb` (notebook, generated 2026-09-20T12:44:28)

| Macro Name | Formatted Value | Raw Value | Source | Description / Manuscript Context |
| :--- | :---: | :---: | :---: | :--- |
| `\statFigFourRFNeuronsTotal` | `14,173` | 14173 | notebook | Total depth-selective neurons with significant receptive fields |
| `\statFigFourRFSessionsTotal` | `84` | 84 | notebook | Total sessions contributing to RF mapping |
| `\statFigFiveNearUncorrected` | `3,078` | 3078 | notebook | Neurons preferring near depths (<20 cm) uncorrected |
| `\statFigFiveMidUncorrected` | `6,744` | 6744 | notebook | Neurons preferring intermediate depths (20-100 cm) uncorrected |
| `\statFigFiveFarUncorrected` | `4,351` | 4351 | notebook | Neurons preferring far depths (>100 cm) uncorrected |
| `\statFigFiveNearCorrected` | `2,476` | 2476 | notebook | Neurons preferring near depths (<20 cm) corrected for viewing angle |
| `\statFigFiveMidCorrected` | `5,832` | 5832 | notebook | Neurons preferring intermediate depths (20-100 cm) corrected for viewing angle |
| `\statFigFiveFarCorrected` | `5,865` | 5865 | notebook | Neurons preferring far depths (>100 cm) corrected for viewing angle |
| `\statFigFiveGradientPvalCorrected` | `$p = 7.40e-09$` | 7.403750800566618e-09 | notebook | Significance of 3D RF depth gradient across visual space (corrected) |
| `\statFigFiveGradientPvalUncorrected` | `$p = 0.0175$` | 0.01752264341898344 | notebook | Significance of depth gradient across visual space (uncorrected) |
| `\statFigFiveAPCorrR` | `$r = -0.166$` | -0.16571200832758723 | notebook | Correlation between preferred depth and anterior-posterior location in V1 |

## Supplementary Figures & Methods: Controls & Experimental Specs

Source notebook: `figsupp1_vis_stim_sync.ipynb` (notebook, generated 2026-09-19T23:53:49)

| Macro Name | Formatted Value | Raw Value | Source | Description / Manuscript Context |
| :--- | :---: | :---: | :---: | :--- |
| `\statSuppDisplayLag` | `$26.6 \pm 1.4~\mathrm{ms}$` | 26.631505885895553 | notebook | Average display lag between wheel movement and monitor update (5-colour photodiode cohort) |
| `\statSuppFrameRate` | `$127 \pm 2~\mathrm{Hz}$` | 127.03871492644178 | notebook | Average effective display refresh rate across sessions (2-colour photodiode cohort) |
| `\statSuppFrameTargetPct` | `$91.5 \pm 0.9\%$` | 91.46000966969683 | notebook | Percentage of frames displayed at full nominal 144 Hz rate (2-colour photodiode cohort) |

Source notebook: `figsupp2_speed.ipynb` (notebook, generated 2026-09-19T17:45:59)

| Macro Name | Formatted Value | Raw Value | Source | Description / Manuscript Context |
| :--- | :---: | :---: | :---: | :--- |
| `\statSuppEyeSessions` | `16` | 16 | notebook | Sessions with pupil/gaze tracking in Fig S2 |
| `\statSuppEyeMice` | `2` | 2 | notebook | Mice with pupil/gaze tracking in Fig S2 |

Source notebook: `figsupp4_size_control.ipynb` (notebook, generated 2026-09-20T12:45:41)

| Macro Name | Formatted Value | Raw Value | Source | Description / Manuscript Context |
| :--- | :---: | :---: | :---: | :--- |
| `\statSuppSizeControlNeurons` | `314` | 314 | notebook | Neurons tested across sphere sizes (5, 10, 20 deg) in Fig S3 |
| `\statSuppSizeControlSessions` | `3` | 3 | notebook | Sessions for stimulus size invariance experiment |
| `\statSuppSizeControlMice` | `2` | 2 | notebook | Mice tested for stimulus size invariance |
| `\statSuppSizeControlRatioFiveTen` | `1.05` | 1.0520057180209808 | notebook | Preferred depth median ratio: 5 vs 10 deg spheres |
| `\statSuppSizeControlRatioPvalFiveTen` | `$p_{ratio} = 0.588$` | 0.5877 | notebook | Bootstrap p-value for ratio: 5 vs 10 deg spheres |
| `\statSuppSizeControlCorrRFiveTen` | `$r = 0.769$` | 0.7685199631136754 | notebook | Spearman correlation r: 5 vs 10 deg spheres |
| `\statSuppSizeControlCorrPvalFiveTen` | `$p_{correlation} < 0.0001$` | 1.8162415814020279e-62 | notebook | Spearman correlation p-value: 5 vs 10 deg spheres |
| `\statSuppSizeControlRatioFiveTwenty` | `1.00` | 0.9999999998197141 | notebook | Preferred depth median ratio: 5 vs 20 deg spheres |
| `\statSuppSizeControlRatioPvalFiveTwenty` | `$p_{ratio} = 0.812$` | 0.8123 | notebook | Bootstrap p-value for ratio: 5 vs 20 deg spheres |
| `\statSuppSizeControlCorrRFiveTwenty` | `$r = 0.754$` | 0.7543691238996617 | notebook | Spearman correlation r: 5 vs 20 deg spheres |
| `\statSuppSizeControlCorrPvalFiveTwenty` | `$p_{correlation} < 0.0001$` | 5.532267866321184e-59 | notebook | Spearman correlation p-value: 5 vs 20 deg spheres |
| `\statSuppSizeControlRatioTenTwenty` | `1.00` | 0.99682570846561 | notebook | Preferred depth median ratio: 10 vs 20 deg spheres |
| `\statSuppSizeControlRatioPvalTenTwenty` | `$p_{ratio} = 0.951$` | 0.9507 | notebook | Bootstrap p-value for ratio: 10 vs 20 deg spheres |
| `\statSuppSizeControlCorrRTenTwenty` | `$r = 0.766$` | 0.7659021899666625 | notebook | Spearman correlation r: 10 vs 20 deg spheres |
| `\statSuppSizeControlCorrPvalTenTwenty` | `$p_{correlation} < 0.0001$` | 8.359022898283888e-62 | notebook | Spearman correlation p-value: 10 vs 20 deg spheres |

Source notebook: `figsupp5_rsof.ipynb` (notebook, generated 2026-09-20T17:23:15)

| Macro Name | Formatted Value | Raw Value | Source | Description / Manuscript Context |
| :--- | :---: | :---: | :---: | :--- |
| `\statSuppRSOFCorrOFDepthR` | `$r = -0.595$` | -0.5947295645149252 | notebook | Correlation between preferred depth and optic flow speed (Fig S5I) |
| `\statSuppRSOFCorrOFDepthPval` | `$p < 0.0001$` | 0.0 | notebook | P-value of correlation between preferred depth and optic flow speed |
| `\statSuppRSOFCorrRSDepthAllR` | `$r = -0.055$` | -0.055057358507232014 | notebook | Correlation between preferred depth and running speed for all neurons (Fig S5J) |
| `\statSuppRSOFCorrRSDepthAllPval` | `$p = 0.553$` | 0.5534 | notebook | P-value of correlation between preferred depth and running speed for all neurons |
| `\statSuppRSOFCorrRSDepthBinOneR` | `$r = 0.507$` | 0.5065385736128153 | notebook | Correlation between preferred depth and running speed (OF 1-10 deg/s) |
| `\statSuppRSOFCorrRSDepthBinOnePval` | `$p < 0.0001$` | 0.0 | notebook | P-value of correlation (OF 1-10 deg/s) |
| `\statSuppRSOFCorrRSDepthBinTwoR` | `$r = 0.396$` | 0.395974503563562 | notebook | Correlation between preferred depth and running speed (OF 10-100 deg/s) |
| `\statSuppRSOFCorrRSDepthBinTwoPval` | `$p < 0.0001$` | 0.0 | notebook | P-value of correlation (OF 10-100 deg/s) |
| `\statSuppRSOFCorrRSDepthBinThreeR` | `$r = 0.364$` | 0.363561848340317 | notebook | Correlation between preferred depth and running speed (OF 100-1000 deg/s) |
| `\statSuppRSOFCorrRSDepthBinThreePval` | `$p < 0.0001$` | 0.0 | notebook | P-value of correlation (OF 100-1000 deg/s) |

Source notebook: `figsupp7_simulation_control.ipynb` (notebook, generated 2026-09-20T12:46:44)

| Macro Name | Formatted Value | Raw Value | Source | Description / Manuscript Context |
| :--- | :---: | :---: | :---: | :--- |
| `\statSuppSimulFreeNeurons` | `316` | 316 | notebook | Free locomotion simulated neurons (Fig S7C) |
| `\statSuppSimulFreeElongatedRatio` | `297/316` | 297/316 | notebook | Free locomotion simulated neurons with elongation > 1.4 ratio |
| `\statSuppSimulFreeElongatedPct` | `94\%` | 93.9873417721519 | notebook | Free locomotion simulated neurons with elongation > 1.4 percentage |
| `\statSuppSimulMotorNeurons` | `295` | 295 | notebook | Motorized wheel simulated neurons (Fig S7D) |
| `\statSuppSimulMotorElongatedRatio` | `20/295` | 20/295 | notebook | Motorized wheel simulated neurons with elongation > 1.4 ratio |
| `\statSuppSimulMotorElongatedPct` | `7\%` | 6.779661016949152 | notebook | Motorized wheel simulated neurons with elongation > 1.4 percentage |
| `\statSuppSimulSessions` | `4` | 4 | notebook | Sessions included in simulation control (Fig S7) |
| `\statSuppSimulMice` | `4` | 4 | notebook | Mice included in simulation control (Fig S7) |

---

## Usage in LaTeX (`v1_depth_map.tex`)

To cite any statistic in the LaTeX manuscript, include `\input{manuscript_stats.tex}` in the preamble,
then use the macro with empty braces to preserve following spaces:

```latex
\input{manuscript_stats.tex}

% In text:
Across the population, \statFigOnePctDepthNeurons{} of cells
(\statFigOneDepthNeurons{} of \statFigOneTotalNeurons{} neurons) exhibited significant depth selectivity...
```
