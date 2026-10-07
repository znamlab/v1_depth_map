# Manuscript Statistics Manifest (Audit Dashboard)

*Compiled from figure notebook outputs at 2026-10-07 17:54:09.*

This document is compiled directly from statistics generated inside the figure notebooks.
Because each number is written by the exact code cell that generates the corresponding figure panel,
these values cannot drift out of sync with the figures.

> ✅ All 142 values were written by a notebook run.

---

## Figure 1: Virtual Depth Selectivity & Experience Independence

Source notebook: `figure1_depth_selectivity.ipynb` (notebook, generated 2026-09-29T15:03:59)

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

Source notebook: `figure2_openloop.ipynb` (notebook, generated 2026-10-01T18:06:58)

| Macro Name | Formatted Value | Raw Value | Source | Description / Manuscript Context |
| :--- | :---: | :---: | :---: | :--- |
| `\statFigTwoOpenLoopTotalDepthNeurons` | `9,429` | 9429 | notebook | Total depth-selective neurons in open-loop sessions (Methods L624) |
| `\statFigTwoOpenLoopNeurons` | `3,948` | 3948 | notebook | Depth-selective neurons compared between closed and open loop (Fig 2H-J) |
| `\statFigTwoOpenLoopSessions` | `34` | 34 | notebook | Sessions contributing to the closed vs open loop comparison (Fig 2H-J) |
| `\statFigTwoOpenLoopMice` | `7` | 7 | notebook | Mice contributing to the closed vs open loop comparison (Fig 2H-J) |
| `\statFigTwoAmpRatioMedian` | `0.98` | 0.9844723486949924 | notebook | Median ratio of peak response closed vs open loop, the triangle in Fig 2H |
| `\statFigTwoAmpRatioPval` | `$p = 0.743$` | 0.7427 | notebook | Hierarchical bootstrap p-value of peak response closed vs open loop ratio |
| `\statFigTwoRSCorrR` | `$r = 0.469$` | 0.46908924802504137 | notebook | Correlation r of preferred running speed (closed vs open loop) |
| `\statFigTwoRSCorrPval` | `$p < 0.0001$` | 0.0 | notebook | Correlation p-value of preferred running speed (closed vs open loop) |
| `\statFigTwoRSRatioMedian` | `1.0456` | 1.0456273579463469 | notebook | Hierarchical bootstrap median of preferred running speed closed vs open loop ratio |
| `\statFigTwoRSRatioPval` | `$p = 0.214$` | 0.2143 | notebook | Hierarchical bootstrap p-value of preferred running speed closed vs open loop ratio |
| `\statFigTwoOFCorrR` | `$r = 0.535$` | 0.5349328735076935 | notebook | Correlation r of preferred optic flow speed (closed vs open loop) |
| `\statFigTwoOFCorrPval` | `$p < 0.0001$` | 0.0 | notebook | Correlation p-value of preferred optic flow speed (closed vs open loop) |
| `\statFigTwoDecoderTotalSessions` | `34` | 34 | notebook | Sessions for closed vs open loop SVM decoding (Fig 2K) |
| `\statFigTwoDecoderFiveDepthSessions` | `7` | 7 | notebook | Sessions with 5 depths in the decoder comparison (Fig 2K) |
| `\statFigTwoDecoderEightDepthSessions` | `27` | 27 | notebook | Sessions with 8 depths for confusion matrices (Fig 2L) |

Source notebook: `figure2_rsof_integration.ipynb` (notebook, generated 2026-09-21T14:10:26)

| Macro Name | Formatted Value | Raw Value | Source | Description / Manuscript Context |
| :--- | :---: | :---: | :---: | :--- |
| `\statFigTwoModelSessions` | `84` | 84 | notebook | Sessions included in model comparison analysis (Fig 2E) |
| `\statFigTwoModelNeurons` | `21,318` | 21318 | notebook | Depth-tuned neurons with at least one significant RS/OF model fit (Fig 2E) |
| `\statFigTwoClosedLoopConjunctiveNeurons` | `17,953` | 17953 | notebook | Depth-tuned neurons with significant conjunctive model fit in closed loop (Methods L621) |
| `\statFigTwoPvalRunningSpeedVsOpticFlow` | `$p < 0.0001$` | 5e-05 | notebook | Hierarchical bootstrap on the proportion of neurons best fit by Running speed vs Optic flow: median difference -0.057, 95% CI [-0.089, -0.027], 20,000 resamples (Fig 2E) |
| `\statFigTwoPvalRunningSpeedVsRsOfRatio` | `$p < 0.0001$` | 5e-05 | notebook | Hierarchical bootstrap on the proportion of neurons best fit by Running speed vs RS/OF: median difference -0.072, 95% CI [-0.121, -0.037], 20,000 resamples (Fig 2E) |
| `\statFigTwoPvalRunningSpeedVsAdditive` | `$p < 0.0001$` | 5e-05 | notebook | Hierarchical bootstrap on the proportion of neurons best fit by Running speed vs Additive: median difference -0.317, 95% CI [-0.369, -0.280], 20,000 resamples (Fig 2E) |
| `\statFigTwoPvalRunningSpeedVsConjunctive` | `$p < 0.0001$` | 5e-05 | notebook | Hierarchical bootstrap on the proportion of neurons best fit by Running speed vs Conjunctive: median difference -0.363, 95% CI [-0.437, -0.327], 20,000 resamples (Fig 2E) |
| `\statFigTwoPvalOpticFlowVsRsOfRatio` | `$p = 0.372$` | 0.3718 | notebook | Hierarchical bootstrap on the proportion of neurons best fit by Optic flow vs RS/OF: median difference -0.014, 95% CI [-0.054, 0.014], 20,000 resamples (Fig 2E) |
| `\statFigTwoPvalOpticFlowVsAdditive` | `$p < 0.0001$` | 5e-05 | notebook | Hierarchical bootstrap on the proportion of neurons best fit by Optic flow vs Additive: median difference -0.261, 95% CI [-0.326, -0.198], 20,000 resamples (Fig 2E) |
| `\statFigTwoPvalOpticFlowVsConjunctive` | `$p < 0.0001$` | 5e-05 | notebook | Hierarchical bootstrap on the proportion of neurons best fit by Optic flow vs Conjunctive: median difference -0.302, 95% CI [-0.398, -0.266], 20,000 resamples (Fig 2E) |
| `\statFigTwoPvalRsOfRatioVsAdditive` | `$p < 0.0001$` | 5e-05 | notebook | Hierarchical bootstrap on the proportion of neurons best fit by RS/OF vs Additive: median difference -0.248, 95% CI [-0.318, -0.176], 20,000 resamples (Fig 2E) |
| `\statFigTwoPvalRsOfRatioVsConjunctive` | `$p < 0.0001$` | 5e-05 | notebook | Hierarchical bootstrap on the proportion of neurons best fit by RS/OF vs Conjunctive: median difference -0.290, 95% CI [-0.367, -0.239], 20,000 resamples (Fig 2E) |
| `\statFigTwoPvalAdditiveVsConjunctive` | `$p = 0.155$` | 0.1554 | notebook | Hierarchical bootstrap on the proportion of neurons best fit by Additive vs Conjunctive: median difference -0.047, 95% CI [-0.126, 0.026], 20,000 resamples (Fig 2E) |
| `\statFigTwoModelComparisonPval` | `$p < 0.0001$` | 5e-05 | notebook | Additive and Conjunctive models outperforming isolated speed and ratio models (Fig 2E): the largest of the 6 hierarchical-bootstrap p-values that claim rests on, floored at the 1/20,000 resolution limit |

## Figure 3: Motorized Wheel & Visuomotor Gain Modulation

Source notebook: `figure3_depth_cells.ipynb` (notebook, generated 2026-09-19T17:39:50)

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

## Figure 4: Three-Dimensional Receptive Fields

Source notebook: `figure4_receptive_fields.ipynb` (notebook, generated 2026-10-06T14:23:56)

| Macro Name | Formatted Value | Raw Value | Source | Description / Manuscript Context |
| :--- | :---: | :---: | :---: | :--- |
| `\statFigFourPopNeurons` | `1,009` | 1009 | notebook | Depth-tuned neurons with significant multi-depth RFs (Fig 4E) |
| `\statFigFourPopSessions` | `5` | 5 | notebook | Multi-depth sessions contributing neurons (Fig 4E) |
| `\statFigFourPopMice` | `4` | 4 | notebook | Mice contributing neurons (Fig 4E) |
| `\statFigFourRFDistNeurons` | `16,945` | 16945 | notebook | Depth-tuned neurons with significant RFs in pairwise distance analysis (Fig 4I-K) |
| `\statFigFourRFDistSessions` | `84` | 84 | notebook | Sessions in pairwise distance analysis (Fig 4I-K) |
| `\statFigFourRFDistMice` | `7` | 7 | notebook | Mice in pairwise distance analysis (Fig 4I-K) |
| `\statFigFourRFDistPairs` | `6,536,044` | 6536044 | notebook | Neuron pairs more than 10 um apart, within the binned range (Fig 4I-K) |

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

Source notebook: `figsupp3_depth_pop.ipynb` (notebook, generated 2026-10-01T16:59:04)

| Macro Name | Formatted Value | Raw Value | Source | Description / Manuscript Context |
| :--- | :---: | :---: | :---: | :--- |
| `\statSuppDepthPopSixfMice` | `6` | 6 | notebook | GCaMP6f mice in Fig S3J |
| `\statSuppDepthPopSixfSessions` | `70` | 70 | notebook | GCaMP6f sessions in Fig S3J |
| `\statSuppDepthPopSixfMedianPctDepthTuned` | `24.6\%` | 24.592213531823337 | notebook | Median across GCaMP6f sessions of the percentage of depth-tuned neurons (Fig S3J) |
| `\statSuppDepthPopSixsMice` | `1` | 1 | notebook | GCaMP6s mice in Fig S3J |
| `\statSuppDepthPopSixsSessions` | `14` | 14 | notebook | GCaMP6s sessions in Fig S3J |
| `\statSuppDepthPopSixsMedianPctDepthTuned` | `80.3\%` | 80.25822204351573 | notebook | Median across GCaMP6s sessions of the percentage of depth-tuned neurons (Fig S3J) |
| `\statSuppDepthPopFiveDepthSessions` | `25` | 25 | notebook | Sessions with 5 virtual depths in Fig S3K |
| `\statSuppDepthPopFiveDepthMice` | `2` | 2 | notebook | Mice with 5 virtual depths in Fig S3K |
| `\statSuppDepthPopFiveDepthNeurons` | `19,762` | 19762 | notebook | Depth-tuned neurons from 5-depth sessions in Fig S3K |
| `\statSuppDepthPopFiveDepthMedianPrefDepth` | `40.9` | 40.909627113949725 | notebook | Median preferred depth (cm) of depth-tuned neurons, 5-depth sessions (Fig S3K) |
| `\statSuppDepthPopFiveDepthPrefDepthIQR` | `18.7--99.6` | 18.656769490288298-99.58611036461171 | notebook | Interquartile range of preferred depth (cm), 5-depth sessions (Fig S3K) |
| `\statSuppDepthPopEightDepthSessions` | `59` | 59 | notebook | Sessions with 8 virtual depths in Fig S3L |
| `\statSuppDepthPopEightDepthMice` | `5` | 5 | notebook | Mice with 8 virtual depths in Fig S3L |
| `\statSuppDepthPopEightDepthNeurons` | `7,332` | 7332 | notebook | Depth-tuned neurons from 8-depth sessions in Fig S3L |
| `\statSuppDepthPopEightDepthMedianPrefDepth` | `51.0` | 50.99836974229177 | notebook | Median preferred depth (cm) of depth-tuned neurons, 8-depth sessions (Fig S3L) |
| `\statSuppDepthPopEightDepthPrefDepthIQR` | `11.8--239.3` | 11.756201639994341-239.30110554619944 | notebook | Interquartile range of preferred depth (cm), 8-depth sessions (Fig S3L) |
| `\statSuppDepthPopKsStat` | `$D = 0.159$` | 0.15884279334335005 | notebook | KS statistic, log preferred depth of 5- vs 8-depth sessions (Fig S3K-L) |
| `\statSuppDepthPopKsPval` | `$p < 0.0001$` | 2.454273634222957e-118 | notebook | KS test p-value, log preferred depth of 5- vs 8-depth sessions (Fig S3K-L) |

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

Source notebook: `figsupp5_rsof.ipynb` (notebook, generated 2026-10-01T18:07:52)

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

Source notebook: `figsupp8_multidepth_receptive_fields.ipynb` (notebook, generated 2026-10-07T17:45:53)

| Macro Name | Formatted Value | Raw Value | Source | Description / Manuscript Context |
| :--- | :---: | :---: | :---: | :--- |
| `\statSuppEightRFCorrNeurons` | `1,009` | 1009 | notebook | Depth-tuned neurons with significant multi-depth RFs in the RF correlation comparison (Fig S8D) |
| `\statSuppEightRFCorrSessions` | `5` | 5 | notebook | Number of multi-depth sessions in the RF correlation comparison (Fig S8D) |
| `\statSuppEightRFCorrMice` | `4` | 4 | notebook | Number of mice in the RF correlation comparison (Fig S8D) |
| `\statSuppEightRFCorrMedianContra` | `$r = 0.334$` | 0.33399676894859875 | notebook | Median Pearson r between single- and multi-depth contralateral RF volumes (Fig S8D) |
| `\statSuppEightRFCorrMedianIpsi` | `$r = 0.031$` | 0.030534735341280744 | notebook | Median Pearson r between single- and multi-depth ipsilateral RF volumes (Fig S8D) |
| `\statSuppEightRFCorrPval` | `$p = 2.80 \times 10^{-182}$` | 2.7965434629083562e-182 | notebook | Mann-Whitney U test p-value, contralateral vs ipsilateral RF correlations (Fig S8D) |
| `\statSuppEightSigRFSessionsSingle` | `84` | 84 | notebook | Number of single-depth sessions (Fig S8E) |
| `\statSuppEightSigRFMedianSingle` | `46.4\%` | 0.464367816091954 | notebook | Median proportion of depth-tuned neurons with significant RFs across single-depth sessions (Fig S8E) |
| `\statSuppEightSigRFSessionsMulti` | `5` | 5 | notebook | Number of multi-depth sessions (Fig S8E) |
| `\statSuppEightSigRFMedianMulti` | `44.9\%` | 0.4492307692307692 | notebook | Median proportion of depth-tuned neurons with significant RFs across multi-depth sessions (Fig S8E) |
| `\statSuppEightSigRFSessionsTotal` | `89` | 89 | notebook | Number of single- and multi-depth sessions (Fig S8E) |
| `\statSuppEightSigRFMedianTotal` | `46.2\%` | 0.46195652173913043 | notebook | Median proportion of depth-tuned neurons with significant RFs across single- and multi-depth sessions (Fig S8E) |
| `\statSuppEightSigRFMiceTotal` | `11` | 11 | notebook | Number of mice, single- and multi-depth sessions (Fig S8E) |
| `\statSuppEightSigRFNeuronsTotal` | `28,966` | 28966 | notebook | Depth-tuned neurons, single- and multi-depth sessions (Fig S8E) |
| `\statSuppEightSigRFSigNeuronsTotal` | `17,954` | 17954 | notebook | Depth-tuned neurons with significant RFs, single- and multi-depth sessions (Fig S8E) |
| `\statSuppEightPeakDepthNeurons` | `1,009` | 1009 | notebook | Depth-tuned neurons with significant multi-depth RFs in the peak depth comparison (Fig S8F) |
| `\statSuppEightPeakDepthR` | `$r = 0.561$` | 0.5606615682902318 | notebook | Spearman correlation of multi- vs single-depth RF peak depth (Fig S8F) |
| `\statSuppEightPeakDepthPval` | `$p = 1.35 \times 10^{-84}$` | 1.3521543842892003e-84 | notebook | p-value of the Spearman correlation of multi- vs single-depth RF peak depth (Fig S8F) |
| `\statSuppEightPeakDepthMedianRatio` | `0.939` | 0.9392100406025342 | notebook | Median ratio of multi- to single-depth RF peak depth (Fig S8F) |
| `\statSuppEightPeakDepthRatioIQR` | `0.669--1.506` | [0.6687090402716057, 1.5061576016632765] | notebook | Interquartile range of the multi- to single-depth RF peak depth ratio (Fig S8F) |

Source notebook: `figsupp9_v1_depth_map.ipynb` (notebook, generated 2026-10-06T18:20:48)

| Macro Name | Formatted Value | Raw Value | Source | Description / Manuscript Context |
| :--- | :---: | :---: | :---: | :--- |
| `\statSuppNineRFNeuronsTotal` | `16,945` | 16945 | notebook | Total depth-selective neurons with significant receptive fields |
| `\statSuppNineRFSessionsTotal` | `84` | 84 | notebook | Total sessions contributing to RF mapping |
| `\statSuppNineNearUncorrected` | `4,125` | 4125 | notebook | Neurons preferring near depths (<20 cm) uncorrected |
| `\statSuppNineMidUncorrected` | `8,088` | 8088 | notebook | Neurons preferring intermediate depths (20-100 cm) uncorrected |
| `\statSuppNineFarUncorrected` | `4,732` | 4732 | notebook | Neurons preferring far depths (>100 cm) uncorrected |
| `\statSuppNineNearCorrected` | `3,313` | 3313 | notebook | Neurons preferring near depths (<20 cm) corrected for viewing angle |
| `\statSuppNineMidCorrected` | `7,275` | 7275 | notebook | Neurons preferring intermediate depths (20-100 cm) corrected for viewing angle |
| `\statSuppNineFarCorrected` | `6,357` | 6357 | notebook | Neurons preferring far depths (>100 cm) corrected for viewing angle |
| `\statSuppNineGradientPvalCorrected` | `$p = 3.95e-09$` | 3.953748301453364e-09 | notebook | Significance of 3D RF depth gradient across visual space (corrected) |
| `\statSuppNineGradientPvalUncorrected` | `$p = 0.0393$` | 0.039301936675758216 | notebook | Significance of depth gradient across visual space (uncorrected) |
| `\statSuppNineAPCorrR` | `$r = -0.140$` | -0.1398648425132868 | notebook | Correlation between preferred depth and anterior-posterior location in V1 |

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
